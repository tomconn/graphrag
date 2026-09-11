"""Unit tests for ingest/pipeline/extract.py — schema validation only.

The LLM boundary (the OpenAI client call inside Extractor.client) is faked
with canned JSON replies; nothing here talks to Ollama or Neo4j.
"""
import json
import logging
from types import SimpleNamespace

import pytest

from pipeline.extract import ExtractionError, Extractor, ExtractedEntity, normalize_name


# ------------------------------------------------------------- test doubles

class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_extractor(schema_path, replies):
    if not isinstance(replies, list):  # a single canned reply
        replies = [replies]
    extractor = Extractor(schema_path=schema_path)
    extractor._client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(replies)))
    return extractor


def make_chunk(text):
    return SimpleNamespace(text=text, section="Some Section", clause="")


def make_doc(doc_id="doc1", doc_type="architecture"):
    return SimpleNamespace(
        doc_id=doc_id, doc_type=doc_type, title="Doc",
        source_path="data/architecture/doc1.md")


def labels_and_keys(entities):
    return sorted((e.label, e.key) for e in entities)


# ------------------------------------------------------------------ happy path

def test_valid_entities_and_relation_pass(schema_path):
    extractor = fake_extractor(schema_path, [{
        "entities": [
            {"label": "System", "name": "Payment Platform"},
            {"label": "System", "name": "Ledger Core"},
        ],
        "relations": [{"type": "DEPENDS_ON", "source_label": "System",
                       "source": "Payment Platform", "target_label": "System",
                       "target": "Ledger Core"}],
    }])
    entities, relations = extractor.extract(make_chunk("text"), make_doc())
    assert labels_and_keys(entities) == [
        ("System", "ledger core"), ("System", "payment platform")]
    assert len(relations) == 1
    rel = relations[0]
    assert rel.rel_type == "DEPENDS_ON"
    assert rel.source_label == "System" and rel.source_key == "payment platform"
    assert rel.target_label == "System" and rel.target_key == "ledger core"


def test_relation_type_is_case_normalized(schema_path):
    extractor = fake_extractor(schema_path, [{
        "entities": [{"label": "System", "name": "A"}, {"label": "System", "name": "B"}],
        "relations": [{"type": "depends_on", "source_label": "System",
                       "source": "A", "target_label": "System", "target": "B"}],
    }])
    _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert relations[0].rel_type == "DEPENDS_ON"


# ------------------------------------------------------------- entity validation

def test_unknown_label_dropped_with_warning(schema_path, caplog):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "Database", "name": "Postgres"}], "relations": []})
    with caplog.at_level(logging.WARNING, logger="pipeline.extract"):
        entities, _ = extractor.extract(make_chunk("text"), make_doc())
    assert entities == []
    assert any("dropping unknown entity label 'Database'" in r.message
               for r in caplog.records)


def test_label_valid_in_schema_but_not_allowed_for_class_dropped(schema_path, caplog):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "Pattern", "name": "Saga"}], "relations": []})
    with caplog.at_level(logging.WARNING, logger="pipeline.extract"):
        entities, _ = extractor.extract(
            make_chunk("text"), make_doc(doc_type="regulatory"))
    assert entities == []
    assert any("not allowed for class 'regulatory'" in r.message
               for r in caplog.records)


def test_malformed_cps_clause_key_dropped(schema_path, caplog):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "CpsClause", "doc": "CPS 230", "number": "third-party"}],
        "relations": []})
    with caplog.at_level(logging.WARNING, logger="pipeline.extract"):
        entities, _ = extractor.extract(
            make_chunk("text"), make_doc(doc_type="regulatory"))
    assert entities == []
    assert any("without a usable key" in r.message for r in caplog.records)


def test_cps_clause_key_built_from_doc_and_number(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "CpsClause", "doc": "CPS 234", "number": "27",
                      "title": "Physical security"}], "relations": []})
    entities, _ = extractor.extract(
        make_chunk("text"), make_doc(doc_type="regulatory"))
    assert len(entities) == 1
    entity = entities[0]
    assert entity.label == "CpsClause"
    assert entity.key == "CPS 234:27"
    assert entity.properties == {"id": "CPS 234:27", "doc": "CPS 234",
                                 "number": "27", "title": "Physical security"}


def test_cps_clause_given_id_backfills_doc_and_number(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "CpsClause", "id": "CPS 230:21", "title": "T"}],
        "relations": []})
    entities, _ = extractor.extract(
        make_chunk("text"), make_doc(doc_type="regulatory"))
    assert entities[0].key == "CPS 230:21"
    assert entities[0].properties["doc"] == "CPS 230"
    assert entities[0].properties["number"] == "21"


def test_name_normalization_merges_case_underscore_whitespace(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [
            {"label": "Pattern", "name": "Transactional_Outbox"},
            {"label": "Pattern", "name": "transactional outbox"},
            {"label": "Pattern", "name": "  Transactional   Outbox "},
        ], "relations": []})
    entities, _ = extractor.extract(make_chunk("text"), make_doc())
    assert labels_and_keys(entities) == [("Pattern", "transactional outbox")]


def test_entity_properties_pruned_to_schema(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "Control", "name": "MFA", "colour": "red",
                      "standard": "CPS 234"}], "relations": []})
    entities, _ = extractor.extract(make_chunk("text"), make_doc())
    assert entities[0].key == "mfa"
    assert set(entities[0].properties) == {"name", "standard"}


def test_id_property_ignored_for_non_cps_entities(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "System", "name": "Core", "id": "SYS-1"}],
        "relations": []})
    entities, _ = extractor.extract(make_chunk("text"), make_doc())
    assert "id" not in entities[0].properties


def test_entity_without_name_dropped(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "System", "name": "   "}], "relations": []})
    entities, _ = extractor.extract(make_chunk("text"), make_doc())
    assert entities == []


def test_key_field_alias_fills_name(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "System", "key": "Payment Platform"}],
        "relations": []})
    entities, _ = extractor.extract(make_chunk("text"), make_doc())
    assert labels_and_keys(entities) == [("System", "payment platform")]


# ---------------------------------------------------------- relation validation

def test_edge_with_missing_endpoint_dropped(schema_path, caplog):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "System", "name": "A"}],
        "relations": [{"type": "DEPENDS_ON", "source_label": "System",
                       "source": "A", "target_label": "System", "target": "  "}],
    })
    with caplog.at_level(logging.WARNING, logger="pipeline.extract"):
        _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert relations == []
    assert any("missing endpoint name(s)" in r.message for r in caplog.records)


def test_unknown_relationship_type_dropped(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "System", "name": "A"}],
        "relations": [{"type": "CONNECTS_TO", "source_label": "System",
                       "source": "A", "target_label": "System", "target": "A"}],
    })
    _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert relations == []


def test_relationship_type_allowed_in_schema_but_not_for_class_dropped(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "System", "name": "A"},
                     {"label": "Pattern", "name": "Saga"}],
        "relations": [{"type": "USES_PATTERN", "source_label": "System",
                       "source": "A", "target_label": "Pattern", "target": "Saga"}],
    })
    _, relations = extractor.extract(
        make_chunk("text"), make_doc(doc_type="regulatory"))
    assert relations == []


def test_edge_endpoints_must_match_schema_arcs(schema_path):
    # DEPENDS_ON only allows System -> System
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "Control", "name": "MFA"},
                     {"label": "System", "name": "Core"}],
        "relations": [{"type": "DEPENDS_ON", "source_label": "Control",
                       "source": "MFA", "target_label": "System", "target": "Core"}],
    })
    _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert relations == []


def test_edge_endpoint_label_not_allowed_for_class_dropped(schema_path, caplog):
    # MITIGATES CodeComponent -> Risk is schema-valid but CodeComponent is not
    # allowed in architecture documents
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "CodeComponent", "name": "validator"},
                     {"label": "Risk", "name": "injection"}],
        "relations": [{"type": "MITIGATES", "source_label": "CodeComponent",
                       "source": "validator", "target_label": "Risk",
                       "target": "injection"}],
    })
    with caplog.at_level(logging.WARNING, logger="pipeline.extract"):
        _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert relations == []
    assert any("endpoint label not allowed" in r.message for r in caplog.records)


def test_cps_clause_endpoint_normalized_from_prose(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "Risk", "name": "unauthorised access"},
                     {"label": "CpsClause", "doc": "CPS 234", "number": "27"}],
        "relations": [{"type": "GOVERNED_BY", "source_label": "Risk",
                       "source": "unauthorised access",
                       "target_label": "CpsClause", "target": "CPS 234 clause 27"}],
    })
    _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert len(relations) == 1
    assert relations[0].target_key == "CPS 234:27"


def test_cps_clause_endpoint_with_bare_id_passes_through(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "Obligation", "name": "72-hour notification"}],
        "relations": [{"type": "DERIVES_FROM", "source_label": "Obligation",
                       "source": "72-hour notification",
                       "target_label": "CpsClause", "target": "CPS 230:21"}],
    })
    _, relations = extractor.extract(
        make_chunk("text"), make_doc(doc_type="regulatory"))
    assert relations[0].target_key == "CPS 230:21"


def test_cps_clause_endpoint_without_standard_name_dropped(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "Risk", "name": "r"}],
        "relations": [{"type": "GOVERNED_BY", "source_label": "Risk",
                       "source": "r", "target_label": "CpsClause",
                       "target": "clause 27"}],
    })
    _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert relations == []


def test_duplicate_relations_deduped(schema_path):
    rel = {"type": "DEPENDS_ON", "source_label": "System", "source": "A",
           "target_label": "System", "target": "B"}
    extractor = fake_extractor(schema_path, {
        "entities": [{"label": "System", "name": "A"}, {"label": "System", "name": "B"}],
        "relations": [dict(rel), dict(rel)]})
    _, relations = extractor.extract(make_chunk("text"), make_doc())
    assert len(relations) == 1


def test_non_dict_payload_items_ignored(schema_path):
    extractor = fake_extractor(schema_path, {
        "entities": ["not a dict", 42],
        "relations": ["nope"],
    })
    entities, relations = extractor.extract(make_chunk("text"), make_doc())
    assert entities == [] and relations == []


# ------------------------------------------------------------------ LLM boundary

def test_empty_chunk_skips_the_llm(schema_path):
    extractor = fake_extractor(schema_path, [])
    entities, relations = extractor.extract(make_chunk("   "), make_doc())
    assert entities == [] and relations == []
    assert extractor._client.chat.completions.calls == []


def test_invalid_json_retried_then_success(schema_path):
    extractor = fake_extractor(schema_path, ["not json", {
        "entities": [{"label": "System", "name": "A"}], "relations": []}])
    entities, _ = extractor.extract(make_chunk("text"), make_doc())
    assert labels_and_keys(entities) == [("System", "a")]
    assert len(extractor._client.chat.completions.calls) == 2


def test_invalid_json_after_max_attempts_raises(schema_path):
    extractor = fake_extractor(schema_path, ["nope", "still nope", "{bad"])
    with pytest.raises(ExtractionError):
        extractor.extract(make_chunk("text"), make_doc())
    assert len(extractor._client.chat.completions.calls) == 3


def test_parse_json_variants(schema_path):
    assert Extractor._parse_json('{"entities": [], "relations": []}') == \
        {"entities": [], "relations": []}
    assert Extractor._parse_json('```json\n{"entities": [1], "relations": []}\n```') == \
        {"entities": [1], "relations": []}
    assert Extractor._parse_json(
        'Here you go: {"entities": [], "relations": [{"type": "X"}]} hope it helps'
    ) == {"entities": [], "relations": [{"type": "X"}]}
    # missing keys default to empty lists
    assert Extractor._parse_json("{}") == {"entities": [], "relations": []}


def test_parse_json_rejects_non_object(schema_path):
    with pytest.raises(ValueError):
        Extractor._parse_json("[1, 2, 3]")


# ------------------------------------------------------------------- normalization

def test_normalize_name_contract():
    assert normalize_name("Transactional_Outbox") == "transactional outbox"
    assert normalize_name("  CPS 234-clause ") == "cps 234 clause"
    assert normalize_name("A__B--C") == "a b c"
    assert normalize_name("") == ""