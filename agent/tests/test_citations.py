"""Unit tests for agent/app/citations.py — citation shape and dedup rules
per docs/contracts.md, with a stubbed Neo4j driver.
"""
import pytest

import app.citations as citations


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


class FakeTx:
    def __init__(self, records):
        self.records = records
        self.queries = []

    def run(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return list(self.records)


class FakeSession:
    def __init__(self, tx):
        self.tx = tx

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute_read(self, work):
        return work(self.tx)


class FakeDriver:
    def __init__(self, records):
        self.records = records
        self.tx = None

    def session(self, default_access_mode=None):
        self.tx = FakeTx(self.records)
        return FakeSession(self.tx)


def record(chunk_id, doc_id, title, doc_type, source_path, section,
           clause="", code_ref=""):
    return FakeRecord({
        "chunk_id": chunk_id, "doc_id": doc_id, "title": title,
        "doc_type": doc_type, "source_path": source_path,
        "section": section, "clause": clause, "code_ref": code_ref,
    })


CITATION_FIELDS = {"doc_id", "title", "doc_type", "source_path",
                   "section", "clause", "code_ref"}


def test_citation_object_shape_per_contract():
    driver = FakeDriver([record(
        "c1", "cps-234-information-security",
        "CPS 234 - Information Security (SYNTHETIC EXTRACT)", "regulatory",
        "data/regulatory/cps-234-information-security.md",
        "Physical security of information assets > Clause 27",
        clause="27")])
    result = citations.build_citations(driver, ["c1"])
    assert len(result) == 1
    citation = result[0]
    assert set(citation) == CITATION_FIELDS
    assert citation["doc_id"] == "cps-234-information-security"
    assert citation["doc_type"] == "regulatory"
    assert citation["clause"] == "27"
    assert citation["code_ref"] == ""  # regulatory citation: empty code_ref
    assert citation["section"] == "Physical security of information assets > Clause 27"


def test_code_citation_carries_code_ref_and_empty_clause():
    driver = FakeDriver([record(
        "c9", "mod-py", "Module", "code", "data/code/mod.py",
        "data/code/mod.py#alpha:7-9", clause="", code_ref="mod.py#alpha:7-9")])
    result = citations.build_citations(driver, ["c9"])
    assert result[0]["clause"] == ""
    assert result[0]["code_ref"] == "mod.py#alpha:7-9"
    assert result[0]["doc_type"] == "code"


def test_null_record_fields_become_empty_strings():
    driver = FakeDriver([record("c1", "", None, "architecture", "p.md", "s")])
    result = citations.build_citations(driver, ["c1"])
    assert result[0]["title"] == ""
    assert result[0]["clause"] == "" and result[0]["code_ref"] == ""


def test_dedup_on_doc_section_clause_code_ref():
    # two different chunks resolve into the same doc/section
    driver = FakeDriver([
        record("c1", "d1", "T", "architecture", "p.md", "Same Section"),
        record("c2", "d1", "T", "architecture", "p.md", "Same Section"),
        record("c3", "d1", "T", "architecture", "p.md", "Other Section"),
    ])
    result = citations.build_citations(driver, ["c1", "c2", "c3"])
    assert len(result) == 2
    assert result[0]["section"] == "Same Section"
    assert result[1]["section"] == "Other Section"


def test_input_ids_deduped_and_graph_ids_skipped():
    driver = FakeDriver([record("c1", "d", "T", "architecture", "p", "s")])
    citations.build_citations(driver, ["c1", "c1", "graph:traversal", ""])
    query, kwargs = driver.tx.queries[0]
    assert kwargs["ids"] == ["c1"]  # deduped, graph: and empty ids filtered


def test_all_ids_filtered_out_means_no_query():
    driver = FakeDriver([record("x", "d", "T", "architecture", "p", "s")])
    assert citations.build_citations(driver, ["graph:traversal", "", None]) == []
    assert driver.tx is None  # the driver was never touched


def test_citation_order_follows_query_result():
    driver = FakeDriver([
        record("c2", "d2", "T2", "code", "p2", "s2", code_ref="r2"),
        record("c1", "d1", "T1", "architecture", "p1", "s1"),
    ])
    result = citations.build_citations(driver, ["c1", "c2"])
    assert [c["doc_id"] for c in result] == ["d2", "d1"]


def test_missing_chunk_ids_in_graph_are_skipped_silently():
    # the query returns nothing for unknown chunk ids
    driver = FakeDriver([])
    assert citations.build_citations(driver, ["ghost-id"]) == []