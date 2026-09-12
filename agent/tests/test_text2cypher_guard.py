"""Unit tests for agent/app/text2cypher.py — the write-keyword guard, prompt
plumbing and the run loop with a stubbed Neo4j driver and LLM.
"""
import json
from pathlib import Path

import pytest

import neo4j

from app import text2cypher

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------- the guard

def test_read_statements_pass():
    statements = [
        "MATCH (n:Chunk) RETURN n LIMIT 5",
        "MATCH (s:System)-[r:DEPENDS_ON]->(t:System) RETURN s.name, t.name, r.source_chunk",
        "MATCH (c:Chunk {clause: '27'}) RETURN c.text",
        "MATCH (n:Pattern) WHERE n.name CONTAINS 'Saga' RETURN count(n)",
    ]
    for statement in statements:
        assert text2cypher._clean_statement(statement) == statement


def test_every_write_keyword_is_rejected():
    writes = [
        "CREATE (n:Thing)",
        "MERGE (n:Thing {id: 1})",
        "MATCH (n) DELETE n",
        "MATCH (n) SET n.flag = true",
        "MATCH (n) DETACH DELETE n",
        "DROP INDEX my_index",
        "MATCH (n) REMOVE n.flag",
        "CALL db.index.fulltext.queryNodes('chunk_text_ft', 'risk')",
    ]
    for statement in writes:
        with pytest.raises(ValueError, match="write keyword"):
            text2cypher._clean_statement(statement)


def test_write_keyword_rejection_is_case_insensitive():
    for statement in ("create (n)", "MeRgE (n)", "match (n) delete n return n"):
        with pytest.raises(ValueError, match="write keyword"):
            text2cypher._clean_statement(statement)


def test_load_and_foreach_are_rejected():
    """LOAD CSV (SSRF via arbitrary URL) and FOREACH (write loop) are not in
    the original keyword set — they must be rejected too."""
    # LOAD CSV always carries a URL literal, so the URL check (which runs
    # first on the unmasked statement) rejects it before the keyword scan.
    for statement in (
        "LOAD CSV FROM 'https://evil.example/x.csv' AS row RETURN row",
        "LOAD CSV WITH HEADERS FROM 'http://internal/x' AS row MATCH (n) RETURN n",
    ):
        with pytest.raises(ValueError, match="URL literal"):
            text2cypher._clean_statement(statement)
    for statement in (
        "MATCH (n) FOREACH (x IN range(1,5) | CREATE (:Thing))",
        "MATCH (n) FOREACH (_ IN n.items | SET n.done = true)",
    ):
        with pytest.raises(ValueError, match="write keyword"):
            text2cypher._clean_statement(statement)


def test_url_literal_is_rejected():
    """A URL literal is a data-exfiltration / SSRF vector even in an
    otherwise read-only statement, so it is rejected on its own."""
    for statement in (
        "MATCH (n) RETURN 'https://attacker.example/collect'",
        "LOAD CSV FROM 'http://169.254.169.254/latest' AS row RETURN row",
        'MATCH (n) WHERE n.url = "https://x.example" RETURN n',
        "RETURN 'HTTPS://UPPERCASE.EXAMPLE' + n.name",
    ):
        with pytest.raises(ValueError, match="URL literal"):
            text2cypher._clean_statement(statement)


def test_word_boundary_aware_no_false_positives():
    """Property names and labels that merely contain keyword letters must
    pass (the regex is \\b-anchored)."""
    statements = [
        "MATCH (s:Settings) WHERE s.updated_at > 10 RETURN s",
        "MATCH (n:Pattern) RETURN n.name",
        "MATCH (c:Chunk) WHERE c.text = 'offset value' RETURN c",
        "MATCH (o:Obligation) WHERE o.status = 'SETTLED' RETURN o",
        "MATCH (n) WHERE n.name = 'reset_at' RETURN n",
    ]
    for statement in statements:
        assert text2cypher._clean_statement(statement) == statement


def test_keyword_inside_a_string_literal_is_not_rejected():
    """The guard masks string literals before scanning, so a keyword inside a
    quoted value is not a false positive."""
    statement = "MATCH (n) WHERE n.note = 'please CALL back' RETURN n"
    assert text2cypher._clean_statement(statement) == statement


def test_write_keyword_outside_a_literal_is_still_rejected():
    for statement in (
        "MERGE (n) RETURN n",
        "MATCH (n) SET n.x = 'SETTLED'",
        "MATCH (n) DELETE n",
        "CALL db.indexes()",
        'MATCH (n) WHERE n.note = "merge me" MERGE (m)',
    ):
        with pytest.raises(ValueError, match="write keyword"):
            text2cypher._clean_statement(statement)


def test_trailing_semicolon_and_fences_are_stripped():
    assert text2cypher._clean_statement("MATCH (n) RETURN n;") == "MATCH (n) RETURN n"
    assert text2cypher._clean_statement(
        "```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"


def test_empty_statement_rejected():
    for raw in ("", "   ", "```\n```", ";"):
        with pytest.raises(ValueError):
            text2cypher._clean_statement(raw)


# ------------------------------------------------ duplicate RETURN column repair

def test_duplicate_return_aliases_are_renamed():
    """Neo4j rejects duplicate result columns; the second and later
    occurrences are renamed so the first keeps its name."""
    statement = (
        "MATCH (o:Obligation)<-[r1]-(c:CpsClause)<-[r2]-(s:System) "
        "RETURN 'Obligation' AS kind, o.name AS name, s.name AS implementer, "
        "r1.source_chunk AS source_chunk, r2.source_chunk AS implementer, "
        "r2.source_document AS source_document LIMIT 10"
    )
    fixed = text2cypher._clean_statement(statement)
    assert "s.name AS implementer," in fixed
    assert "r2.source_chunk AS implementer_2" in fixed
    assert "r1.source_chunk AS source_chunk" in fixed


def test_dedupe_triple_duplicate_and_avoids_existing_names():
    statement = "MATCH (n) RETURN n.x AS a, n.y AS a, n.z AS a, n.w AS a_2"
    fixed = text2cypher._dedupe_return_aliases(statement)
    assert fixed == (
        "MATCH (n) RETURN n.x AS a, n.y AS a_2, n.z AS a_3, n.w AS a_2_2"
    )


def test_dedupe_ignores_commas_and_literals():
    """Commas and 'AS'-looking text inside literals or function calls are not
    item boundaries; a unique-alias clause passes through unchanged."""
    statement = (
        "MATCH (n) WHERE n.note = 'a, b' "
        "RETURN collect(n.tag) AS tags, 'AS x' AS note, n LIMIT 5"
    )
    assert text2cypher._dedupe_return_aliases(statement) == statement


def test_dedupe_scans_only_the_last_return_clause():
    statement = (
        "MATCH (a) RETURN a.x AS kind UNION MATCH (b) RETURN b.y AS kind"
    )
    assert text2cypher._dedupe_return_aliases(statement) == statement


def test_dedupe_handles_multiline_and_order_by():
    statement = (
        "MATCH (n) RETURN n.a AS x,\n"
        "  n.b AS x\n"
        "ORDER BY x LIMIT 5"
    )
    fixed = text2cypher._dedupe_return_aliases(statement)
    assert "n.b AS x_2" in fixed


def test_run_text2cypher_repairs_duplicate_alias_without_llm_retry():
    """A duplicate-alias statement is fixed deterministically and runs on the
    first attempt — no second LLM call is spent on it."""
    driver = FakeDriver(rows=[{"source_chunk": "c1"}])
    result, calls = run_with_replies(
        ["MATCH (n) RETURN n.x AS k, n.y AS k"], driver)
    assert result is not None
    assert result["cypher"] == "MATCH (n) RETURN n.x AS k, n.y AS k_2"
    assert result["rows"] == [{"source_chunk": "c1"}]
    assert len(calls) == 1


def test_empty_output_retry_prompt_is_actionable():
    driver = FakeDriver(rows=[{"x": 1}])
    result, calls = run_with_replies(
        ["   ", "MATCH (n) RETURN n"], driver)
    assert result is not None
    assert "contained no Cypher statement" in calls[1]
    assert "ONLY the Cypher query" in calls[1]


def test_retry_prompt_error_is_trimmed():
    driver = FakeDriver(rows=[{"x": 1}])

    class VerboseErrorDriver(FakeDriver):
        def session(self, default_access_mode=None):
            raise RuntimeError("x" * 1000)

    result, calls = run_with_replies(
        ["MATCH (bad", "MATCH (n) RETURN n", "MATCH (n) RETURN n"],
        VerboseErrorDriver())
    assert result is None
    assert len(calls) == text2cypher.MAX_ATTEMPTS
    assert calls[1].count("x" * 1000) == 0
    assert "[truncated]" in calls[1]


# ------------------------------------------------------- run_text2cypher (fakes)

class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeResult:
    def consume(self):
        return None


class FakeTx:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def run(self, query, parameters=None, **kwargs):
        # Mirror the real driver signature: a kwarg named `query` collides
        # with this first argument and must raise (see BM25 retriever bug).
        if parameters is None:
            parameters = kwargs
        else:
            assert not kwargs, "cannot pass both parameters and kwargs"
        self.queries.append((query, parameters))
        if query.startswith("EXPLAIN"):
            return FakeResult()
        return [FakeRecord(row) for row in self.rows]


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
    def __init__(self, rows=None):
        self.rows = rows or []
        self.tx = None

    def session(self, default_access_mode=None):
        assert default_access_mode == neo4j.READ_ACCESS
        self.tx = FakeTx(self.rows)
        return FakeSession(self.tx)


def run_with_replies(replies, driver):
    calls = []
    pending = list(replies)

    def complete_fn(prompt):
        calls.append(prompt)
        return pending.pop(0)

    result = text2cypher.run_text2cypher(
        driver, "question", {"knowledge_layer": {}}, complete_fn)
    return result, calls


def test_run_text2cypher_success_collects_chunk_refs():
    rows = [{"source_chunk": "c1", "a": {"source_chunk": "c2"},
             "a_dup": {"source_chunk": "c1"}}]
    driver = FakeDriver(rows=rows)
    result, calls = run_with_replies(["MATCH (n) RETURN n"], driver)
    assert result is not None
    assert result["cypher"] == "MATCH (n) RETURN n"
    assert result["rows"] == rows
    assert result["chunk_refs"] == ["c1", "c2"]  # deduped, order preserved
    assert driver.tx.queries[0][0] == "EXPLAIN MATCH (n) RETURN n"
    assert driver.tx.queries[1][0] == "MATCH (n) RETURN n"
    assert len(calls) == 1


def test_run_text2cypher_retries_on_write_keyword_then_gives_up():
    driver = FakeDriver(rows=[{"x": 1}])
    result, calls = run_with_replies(
        ["CREATE (n)", "MERGE (n)", "SET x = 1"], driver)
    assert result is None
    assert len(calls) == text2cypher.MAX_ATTEMPTS
    assert driver.tx is None  # the driver was never invoked


def test_run_text2cypher_retries_driver_error_then_succeeds():
    class FlakyDriver(FakeDriver):
        def __init__(self):
            super().__init__(rows=[{"id": "c9"}])
            self.attempts = 0

        def session(self, default_access_mode=None):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("Invalid syntax near RETURN")
            return super().session(default_access_mode)

    driver = FlakyDriver()
    result, calls = run_with_replies(
        ["MATCH (bad syntax", "MATCH (n) RETURN n"], driver)
    assert result is not None
    assert result["cypher"] == "MATCH (n) RETURN n"
    # the second prompt carries the previous statement and the driver error
    assert "MATCH (bad syntax" in calls[1]
    assert "Invalid syntax near RETURN" in calls[1]


def test_run_text2cypher_truncates_rows():
    rows = [{"i": i} for i in range(30)]
    result, _ = run_with_replies(["MATCH (n) RETURN n"], FakeDriver(rows=rows))
    assert len(result["rows"]) == text2cypher.MAX_ROWS


def test_rows_to_text_truncates_each_row():
    rows = [{"text": "y" * 2000}]
    lines = text2cypher.rows_to_text(rows).split("\n")
    assert len(lines) == 1
    assert len(lines[0]) == text2cypher.MAX_ROW_TEXT_CHARS


def test_rows_to_text_is_json_per_line():
    rows = [{"a": 1, "b": "two"}]
    assert json.loads(text2cypher.rows_to_text(rows)) == rows[0]


# ------------------------------------------------------------- prompt and schema

def test_render_knowledge_layer(schema):
    rendered = text2cypher.render_knowledge_layer(schema)
    assert "NODE LABELS:" in rendered
    assert ":System" in rendered and ":CpsClause" in rendered
    assert "RELATIONSHIPS:" in rendered
    assert "- (: System)-[:DEPENDS_ON]->(: System)" in rendered
    assert "PROVENANCE:" in rendered
    assert "source_chunk" in rendered and "source_document" in rendered


def test_build_prompt_includes_question_and_retry_context(schema):
    rendered = text2cypher.render_knowledge_layer(schema)
    prompt = text2cypher._build_prompt("how do systems relate?", rendered)
    assert "QUESTION: how do systems relate?" in prompt
    assert "never use CREATE" in prompt
    retry = text2cypher._build_prompt("q", rendered, "MATCH (x", "Syntax error")
    assert "PREVIOUS ATTEMPT FAILED." in retry
    assert "MATCH (x" in retry and "Syntax error" in retry


def test_load_schema_reads_the_yaml_file(schema):
    loaded = text2cypher.load_schema(str(REPO_ROOT / "schema" / "graph_schema.yaml"))
    assert loaded == schema

def test_build_prompt_warns_about_unlisted_endpoint_pairs(schema):
    """The prompt must forbid inventing edges between labels no listed
    relationship pair connects (e.g. System to Obligation), which the model
    otherwise hallucinates and the graph can never match."""
    prompt = text2cypher._build_prompt(
        "which systems implement the backup obligation?",
        text2cypher.render_knowledge_layer(schema))
    assert "connects ONLY the endpoint pairs listed" in prompt
    assert "do not invent an edge" in prompt
