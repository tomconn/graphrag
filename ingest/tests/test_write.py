"""Unit tests for pipeline/write.py — GraphWriter against a stubbed Neo4j
driver: index/constraint setup, batched upserts, edge provenance and the
missing-endpoint skip contract.
"""
from types import SimpleNamespace

import pytest

import pipeline.write as write_mod
from pipeline.write import BATCH_SIZE, STMT_CHUNK, STMT_DOCUMENT, GraphWriter


class FakeSummary:
    def __init__(self, relationships_created=1, properties_set=5):
        self.counters = SimpleNamespace(
            relationships_created=relationships_created,
            properties_set=properties_set)


class FakeResult:
    def __init__(self, summary):
        self._summary = summary

    def consume(self):
        return self._summary


class FakeSession:
    """Session stand-in; doubles as the tx handed to execute_write."""

    def __init__(self, driver, summary_factory):
        self.driver = driver
        self.queries = []
        self._summary_factory = summary_factory

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, parameters=None, **kwargs):
        # index/constraint statements run without parameters; data writes
        # always parameterize (asserted by the specific upsert tests)
        self.queries.append((query, parameters))
        self.driver.queries.append((query, parameters))
        return FakeResult(self._summary_factory(query))

    def execute_write(self, work):
        self.driver.transactions.append(self)
        work(self)  # tx == session in the stub


class FakeDriver:
    def __init__(self, summary_factory=None):
        self.sessions = []
        self.transactions = []
        self.queries = []
        self.closed = False
        self._summary_factory = summary_factory or (lambda query: FakeSummary())

    def session(self):
        s = FakeSession(self, self._summary_factory)
        self.sessions.append(s)
        return s

    def close(self):
        self.closed = True

    def verify_connectivity(self):
        pass


@pytest.fixture()
def fake_driver(monkeypatch):
    driver = FakeDriver()
    monkeypatch.setattr(
        write_mod.GraphDatabase, "driver",
        staticmethod(lambda uri, auth=None: driver), raising=False)
    return driver


def make_writer(driver=None, dim=4, labels=None):
    default_labels = {"CpsClause": "id", "Pattern": "name"}
    return GraphWriter(
        "bolt://stub:7687", "neo4j", "pw", dim,
        knowledge_labels=default_labels if labels is None else labels)


# ------------------------------------------------------------------ __init__

def test_init_rejects_unreasonable_dim(fake_driver, monkeypatch):
    with pytest.raises(ValueError, match="Unreasonable EMBEDDING_DIM"):
        make_writer(fake_driver, dim=0)
    with pytest.raises(ValueError, match="Unreasonable EMBEDDING_DIM"):
        make_writer(fake_driver, dim=4097)


def test_init_creates_indexes_and_constraints(fake_driver):
    make_writer(fake_driver, dim=384)
    first_session = fake_driver.sessions[0]
    queries = [q for q, _ in first_session.queries]
    assert any("CREATE VECTOR INDEX chunk_embeddings" in q for q in queries)
    assert any("`vector.dimensions`: 384" in q for q in queries)
    assert any("cosine" in q for q in queries)
    assert any("CREATE FULLTEXT INDEX chunk_text_ft" in q for q in queries)
    # constraints session, labels in sorted order
    constraint_session = fake_driver.sessions[1]
    cqueries = [q for q, _ in constraint_session.queries]
    assert cqueries == [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:`CpsClause`) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:`Pattern`) REQUIRE n.name IS UNIQUE",
    ]


def test_init_without_labels_skips_constraints(fake_driver):
    writer = make_writer(fake_driver, labels={})
    # only the index session ran; no constraint statements
    assert all("CREATE CONSTRAINT" not in q for q, _ in fake_driver.queries)
    assert writer.knowledge_labels == {}


def test_close_closes_driver(fake_driver):
    writer = make_writer(fake_driver)
    writer.close()
    assert fake_driver.closed


# ---------------------------------------------------------------- batching

def test_run_batch_splits_into_transactions(fake_driver, monkeypatch):
    monkeypatch.setattr(write_mod, "BATCH_SIZE", 2)
    writer = make_writer(fake_driver, labels={})
    statements = [("RETURN 1", {"i": i}) for i in range(5)]
    writer._run_batch(statements)
    # 5 statements / batch 2 -> 3 transactions (2 + 2 + 1)
    assert len(fake_driver.transactions) == 3
    assert sum(len(t.queries) for t in fake_driver.transactions) == 5


# --------------------------------------------------------- replace_document

def make_doc():
    return SimpleNamespace(
        doc_id="doc-1", doc_type="regulatory", title="Doc One",
        source_path="data/regulatory/doc-1.md")


def make_chunk(cid, text="body"):
    return SimpleNamespace(
        id=cid, text=text, embedding=[0.1, 0.2], section="s", clause="c",
        code_ref="", doc_type="regulatory")


def test_replace_document_runs_cleanup_then_upserts(fake_driver):
    writer = make_writer(fake_driver, labels={})
    doc, chunks = make_doc(), [make_chunk("d-0"), make_chunk("d-1")]
    writer.replace_document(doc, chunks)

    # the LAST session before the batch upserts ran the cleanup transaction
    cleanup_session = fake_driver.sessions[1]  # [0] is the index session
    cleanup_queries = [q for q, _ in cleanup_session.queries]
    assert any("DETACH DELETE c" in q for q in cleanup_queries)
    assert any("WHERE r.source_document = $id DELETE r" in q for q in cleanup_queries)
    # cleanup params carry the doc id
    assert cleanup_session.queries[1][1] == {"id": "doc-1"}

    assert any(q == STMT_DOCUMENT for q, _ in fake_driver.queries)
    doc_params = next(p for q, p in fake_driver.queries if q == STMT_DOCUMENT)
    assert doc_params["props"]["title"] == "Doc One"
    assert doc_params["props"]["source_path"] == "data/regulatory/doc-1.md"
    chunk_statements = [q for q, _ in fake_driver.queries if q == STMT_CHUNK]
    assert len(chunk_statements) == 2


def test_replace_document_with_no_chunks_still_upserts_doc(fake_driver):
    writer = make_writer(fake_driver, labels={})
    writer.replace_document(make_doc(), [])
    assert any(q == STMT_DOCUMENT for q, _ in fake_driver.queries)
    assert not any(q == STMT_CHUNK for q, _ in fake_driver.queries)


# ------------------------------------------------------------ knowledge layer

def make_entity(label, key):
    return SimpleNamespace(label=label, key=key, properties={"key": key, "name": key})


def test_write_entities_returns_count_and_interpolates_label(fake_driver):
    writer = make_writer(fake_driver, labels={})
    entities = [make_entity("CpsClause", "doc:1"), make_entity("CpsClause", "doc:2"),
                make_entity("Pattern", "p1")]
    assert writer.write_entities("doc-1", entities) == 3
    entity_queries = [q for q, _ in fake_driver.queries if "MERGE (e:" in q]
    assert len(entity_queries) == 3
    assert "MERGE (e:`CpsClause` {key: $key})" in entity_queries[0]


def make_relation(slabel, skey, rel_type, tlabel, tkey):
    return SimpleNamespace(
        source_label=slabel, source_key=skey, rel_type=rel_type,
        target_label=tlabel, target_key=tkey)


def test_write_edges_returns_edges_and_interpolates(fake_driver):
    writer = make_writer(fake_driver, labels={})
    rels = [make_relation("CpsClause", "doc:1", "REQUIRES", "Pattern", "p1")]
    count, edges = writer.write_edges("doc-1", "doc-1#0", rels)
    assert (count, len(edges)) == (1, 1)
    query, params = edges[0]
    assert "MATCH (a:`CpsClause` {key: $skey})" in query
    assert "MERGE (a)-[r:`REQUIRES`]->(b)" in query
    assert params["doc_id"] == "doc-1" and params["chunk_id"] == "doc-1#0"


def test_write_edges_batches_at_batch_size(fake_driver, monkeypatch):
    monkeypatch.setattr(write_mod, "BATCH_SIZE", 2)
    writer = make_writer(fake_driver, labels={})
    rels = [make_relation("Pattern", f"p{i}", "RELATES", "Pattern", f"q{i}")
            for i in range(5)]
    count, _ = writer.write_edges("doc-1", "doc-1#0", rels)
    assert count == 5
    # one transaction per batch of 2 -> 3 execute_write transactions
    assert len(fake_driver.transactions) == 3


def test_write_edges_logs_skipped_missing_endpoint(monkeypatch, caplog):
    """counters all zero = the MERGE matched nothing -> skip + warn."""
    driver = FakeDriver(summary_factory=lambda q: FakeSummary(0, 0))
    monkeypatch.setattr(
        write_mod.GraphDatabase, "driver",
        staticmethod(lambda uri, auth=None: driver), raising=False)
    writer = make_writer(driver, labels={})
    with caplog.at_level("WARNING"):
        writer.write_edges(
            "doc-1", "doc-1#0",
            [make_relation("CpsClause", "ghost", "REQUIRES", "Pattern", "p1")])
    assert any("endpoint entity was not found" in rec.message
               for rec in caplog.records)
    assert driver.sessions, "edge statements should still have been attempted"


def test_write_edges_no_warning_when_merged(fake_driver, caplog):
    writer = make_writer(fake_driver, labels={})
    with caplog.at_level("WARNING"):
        writer.write_edges(
            "doc-1", "doc-1#0",
            [make_relation("CpsClause", "doc:1", "REQUIRES", "Pattern", "p1")])
    assert not any("endpoint entity was not found" in rec.message
                   for rec in caplog.records)


def test_batch_size_constant_is_500():
    # docs/contracts.md pins batches of ~500
    assert BATCH_SIZE == 500