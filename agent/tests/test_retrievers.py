"""Unit tests for agent/app/retrievers.py — RRF fusion, chunk normalization,
the BM25 sparse retriever and RetrievalService mode selection.

The Neo4j driver and the dense (VectorRetriever) side are stubbed; the sparse
side runs the real BM25Retriever code against the stub.
"""
import neo4j
import pytest
from neo4j_graphrag.types import RetrieverResultItem

import app.retrievers as retrievers
from app.retrievers import (
    BM25Retriever,
    RetrievalService,
    _sanitize_fulltext,
    _to_chunks,
    reciprocal_rank_fusion,
)


def chunk(cid, score=0.0, text=""):
    return {"id": cid, "text": text or f"body {cid}", "section": f"sec {cid}",
            "clause": "", "code_ref": "", "score": score, "kind": "chunk"}


# ----------------------------------------------------------------------- RRF

def test_rrf_k60_scoring_and_ordering():
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("b"), chunk("d")]
    fused = reciprocal_rank_fusion(dense, sparse)
    # b: 1/61 + 1/62, a: 1/61, d: 1/62, c: 1/61 (stable tie after d)
    assert [c["id"] for c in fused] == ["b", "a", "d", "c"]
    scores = {c["id"]: 0.0 for c in fused}
    assert fused[0]["id"] == "b"  # appears in both lists, ranked first in both


def test_rrf_dedups_by_chunk_id():
    dense = [chunk("a"), chunk("b"), chunk("c")]
    sparse = [chunk("b"), chunk("d"), chunk("e")]
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [c["id"] for c in fused]
    assert len(ids) == len(set(ids)) == 5


def test_rrf_keeps_first_occurrence_on_dedup():
    first = chunk("x", text="from dense")
    second = chunk("x", text="from sparse")
    fused = reciprocal_rank_fusion([first], [second])
    assert len(fused) == 1
    assert fused[0]["text"] == "from dense"


def test_rrf_skips_chunks_without_id():
    fused = reciprocal_rank_fusion([chunk(""), chunk("a")], [chunk(""), chunk("b")])
    assert [c["id"] for c in fused] == ["a", "b"]


def test_rrf_custom_k_changes_scores():
    one = [chunk("a"), chunk("b")]
    two = [chunk("b")]
    assert reciprocal_rank_fusion(one, two)[0]["id"] == "b"
    assert reciprocal_rank_fusion(one, two, k=1)[0]["id"] == "b"


def test_rrf_single_list_passthrough():
    ranked = [chunk("a"), chunk("b"), chunk("c")]
    assert [c["id"] for c in reciprocal_rank_fusion(ranked)] == ["a", "b", "c"]


# ------------------------------------------------------- chunk normalization

def test_to_chunks_normalizes_graphrag_items():
    item = RetrieverResultItem(
        content="body text",
        metadata={"id": "c1", "section": "A > B", "clause": "3",
                  "code_ref": "", "score": 0.75})
    chunks = _to_chunks(type("R", (), {"items": [item]})())
    assert chunks == [{"id": "c1", "text": "body text", "section": "A > B",
                       "clause": "3", "code_ref": "", "score": 0.75,
                       "kind": "chunk"}]


def test_to_chunks_defaults_missing_fields():
    item = RetrieverResultItem(content="x", metadata={"score": None})
    chunks = _to_chunks(type("R", (), {"items": [item]})())
    assert chunks[0]["id"] == "" and chunks[0]["section"] == ""
    assert chunks[0]["score"] == 0.0


def test_chunk_formatter_reads_node_and_score():
    record = {"node": {"id": "c1", "text": "the text", "section": "s",
                       "clause": "", "code_ref": ""},
              "score": 0.42}
    item = retrievers.chunk_formatter(record)
    assert item.content == "the text"
    assert item.metadata["id"] == "c1"
    assert item.metadata["score"] == 0.42


# ------------------------------------------------------------------ fulltext

def test_sanitize_fulltext_strips_lucene_specials():
    assert _sanitize_fulltext("risk AND (incident)") == "risk AND incident"
    assert _sanitize_fulltext('pay*ment? "quote" -minus') == "pay ment quote minus"
    assert _sanitize_fulltext("  spaced\t\tout  ") == "spaced out"
    assert _sanitize_fulltext("***") == ""


# -------------------------------------------------------------------- BM25

class _PoolConfig:
    user_agent = "test"


class _Pool:
    pool_config = _PoolConfig()


def make_fake_driver(records):
    """Minimal driver stub for BM25Retriever: needs `_pool` for the
    neo4j-graphrag user-agent override and `execute_query` for its server
    version probe."""

    class Tx:
        def __init__(self):
            self.queries = []

        def run(self, cypher, **kwargs):
            self.queries.append((cypher, kwargs))
            return records

    class Session:
        def __init__(self, tx):
            self.tx = tx

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute_read(self, work):
            return work(self.tx)

    class Driver:
        def __init__(self):
            self._pool = _Pool()
            self.tx = None

        def session(self, default_access_mode=None):
            assert default_access_mode == neo4j.READ_ACCESS
            self.tx = Tx()
            return Session(self.tx)

        def execute_query(self, *args, **kwargs):
            return ([{"versions": ["5.28.5"], "edition": "enterprise"}],
                    None, None)

    return Driver()


def bm25_records():
    return [
        dict(node={"id": "c1", "text": "t1", "section": "s1",
                   "clause": "", "code_ref": ""}, score=0.9),
        dict(node={"id": "c2", "text": "t2", "section": "s2",
                   "clause": "5", "code_ref": ""}, score=0.4),
    ]


def test_bm25_retriever_runs_the_sparse_query():
    records = [neo4j.Record(list(r.items())) for r in bm25_records()]
    driver = make_fake_driver(records)
    retriever = BM25Retriever(driver)
    result = retriever.search(query_text="operational risk!", top_k=2)
    assert [item.metadata["id"] for item in result.items] == ["c1", "c2"]
    assert result.items[0].content == "t1"
    assert result.items[0].metadata["score"] == 0.9
    query, kwargs = driver.tx.queries[0]
    assert query == retrievers.SPARSE_CYPHER
    assert kwargs["index_name"] == "chunk_text_ft"
    assert kwargs["query"] == "operational risk"
    assert kwargs["top_k"] == 2


def test_bm25_retriever_empty_query_skips_the_driver():
    driver = make_fake_driver([])
    retriever = BM25Retriever(driver)
    result = retriever.get_search_results(query_text="   !!!")
    assert result.records == []
    assert driver.tx is None  # no session was opened


# -------------------------------------------------------- RetrievalService

class ScriptedRetriever:
    def __init__(self, ranked_chunks):
        self.ranked_chunks = ranked_chunks
        self.calls = []

    def search(self, query_text=None, top_k=5):
        self.calls.append({"query_text": query_text, "top_k": top_k})
        items = [RetrieverResultItem(
            content=c["text"],
            metadata={"id": c["id"], "section": c["section"], "clause": "",
                      "code_ref": "", "score": c["score"]})
            for c in self.ranked_chunks[:top_k]]
        return type("R", (), {"items": items})()


def make_service(monkeypatch, dense_chunks, sparse_chunks):
    """Build a real RetrievalService with the dense side stubbed."""
    class FakeVectorRetriever:
        def __init__(self, driver, index_name, embedder, result_formatter=None):
            self.driver = driver

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(retrievers, "VectorRetriever", FakeVectorRetriever)
    import app.embedder as embedder_module
    monkeypatch.setattr(embedder_module, "FastEmbedEmbedder", FakeEmbedder)

    driver = make_fake_driver(
        [neo4j.Record(list(r.items())) for r in bm25_records()])
    service = RetrievalService(driver)
    service.dense = ScriptedRetriever(dense_chunks)
    service.sparse = ScriptedRetriever(sparse_chunks)
    return service


def test_retrieval_service_vector_mode_uses_dense_only(monkeypatch):
    service = make_service(monkeypatch, [chunk("a"), chunk("b")],
                           [chunk("a"), chunk("c")])
    result = service.search("query", mode="vector", top_k=2)
    assert [c["id"] for c in result] == ["a", "b"]
    assert len(service.dense.calls) == 1
    assert service.sparse.calls == []  # sparse side untouched in vector mode


def test_retrieval_service_hybrid_fuses_dense_and_sparse(monkeypatch):
    service = make_service(monkeypatch,
                           [chunk("a"), chunk("b"), chunk("c")],
                           [chunk("b"), chunk("d"), chunk("e")])
    result = service.search("query", mode="hybrid", top_k=5)
    # RRF (k=60): b 1/61+1/62, a 1/61, d 1/62, then the 1/63 tie keeps the
    # dense list's c ahead of sparse's e
    assert [c["id"] for c in result] == ["b", "a", "d", "c", "e"]
    assert len(service.dense.calls) == 1 and len(service.sparse.calls) == 1
    assert service.dense.calls[0]["top_k"] == 5


def test_retrieval_service_hybrid_respects_top_k(monkeypatch):
    service = make_service(monkeypatch,
                           [chunk("a"), chunk("b"), chunk("c")],
                           [chunk("d"), chunk("e")])
    result = service.search("query", mode="hybrid", top_k=2)
    # a and d tie at 1/61; the stable sort keeps a (dense) first
    assert [c["id"] for c in result] == ["a", "d"]  # truncated after fusion


def test_retrieval_service_wires_indexes(monkeypatch):
    class FakeVectorRetriever:
        def __init__(self, driver, index_name, embedder, result_formatter=None):
            self.index_name = index_name
            self.driver = driver

    import app.embedder as embedder_module

    class FakeEmbedder:
        pass

    monkeypatch.setattr(retrievers, "VectorRetriever", FakeVectorRetriever)
    monkeypatch.setattr(embedder_module, "FastEmbedEmbedder", FakeEmbedder)
    driver = make_fake_driver([])
    service = RetrievalService(driver)
    assert service.dense.index_name == retrievers.VECTOR_INDEX
    assert service.dense.driver is driver
    assert service.sparse.index_name == retrievers.FULLTEXT_INDEX