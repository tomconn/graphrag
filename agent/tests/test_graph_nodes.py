"""Unit tests for agent/app/graph.py — the helper functions, each node in
isolation (LLM and retrieval stubbed), the routing functions, step_detail,
and one end-to-end invoke through the compiled graph.
"""
from types import SimpleNamespace

import pytest

import app.citations as citations
import app.llm as llm
import app.text2cypher as text2cypher
from app import graph as graph_mod
from app.graph import (
    MAX_CONTEXT_CHUNKS,
    _cite_node,
    _first_token,
    _merge_chunks,
    _render_context,
    _rewrite_node,
    _route_after_retrieve,
    _route_after_sufficiency,
    _route_node,
    _retrieve_node,
    _sufficiency_node,
    _synthesize_node,
    _traverse_node,
    build_graph,
    step_detail,
)
from app.retrievers import DEFAULT_TOP_K


def chunk(cid, text="body", score=0.5, **extra):
    base = {"id": cid, "text": text, "section": f"sec-{cid}", "clause": "",
            "code_ref": "", "score": score, "kind": "chunk"}
    base.update(extra)
    return base


class FakeService:
    def __init__(self, chunks=None):
        self.chunks = chunks or [chunk("c1"), chunk("c2")]
        self.search_calls = []
        self.dense = SimpleNamespace(driver=SimpleNamespace())

    def search(self, query, mode="hybrid", top_k=DEFAULT_TOP_K):
        self.search_calls.append((query, mode, top_k))
        return self.chunks


@pytest.fixture(autouse=True)
def no_stream_writer(monkeypatch):
    # _synthesize_node is called directly (outside a LangGraph runtime)
    monkeypatch.setattr(graph_mod, "get_stream_writer", lambda: None)


# ------------------------------------------------------------------ helpers

def test_first_token_strips_punctuation():
    assert _first_token("Sufficient, because the context", "x") == "sufficient"
    assert _first_token('  "insufficient: missing dates"', "x") == "insufficient"


def test_first_token_defaults():
    assert _first_token("", "fallback") == "fallback"
    assert _first_token("   ", "fallback") == "fallback"
    assert _first_token(",,, !!", "fallback") == "fallback"


def test_merge_chunks_dedups_keeping_first():
    merged = _merge_chunks([chunk("a", text="first")], [chunk("a", text="second"),
                                                        chunk("b")])
    assert [c["id"] for c in merged] == ["a", "b"]
    assert merged[0]["text"] == "first"


def test_merge_chunks_caps_context_size():
    chunks = [chunk(f"c{i}") for i in range(MAX_CONTEXT_CHUNKS + 5)]
    assert len(_merge_chunks([], chunks)) == MAX_CONTEXT_CHUNKS


def test_render_context_prefers_section_then_coderef_then_id():
    context = [
        chunk("a", text="A" * 10, section="Scope"),
        chunk("b", text="B", section="", code_ref="app/x.py:10"),
        chunk("c", text="C", section="", code_ref=""),
    ]
    rendered = _render_context(context, limit=5)
    assert "[1] (Scope) AAAAA" in rendered
    assert "[2] (app/x.py:10) B" in rendered
    assert "[3] (c) C" in rendered


# -------------------------------------------------------------------- nodes

def test_route_node_known_and_unknown_routes(monkeypatch):
    monkeypatch.setattr(
        llm, "complete",
        lambda prompt, temperature=0.2: "relationship because multi-hop")
    state = _route_node({"question": "How do A and B relate?"})
    assert state["route"] == "relationship"
    assert "multi-hop" in state["route_rationale"]

    monkeypatch.setattr(llm, "complete", lambda prompt, temperature=0.2: "gibberish")
    assert _route_node({"question": "q"})["route"] == "lookup"  # falls back


def test_route_node_empty_output_falls_back(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda prompt, temperature=0.2: "")
    assert _route_node({"question": "q"})["route"] == "lookup"


def test_rewrite_node_strips_quotes(monkeypatch):
    monkeypatch.setattr(
        llm, "complete", lambda prompt, temperature=0.2: ' "cleaned query" ')
    assert _rewrite_node({"question": "raw"})["rewritten"] == "cleaned query"


def test_rewrite_node_keeps_question_when_model_returns_empty(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda prompt, temperature=0.2: "   ")
    assert _rewrite_node({"question": "original"})["rewritten"] == "original"


def test_retrieve_node_merges_and_counts_iterations():
    service = FakeService([chunk("c2", text="new")])
    state = {"question": "q", "rewritten": "rq", "context": [chunk("c1")],
             "iterations": 0, "sufficiency_note": "need more on X"}
    update = _retrieve_node(state, service)
    assert service.search_calls == [("rq", "hybrid", DEFAULT_TOP_K)]
    assert [c["id"] for c in update["context"]] == ["c1", "c2"]
    assert update["chunk_ids"] == ["c2"]
    assert update["iterations"] == 1
    assert "need more on X" in update["notes"]


def test_retrieve_node_uses_question_without_rewrite():
    service = FakeService([])
    _retrieve_node({"question": "raw question"}, service)
    assert service.search_calls[0][0] == "raw question"


def test_retrieve_node_no_duplicate_note():
    service = FakeService([])
    update = _retrieve_node(
        {"question": "q", "notes": ["already there"],
         "sufficiency_note": "already there"}, service)
    assert update["notes"].count("already there") == 1


def test_route_after_retrieve():
    assert _route_after_retrieve({"route": "relationship"}) == "traverse"
    assert _route_after_retrieve(
        {"route": "compliance-mapping"}) == "traverse"
    assert _route_after_retrieve({"route": "lookup"}) == "sufficiency"
    # vector mode skips traversal even for relationship questions
    assert _route_after_retrieve(
        {"route": "relationship", "retrieval_mode": "vector"}) == "sufficiency"


def test_route_after_sufficiency():
    assert _route_after_sufficiency({"sufficient": True}) == "synthesize"
    assert _route_after_sufficiency({"sufficient": False}) == "retrieve"


def test_sufficiency_node_hard_cap_bypasses_llm(monkeypatch):
    called = []
    monkeypatch.setattr(llm, "complete", lambda p, temperature=0.2: called.append(p))
    update = _sufficiency_node({"iterations": 3, "context": [chunk("c1")]})
    assert update["sufficient"] is True
    assert "hard cap" in update["sufficiency_note"]
    assert not called  # cap short-circuits before the LLM


def test_sufficiency_node_judges_context(monkeypatch):
    monkeypatch.setattr(
        llm, "complete", lambda p, temperature=0.2: "sufficient")
    update = _sufficiency_node(
        {"question": "q", "iterations": 1, "context": [chunk("c1")]})
    assert update["sufficient"] is True and update["iterations"] == 1

    monkeypatch.setattr(
        llm, "complete", lambda p, temperature=0.2:
        "insufficient: missing clause 15 details")
    update = _sufficiency_node(
        {"question": "q", "iterations": 1, "context": []})
    assert update["sufficient"] is False
    assert "clause 15" in update["sufficiency_note"]


def test_traverse_node_appends_graph_chunk(monkeypatch):
    monkeypatch.setattr(text2cypher, "load_schema", lambda: {"labels": {}})
    monkeypatch.setattr(text2cypher, "run_text2cypher", lambda *a: {
        "cypher": "MATCH (n) RETURN n LIMIT 1",
        "rows": [{"name": "p1"}],
        "chunk_refs": ["g1"],
    })
    update = _traverse_node(
        {"question": "q", "context": [chunk("c1")]}, FakeService([]))
    assert update["cypher"].startswith("MATCH")
    assert update["graph_chunk_ids"] == ["g1"]
    ids = [c["id"] for c in update["context"]]
    assert ids == ["c1", "graph:traversal"]
    graph_chunk = update["context"][-1]
    assert graph_chunk["kind"] == "graph"
    assert "p1" in graph_chunk["text"]


def test_traverse_node_records_failure_note(monkeypatch):
    monkeypatch.setattr(text2cypher, "load_schema", lambda: {})
    monkeypatch.setattr(text2cypher, "run_text2cypher", lambda *a: None)
    update = _traverse_node({"question": "q", "context": []}, FakeService([]))
    assert "Text2Cypher failed" in update["notes"][0]
    assert "cypher" not in update


def test_cite_node_combines_retrieval_and_graph_ids(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        citations, "build_citations",
        lambda driver, ids: seen.update(ids=ids) or [{"chunk_id": i} for i in ids])
    update = _cite_node({"chunk_ids": ["c1"], "graph_chunk_ids": ["g1"]},
                        FakeService([]))
    assert seen["ids"] == ["c1", "g1"]
    assert len(update["citations"]) == 2


def test_synthesize_node_streams_and_joins(monkeypatch):
    monkeypatch.setattr(llm, "stream", lambda prompt, temperature=0.2:
                        iter(["Answer ", "part"]))
    tokens = []
    monkeypatch.setattr(graph_mod, "get_stream_writer", lambda: tokens.append)
    update = _synthesize_node({"question": "q", "context": [chunk("c1")],
                               "notes": ["note one"]})
    assert update["answer"] == "Answer part"
    assert tokens == [{"type": "token", "text": "Answer "},
                      {"type": "token", "text": "part"}]


def test_synthesize_node_empty_notes(monkeypatch):
    monkeypatch.setattr(llm, "stream", lambda prompt, temperature=0.2: iter(["x"]))
    update = _synthesize_node({"question": "q", "context": []})
    assert update["answer"] == "x"


# -------------------------------------------------------------- step_detail

def test_step_detail_routes_and_rewrite():
    detail, emit = step_detail("route", {"route": "lookup",
                                         "route_rationale": "why"})
    assert detail == {"route": "lookup", "rationale": "why"} and emit

    detail, emit = step_detail("rewrite", {"rewritten": "r"})
    assert detail == {"rewritten": "r"} and emit


def test_step_detail_retrieve_maps_scores():
    update = {
        "retrieval_mode": "hybrid", "iterations": 2,
        "chunk_ids": ["c1", "missing"],
        "context": [chunk("c1", score=0.75)],
    }
    detail, emit = step_detail("retrieve", update)
    assert detail["mode"] == "hybrid"
    assert detail["chunks"] == 2
    assert detail["scores"] == [0.75, 0.0]  # unknown id scores 0
    assert emit


def test_step_detail_traverse_sufficiency_synthesize():
    detail, emit = step_detail("traverse", {"cypher": "MATCH"})
    assert detail == {"cypher": "MATCH"} and emit

    detail, emit = step_detail(
        "sufficiency", {"sufficient": False, "iterations": 1,
                        "sufficiency_note": "need more"})
    assert detail == {"sufficient": False, "iterations": 1, "note": "need more"}
    assert emit

    assert step_detail("synthesize", {"answer": "a"}) == ({}, True)


def test_step_detail_cite_emits_nothing():
    assert step_detail("cite", {"citations": []}) == ({}, False)


# ----------------------------------------------------------- compiled graph

class StubRetrievalService:
    """Replaces retrievers.RetrievalService inside build_graph."""

    def __init__(self, driver):
        self.driver = driver
        self.dense = SimpleNamespace(driver=driver)

    def search(self, query, mode="hybrid", top_k=DEFAULT_TOP_K):
        return [chunk("c1", score=0.9), chunk("c2", score=0.8)]


def test_build_graph_end_to_end_relationship_route(monkeypatch):
    """Full invoke through the compiled graph with every edge stubbed."""
    answers = iter([
        "relationship asks how entities connect",   # route
        "rewritten: A B relationship",              # rewrite
        "sufficient",                               # sufficiency
    ])
    monkeypatch.setattr(
        llm, "complete", lambda prompt, temperature=0.2: next(answers))
    monkeypatch.setattr(
        llm, "stream", lambda prompt, temperature=0.2: iter(["The answer."]))
    monkeypatch.setattr(graph_mod.retrievers, "RetrievalService",
                        StubRetrievalService)
    monkeypatch.setattr(text2cypher, "load_schema", lambda: {})
    monkeypatch.setattr(text2cypher, "run_text2cypher", lambda *a: {
        "cypher": "MATCH (n) RETURN n", "rows": [{"name": "p1"}],
        "chunk_refs": ["g1"]})
    monkeypatch.setattr(
        citations, "build_citations",
        lambda driver, ids: [{"chunk_id": i} for i in ids])

    compiled = build_graph(SimpleNamespace())
    result = compiled.invoke({"question": "How do A and B relate?"})

    assert result["route"] == "relationship"
    assert result["rewritten"] == "rewritten: A B relationship"
    assert result["cypher"] == "MATCH (n) RETURN n"
    assert result["answer"] == "The answer."
    assert result["iterations"] == 1
    assert result["citations"] == [{"chunk_id": "c1"},
                                   {"chunk_id": "c2"},
                                   {"chunk_id": "g1"}]


def test_build_graph_lookup_route_retries_once_then_synthesizes(monkeypatch):
    """lookup skips traversal; one insufficiency loop, then synthesize."""
    answers = iter([
        "lookup plain fact question",   # route
        "rewritten: fact",              # rewrite
        "insufficient: need clause 15", # sufficiency (iteration 1)
        "sufficient",                   # sufficiency (iteration 2)
    ])
    monkeypatch.setattr(
        llm, "complete", lambda prompt, temperature=0.2: next(answers))
    monkeypatch.setattr(
        llm, "stream", lambda prompt, temperature=0.2: iter(["Answer."]))
    monkeypatch.setattr(graph_mod.retrievers, "RetrievalService",
                        StubRetrievalService)
    monkeypatch.setattr(citations, "build_citations", lambda driver, ids: [])

    compiled = build_graph(SimpleNamespace())
    result = compiled.invoke(
        {"question": "What does clause 15 say?", "retrieval_mode": "vector"})

    assert result["route"] == "lookup"
    assert result["iterations"] == 2  # one retrieve + one retry
    assert "insufficient: need clause 15" in result["notes"]
    assert result["answer"] == "Answer."