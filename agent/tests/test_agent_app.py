"""Unit tests for agent/app/main.py — the FastAPI /health and /chat (SSE)
endpoints with the compiled agent graph replaced by a stub.
"""
import json
from types import SimpleNamespace

import neo4j
import pytest

import app.main as main_mod


def sse_events(text):
    """Parse an SSE body into (type, payload) tuples."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: ")
        events.append(json.loads(block[len("data: "):]))
    return events


class StubCompiledGraph:
    """astream yields (stream_mode, chunk) pairs like the real graph."""

    def __init__(self, updates=None, fail=False):
        self.updates = updates or [
            ("updates", {"route": {"route": "lookup", "route_rationale": "why"}}),
            ("custom", {"type": "token", "text": "Hello "}),
            ("custom", {"type": "token", "text": "world"}),
            ("updates", {"retrieve": {"chunk_ids": ["c1"], "retrieval_mode":
                                      "hybrid", "iterations": 1,
                                      "context": []}}),
            ("updates", {"synthesize": {"answer": "Hello world"}}),
            ("updates", {"cite": {"citations": [{"chunk_id": "c1"}]}}),
        ]
        self.fail = fail
        self.calls = []

    async def astream(self, inputs, stream_mode=None, config=None):
        self.calls.append(inputs)
        if self.fail:
            raise RuntimeError("model tag unavailable")
        for item in self.updates:
            yield item


@pytest.fixture()
def stubbed_graph(monkeypatch):
    stub = StubCompiledGraph()
    monkeypatch.setattr(main_mod, "_COMPILED_GRAPH", stub)
    return stub


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_streams_steps_tokens_and_done(client, stubbed_graph):
    resp = client.post("/chat", json={"message": "What is clause 15?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp.text)
    types = [e["type"] for e in events]
    assert "step" in types and "token" in types and "done" in types

    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Hello world"
    assert done["citations"] == [{"chunk_id": "c1"}]
    assert done["trace_id"]

    steps = [e for e in events if e["type"] == "step"]
    assert [s["step"] for s in steps] == ["route", "retrieve", "synthesize"]
    assert steps[0]["detail"]["route"] == "lookup"

    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["Hello ", "world"]
    # question and mode forwarded into the graph inputs
    assert stubbed_graph.calls[0]["question"] == "What is clause 15?"
    assert stubbed_graph.calls[0]["retrieval_mode"] == "hybrid"


def test_chat_retrieval_mode_override(client, stubbed_graph):
    client.post("/chat", json={"message": "q", "retrieval_mode": "VECTOR"})
    assert stubbed_graph.calls[0]["retrieval_mode"] == "vector"


def test_chat_empty_message_is_an_error_event(client, stubbed_graph, tmp_path,
                                              monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    resp = client.post("/chat", json={"message": "   "})
    events = sse_events(resp.text)
    assert events[0]["type"] == "error"
    assert events[0]["message"] == "empty message"
    assert stubbed_graph.calls == []  # graph never invoked


def test_chat_graph_failure_is_an_error_event(client, monkeypatch):
    stub = StubCompiledGraph(fail=True)
    monkeypatch.setattr(main_mod, "_COMPILED_GRAPH", stub)
    resp = client.post("/chat", json={"message": "q"})
    events = sse_events(resp.text)
    assert events[0]["type"] == "error"
    assert "model tag unavailable" in events[0]["message"]


def test_chat_initialisation_failure_is_an_error_event(client, monkeypatch):
    """_COMPILED_GRAPH unset and driver creation explodes -> error event."""
    monkeypatch.setattr(main_mod, "_COMPILED_GRAPH", None)

    def explode(uri, auth=None):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(neo4j.GraphDatabase, "driver",
                        staticmethod(explode), raising=False)
    resp = client.post("/chat", json={"message": "q"})
    events = sse_events(resp.text)
    assert events[0]["type"] == "error"
    assert "agent initialisation failed" in events[0]["message"]
    assert "neo4j down" in events[0]["message"]


def test_compiled_graph_cached_after_first_build(monkeypatch):
    """Second call to _compiled_graph returns the memoised instance."""
    import app.main as m
    fake_driver = SimpleNamespace(verify_connectivity=lambda: None)
    monkeypatch.setattr(m, "_COMPILED_GRAPH", None)
    monkeypatch.setattr(
        neo4j.GraphDatabase, "driver",
        staticmethod(lambda uri, auth=None: fake_driver), raising=False)
    monkeypatch.setattr(m.graph_module, "build_graph", lambda driver: fake_driver)
    assert m._compiled_graph() is fake_driver
    assert m._compiled_graph() is fake_driver  # second call must not rebuild


def test_sse_format_is_parseable():
    assert main_mod._sse({"type": "token", "text": "ünïcode"}) == \
        'data: {"type": "token", "text": "ünïcode"}\n\n'