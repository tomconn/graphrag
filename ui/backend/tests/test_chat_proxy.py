"""Unit tests for the ui/backend chat proxy (POST /api/chat) — the agent
SSE relay, its error paths, and the /api/health agent probe. The agent-side
httpx client is replaced with a stub.
"""
import json
from types import SimpleNamespace

import httpx
import pytest


class FakeAgentResponse:
    def __init__(self, status_code=200, chunks=(), body=b""):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._body = body

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return self._body


class FakeAgentClient:
    """httpx.AsyncClient stand-in: streams canned agent responses."""

    last_payload = None

    def __init__(self, responses, raise_on_stream=None):
        self.responses = list(responses)
        self.raise_on_stream = raise_on_stream

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, json=None):
        FakeAgentClient.last_payload = json
        if self.raise_on_stream is not None:
            raise self.raise_on_stream
        return _StreamCtx(self.responses.pop(0))


class _StreamCtx:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        return False


def sse_events(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block:
            events.append(json.loads(block[len("data: "):]))
    return events


@pytest.fixture()
def patch_agent_client(ui_app, monkeypatch):
    """Install a factory the tests can seed with canned responses."""
    holder = SimpleNamespace()

    def install(responses=(), raise_on_stream=None):
        def factory(*args, **kwargs):
            return FakeAgentClient(responses, raise_on_stream)
        monkeypatch.setattr(ui_app.httpx, "AsyncClient", factory)
        return holder

    holder.install = install
    return holder


# ------------------------------------------------------------- POST /api/chat

def test_chat_proxies_agent_stream(ui_app, patch_agent_client):
    patch_agent_client.install(responses=[FakeAgentResponse(chunks=[
        b'data: {"type": "step", "step": "route"}\n\n',
        b'data: {"type": "token", "text": "Hello"}\n\n',
    ])])
    from fastapi.testclient import TestClient
    with TestClient(ui_app.app) as c:
        resp = c.post("/api/chat", json={"message": "hi",
                                         "retrieval_mode": "hybrid"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    # proxied byte-for-byte
    assert b'"type": "step"' in resp.content
    assert b'"type": "token"' in resp.content
    # the agent received the request payload
    assert FakeAgentClient.last_payload == {"message": "hi",
                                            "retrieval_mode": "hybrid"}


def test_chat_non_200_becomes_error_event(ui_app, patch_agent_client):
    patch_agent_client.install(responses=[FakeAgentResponse(
        status_code=503, body=b"agent exploded")])
    from fastapi.testclient import TestClient
    with TestClient(ui_app.app) as c:
        resp = c.post("/api/chat", json={"message": "hi"})
    assert b'"agent returned HTTP 503"' in resp.content


def test_chat_connection_error_becomes_error_event(ui_app, patch_agent_client):
    patch_agent_client.install(
        raise_on_stream=httpx.ConnectError("connection refused"))
    from fastapi.testclient import TestClient
    with TestClient(ui_app.app) as c:
        resp = c.post("/api/chat", json={"message": "hi"})
    assert b"agent unreachable" in resp.content


def test_sse_event_encoding(ui_app):
    payload = {"type": "token", "text": "hi"}
    assert ui_app._sse_event(payload) == \
        b'data: {"type": "token", "text": "hi"}\n\n'


# ---------------------------------------------------------------- /api/health

def test_health_reports_agent_ok(ui_app, monkeypatch):
    class OkClient(FakeAgentClient):
        async def get(self, url, timeout=None):
            return FakeAgentResponse(status_code=200)

    monkeypatch.setattr(
        ui_app.httpx, "AsyncClient", lambda *a, **kw: OkClient(()))
    from fastapi.testclient import TestClient
    with TestClient(ui_app.app) as c:
        body = c.get("/api/health").json()
    assert body == {"status": "ok", "agent": "ok"}


def test_health_reports_agent_unreachable(ui_app, monkeypatch):
    class DeadClient(FakeAgentClient):
        async def get(self, url, timeout=None):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(
        ui_app.httpx, "AsyncClient", lambda *a, **kw: DeadClient(()))
    from fastapi.testclient import TestClient
    with TestClient(ui_app.app) as c:
        body = c.get("/api/health").json()
    assert body == {"status": "ok", "agent": "unreachable"}