"""Unit tests for agent/app/llm.py — the OpenAI-compatible client with the
OpenAI object stubbed: complete() and stream() over canned responses.
"""
from types import SimpleNamespace

import pytest

import app.llm as llm


class FakeCompletions:
    """chat.completions.create stand-in recording the request."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


@pytest.fixture()
def stub_client(monkeypatch):
    """Patch llm._CLIENT with a fake whose chat.completions is controllable."""
    holder = SimpleNamespace()

    def install(response, stream_chunks=None):
        completions = FakeCompletions(response)
        completions.stream_chunks = stream_chunks
        fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm, "_CLIENT", fake)
        return completions

    holder.install = install
    return holder


# ------------------------------------------------------------------ complete

def test_complete_returns_stripped_content(stub_client):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  hello \n"))])
    completions = stub_client.install(response)
    assert llm.complete("prompt") == "hello"
    call = completions.calls[0]
    assert call["messages"] == [{"role": "user", "content": "prompt"}]
    assert call["temperature"] == llm._TEMPERATURE
    assert not call.get("stream")


def test_complete_none_content_becomes_empty(stub_client):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    stub_client.install(response)
    assert llm.complete("prompt") == ""


def test_complete_custom_temperature(stub_client):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="x"))])
    completions = stub_client.install(response)
    llm.complete("p", temperature=0.9)
    assert completions.calls[0]["temperature"] == 0.9


# -------------------------------------------------------------------- stream

def test_stream_yields_token_contents(stub_client):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Hello"))]),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=" world"))]),
    ]
    completions = stub_client.install(None, stream_chunks=chunks)
    response = iter(chunks)
    completions.create = lambda **kw: (completions.calls.append(kw), response)[1]
    assert list(llm.stream("p")) == ["Hello", " world"]
    assert completions.calls[0]["stream"] is True


def test_stream_skips_empty_deltas_and_role_chunks(stub_client):
    chunks = [
        SimpleNamespace(choices=[]),                                # no choices
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None))]),                 # null delta
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="real"))]),
    ]
    completions = stub_client.install(None, stream_chunks=chunks)
    completions.create = lambda **kw: iter(chunks)
    assert list(llm.stream("p")) == ["real"]


# ------------------------------------------------------------------ _model()

def test_model_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    assert llm._model() == "test-model"
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert llm._model() == "glm-5.3-flash:cloud"


def test_client_is_cached(monkeypatch):
    """_client() builds once and reuses the module-level singleton."""
    built = []

    def factory(**kwargs):
        built.append(SimpleNamespace(chat=SimpleNamespace()))
        return built[-1]

    monkeypatch.setattr(llm, "_CLIENT", None)
    monkeypatch.setattr(llm.openai, "OpenAI", factory)
    first = llm._client()
    second = llm._client()
    assert first is second
    assert len(built) == 1

# ------------------------------------------------------------------ purpose

def test_complete_logs_purpose(stub_client, caplog):
    """The call and reply are logged with the caller-supplied stage label."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    stub_client.install(response)
    with caplog.at_level("INFO", logger="app.llm"):
        llm.complete("prompt", purpose="route")
    messages = [r.getMessage() for r in caplog.records]
    assert any("llm call purpose=route" in m for m in messages)
    assert any("llm reply purpose=route" in m for m in messages)


def test_complete_purpose_defaults_to_prompt_excerpt(stub_client, caplog):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
    stub_client.install(response)
    with caplog.at_level("INFO", logger="app.llm"):
        llm.complete("Classify the question into exactly one intent category")
    assert any("purpose=Classify the question into" in r.getMessage()
               for r in caplog.records)


def test_stream_logs_reply_when_consumed(stub_client, caplog):
    chunks = [SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content="ok"))])]
    completions = stub_client.install(None, stream_chunks=chunks)
    completions.create = lambda **kw: iter(chunks)
    with caplog.at_level("INFO", logger="app.llm"):
        assert list(llm.stream("p", purpose="synthesize")) == ["ok"]
    assert any("llm reply purpose=synthesize" in r.getMessage()
               for r in caplog.records)
