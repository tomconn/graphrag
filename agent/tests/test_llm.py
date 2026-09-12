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


# ------------------------------------------------------- budget + truncation

def test_complete_max_tokens_override(stub_client):
    """The stage-supplied budget overrides the default cap."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    completions = stub_client.install(response)
    llm.complete("p", max_tokens=8192)
    assert completions.calls[0]["max_tokens"] == 8192
    llm.complete("p")
    assert completions.calls[1]["max_tokens"] == llm._DEFAULT_MAX_TOKENS


def test_complete_logs_warning_on_truncated_reply(stub_client, caplog):
    """finish_reason=length means the output was cut (e.g. a Cypher statement
    severed mid-pattern) — the log must say so explicitly."""
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="MATCH (c:CpsClause)<-[g:REL"),
        finish_reason="length")])
    stub_client.install(response)
    with caplog.at_level("WARNING", logger="app.llm"):
        llm.complete("p", purpose="text2cypher", max_tokens=100)
    assert any("truncated by the 100-token cap" in r.getMessage()
               for r in caplog.records)


def test_complete_no_warning_when_finish_is_stop(stub_client, caplog):
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="ok"), finish_reason="stop")])
    stub_client.install(response)
    with caplog.at_level("WARNING", logger="app.llm"):
        llm.complete("p")
    assert not [r for r in caplog.records if "truncated" in r.getMessage()]


def test_stream_logs_warning_on_truncated_reply(stub_client, caplog):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="partial"),
            finish_reason=None)]),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=""),
            finish_reason="length")]),
    ]
    completions = stub_client.install(None, stream_chunks=chunks)
    completions.create = lambda **kw: iter(chunks)
    with caplog.at_level("WARNING", logger="app.llm"):
        assert list(llm.stream("p", max_tokens=50)) == ["partial"]
    assert any("truncated by the 50-token cap" in r.getMessage()
               for r in caplog.records)
