"""Unit tests for agent/app/embedder.py — lazy FastEmbed load (stubbed),
embed_query conversion and the double-checked locking singleton behaviour.
"""
import sys
import threading
from types import SimpleNamespace

import pytest

import app.embedder as embedder_mod
from app.embedder import FastEmbedEmbedder


class FakeTextEmbedding:
    instances = []

    def __init__(self, model_name=None, lazy_load=False):
        assert lazy_load is True, "agent embedder must lazy-load"
        self.model_name = model_name
        self.query_calls = []
        FakeTextEmbedding.instances.append(self)

    def query_embed(self, text):
        self.query_calls.append(text)
        # 4-dim vector; value = len(text) so conversion is observable
        return iter([[float(len(text)), 0.0, 1.0, 2.0]])


@pytest.fixture(autouse=True)
def fake_fastembed(monkeypatch):
    FakeTextEmbedding.instances = []
    monkeypatch.setitem(sys.modules, "fastembed",
                        SimpleNamespace(TextEmbedding=FakeTextEmbedding))


def test_embed_query_returns_float_list():
    embedder = FastEmbedEmbedder()
    vector = embedder.embed_query("hello")
    assert vector == [5.0, 0.0, 1.0, 2.0]
    assert all(isinstance(x, float) for x in vector)


def test_model_is_lazy_and_cached():
    embedder = FastEmbedEmbedder()
    assert not FakeTextEmbedding.instances  # nothing loaded at construction
    embedder.embed_query("a")
    embedder.embed_query("b")
    assert len(FakeTextEmbedding.instances) == 1  # loaded once
    assert FakeTextEmbedding.instances[0].query_calls == ["a", "b"]


def test_embedder_model_name_from_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "env/model")
    FastEmbedEmbedder().embed_query("x")
    assert FakeTextEmbedding.instances[0].model_name == "env/model"
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    FastEmbedEmbedder().embed_query("x")
    assert FakeTextEmbedding.instances[-1].model_name == \
        "mixedbread-ai/mxbai-embed-large-v1"


def test_ensure_model_thread_safe_single_load():
    """Concurrent first embeds must not build the model twice."""
    embedder = FastEmbedEmbedder()
    barrier = threading.Barrier(4)
    results = []

    def worker():
        barrier.wait()
        results.append(embedder.embed_query("x"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4
    assert len(FakeTextEmbedding.instances) == 1