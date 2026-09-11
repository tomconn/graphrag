"""Unit tests for pipeline/embed.py — FastEmbed adapter with the model
stubbed: dim assertion, batch ordering, count-mismatch error, in-place
chunk filling.
"""
import sys
from types import SimpleNamespace

import pytest

import pipeline.embed as embed_mod
from pipeline.embed import Embedder


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding; returns fixed vectors."""

    instances = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.batches = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, batch):
        self.batches.append(list(batch))
        # one vector per text: dimension 3, value = len(text) so order is
        # observable
        return [[float(len(t)), 0.5, 1.0] for t in batch]


@pytest.fixture(autouse=True)
def fake_fastembed(monkeypatch):
    FakeTextEmbedding.instances = []
    monkeypatch.setitem(sys.modules, "fastembed",
                        SimpleNamespace(TextEmbedding=FakeTextEmbedding))


def make_embedder(expected_dim=3, model_name=None):
    return Embedder(model_name=model_name, expected_dim=expected_dim)


# ------------------------------------------------------------------ __init__

def test_init_accepts_matching_dim():
    embedder = make_embedder(expected_dim=3)
    assert embedder.dim == 3
    assert FakeTextEmbedding.instances[0].model_name == \
        "mixedbread-ai/mxbai-embed-large-v1"  # env-free default


def test_init_uses_explicit_model_name():
    make_embedder(model_name="custom/model")
    assert FakeTextEmbedding.instances[0].model_name == "custom/model"


def test_init_model_name_from_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "env/model")
    make_embedder()
    assert FakeTextEmbedding.instances[0].model_name == "env/model"


def test_init_expected_dim_from_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "3")
    embedder = Embedder()
    assert embedder.dim == 3


def test_init_dim_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        make_embedder(expected_dim=1024)  # fake model outputs 3 dims


# -------------------------------------------------------------------- embed

def test_embed_empty_list_returns_empty():
    assert make_embedder().embed([]) == []


def test_embed_preserves_order():
    vectors = make_embedder().embed(["aa", "bbb", "c"])
    assert [v[0] for v in vectors] == [2.0, 3.0, 1.0]
    assert all(len(v) == 3 for v in vectors)


def test_embed_batches_at_embed_batch(monkeypatch):
    monkeypatch.setattr(embed_mod, "EMBED_BATCH", 2)
    embedder = make_embedder()
    embedder.embed(["a", "b", "c", "d", "e"])
    model = FakeTextEmbedding.instances[-1]
    # batches[0] is the __init__ "dimension probe"; the call batches after it
    assert [len(b) for b in model.batches[1:]] == [2, 2, 1]
    assert model.batches[1] == ["a", "b"]


def test_embed_raises_on_vector_count_mismatch(monkeypatch):
    class ShortModel(FakeTextEmbedding):
        def embed(self, batch):
            if len(batch) == 1:  # the init "dimension probe" must still work
                return [[float(len(batch[0])), 0.5, 1.0]]
            return [[1.0, 2.0, 3.0]]  # one vector regardless of batch size

    monkeypatch.setitem(sys.modules, "fastembed",
                        SimpleNamespace(TextEmbedding=ShortModel))
    with pytest.raises(RuntimeError, match="returned 1 vectors"):
        make_embedder().embed(["a", "b"])


# -------------------------------------------------------------- embed_chunks

def test_embed_chunks_fills_in_place():
    chunks = [SimpleNamespace(id=f"c{i}", text=t, embedding=None)
              for i, t in enumerate(["aa", "bbb"])]
    make_embedder().embed_chunks(chunks)
    assert chunks[0].embedding == [2.0, 0.5, 1.0]
    assert chunks[1].embedding == [3.0, 0.5, 1.0]


def test_embed_chunks_empty_is_noop():
    assert make_embedder().embed_chunks([]) is None