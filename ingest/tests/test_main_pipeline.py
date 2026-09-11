"""Unit tests for pipeline/main.py — the per-document orchestration stages
(stub chunker/embedder/extractor/writer) and the run()/main() wiring.
"""
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

import pipeline.main as main_mod
from pipeline.main import _int_env, _require, main, process_document, run


def make_doc():
    return SimpleNamespace(
        doc_id="doc-1", doc_type="regulatory", title="Doc One",
        source_path="data/regulatory/doc-1.md", text="body text")


def make_chunk(cid):
    return SimpleNamespace(id=cid, text=f"chunk {cid}")


class FakeEmbedder:
    def __init__(self):
        self.embedded = []

    def embed_chunks(self, chunks):
        self.embedded.append([c.id for c in chunks])


class FakeExtractor:
    """Returns canned (entities, relations) or raises for one chunk id."""

    node_labels = {"CpsClause": {"key": "id"}, "Pattern": {"key": "name"}}

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.calls = []

    def extract(self, chunk, doc):
        self.calls.append(chunk.id)
        if chunk.id in self.fail_ids:
            raise RuntimeError("LLM exploded")
        return (
            [SimpleNamespace(label="CpsClause", key=f"e-{chunk.id}")],
            [SimpleNamespace(rel_type="REQUIRES", source_label="CpsClause",
                             source_key=f"e-{chunk.id}",
                             target_label="Pattern", target_key="p-1")],
        )


class FakeWriter:
    def __init__(self):
        self.documents = []
        self.entities = {}
        self.edges = {}
        self.closed = False

    def replace_document(self, doc, chunks):
        self.replaced = (doc, chunks)

    def write_entities(self, doc_id, entities):
        self.entities.setdefault(doc_id, []).extend(entities)

    def write_edges(self, doc_id, chunk_id, relations):
        self.edges[(doc_id, chunk_id)] = relations

    def close(self):
        self.closed = True


# --------------------------------------------------------------- env helpers

def test_int_env(monkeypatch):
    monkeypatch.setenv("SOME_INT", "5")
    assert _int_env("SOME_INT", 6) == 5
    monkeypatch.delenv("SOME_INT", raising=False)
    assert _int_env("SOME_INT", 6) == 6


def test_require_present_and_missing(monkeypatch):
    monkeypatch.setenv("PRESENT", "value")
    assert _require("PRESENT") == "value"
    with pytest.raises(RuntimeError, match="Required environment variable"):
        _require("MISSING_ENV_123")


# ---------------------------------------------------------- process_document

def test_process_document_happy_path():
    doc = make_doc()
    chunker = lambda *a: [make_chunk("d-0"), make_chunk("d-1")]  # noqa: E731
    embedder, extractor, writer = FakeEmbedder(), FakeExtractor(), FakeWriter()
    counters = Counter()
    seen_entities, seen_edges = set(), set()
    timings = {"embed": 0.0, "extract": 0.0, "write": 0.0}

    stats = process_document(
        doc, chunker, embedder, extractor, writer, counters,
        seen_entities, seen_edges, concurrency=2, timings=timings)

    assert stats["chunks"] == 2
    assert stats["entities"] == 2 and stats["edges"] == 2
    assert stats["failed_chunks"] == 0
    assert counters["documents"] == 1 and counters["chunks"] == 2
    assert writer.replaced[0] is doc
    assert embedder.embedded == [["d-0", "d-1"]]
    # extraction ran for both chunks and writes are keyed by chunk
    assert [e.key for e in writer.entities["doc-1"]] == ["e-d-0", "e-d-1"]
    assert len(writer.edges) == 2
    assert seen_entities == {("CpsClause", "e-d-0"), ("CpsClause", "e-d-1")}
    assert seen_edges == {("REQUIRES", "CpsClause", "e-d-0", "Pattern", "p-1"),
                          ("REQUIRES", "CpsClause", "e-d-1", "Pattern", "p-1")}
    assert all(t >= 0.0 for t in timings.values())


def test_process_document_no_chunks_writes_nothing():
    writer, embedder = FakeWriter(), FakeEmbedder()
    counters = Counter()
    stats = process_document(
        make_doc(), lambda *a: [], embedder, FakeExtractor(), writer,
        counters)
    assert stats == {"chunks": 0, "entities": 0, "edges": 0, "failed_chunks": 0}
    assert not hasattr(writer, "replaced")
    assert embedder.embedded == []
    assert counters == Counter()


def test_process_document_extraction_failure_degrades():
    """A failed chunk extraction still writes the retrieval layer."""
    writer, extractor = FakeWriter(), FakeExtractor(fail_ids={"d-1"})
    counters = Counter()
    stats = process_document(
        make_doc(), lambda *a: [make_chunk("d-0"), make_chunk("d-1")],
        FakeEmbedder(), extractor, writer, counters)
    assert stats["failed_chunks"] == 1
    assert counters["failed_chunks"] == 1
    assert stats["chunks"] == 2
    # failed chunk degrades to empty entity/edge lists
    assert writer.edges[("doc-1", "d-1")] == []
    assert len(writer.entities["doc-1"]) == 1  # only the successful chunk


def test_process_document_defaults_create_their_own_state():
    stats = process_document(
        make_doc(), lambda *a: [make_chunk("only")], FakeEmbedder(),
        FakeExtractor(), FakeWriter(), Counter())
    assert stats["chunks"] == 1


# ------------------------------------------------------------- run() wiring

@pytest.fixture()
def fake_stack(monkeypatch):
    """Patch every stage module so run() drives the whole pipeline on fakes."""
    import pipeline.chunk as chunk_mod
    import pipeline.embed as embed_mod
    import pipeline.extract as extract_mod
    import pipeline.parse as parse_mod
    import pipeline.write as write_mod

    doc = make_doc()
    load_docs = SimpleNamespace(load_documents=lambda root: [doc])
    monkeypatch.setattr(parse_mod, "load_documents", load_docs.load_documents)

    writer = FakeWriter()
    embedder = FakeEmbedder()
    extractor = FakeExtractor()

    monkeypatch.setattr(chunk_mod, "chunk_params", lambda: (512, 64))
    monkeypatch.setattr(chunk_mod, "chunk_document",
                        lambda *a: [make_chunk("d-0"), make_chunk("d-1")])
    monkeypatch.setattr(embed_mod, "Embedder", lambda *a, **kw: embedder)
    monkeypatch.setattr(extract_mod, "Extractor", lambda: extractor)
    monkeypatch.setattr(
        write_mod, "GraphWriter",
        lambda *a, **kw: writer)
    import pipeline.logsetup as logsetup
    monkeypatch.setattr(logsetup, "setup_logging", lambda: None)

    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    return SimpleNamespace(doc=doc, writer=writer, embedder=embedder,
                           extractor=extractor)


def test_run_processes_all_documents_and_closes_writer(fake_stack, caplog):
    run()
    assert fake_stack.writer.closed
    assert fake_stack.extractor.calls == ["d-0", "d-1"]


def test_run_with_empty_corpus_still_closes(fake_stack, monkeypatch):
    import pipeline.parse as parse_mod
    monkeypatch.setattr(parse_mod, "load_documents", lambda root: [])
    run()
    assert fake_stack.writer.closed
    assert fake_stack.extractor.calls == []


def test_run_requires_neo4j_password(monkeypatch):
    import pipeline.logsetup as logsetup
    monkeypatch.setattr(logsetup, "setup_logging", lambda: None)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD"):
        run()


def test_main_exits_nonzero_on_fatal_error(monkeypatch):
    monkeypatch.setattr(main_mod, "run", lambda: (_ for _ in ()).throw(
        RuntimeError("boom")))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_exits_zero_on_success(monkeypatch):
    monkeypatch.setattr(main_mod, "run", lambda: None)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_main_module_is_importable_without_side_effects():
    assert "pipeline.main" in sys.modules