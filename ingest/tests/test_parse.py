"""Unit tests for pipeline/parse.py — classification, titles, corpus walk,
document loading (incl. duplicate-stem disambiguation) and PDF handling.
"""
import sys
import types

import pytest

from pipeline.parse import (
    DOC_TYPES,
    SourceDocument,
    classify,
    doc_title,
    load_documents,
    pdf_to_text,
    read_file,
    walk_data,
)


# ------------------------------------------------------------------ classify

def test_classify_known_types():
    for doc_type in DOC_TYPES:
        assert classify(doc_type) == doc_type


def test_classify_unknown_returns_none():
    assert classify("misc") is None
    assert classify("") is None


# ----------------------------------------------------------------- doc_title

def test_doc_title_first_h1_wins():
    text = "# The Real Title\n\nsome body\n# Second Heading\n"
    assert doc_title(text, "fallback") == "The Real Title"


def test_doc_title_ignores_h2_and_below():
    assert doc_title("## Sub\n### Deeper\n", "stem") == "stem"


def test_doc_title_skips_empty_heading():
    assert doc_title("#\n# Real\n", "stem") == "Real"


def test_doc_title_fallback_on_no_heading():
    assert doc_title("plain text\nmore\n", "stem") == "stem"


# ----------------------------------------------------------------- read_file

@pytest.mark.parametrize("suffix,content", [
    (".md", "markdown body"),
    (".txt", "plain body"),
    (".py", "print('hi')\n"),
])
def test_read_file_text_and_code(tmp_path, suffix, content):
    path = tmp_path / f"doc{suffix}"
    path.write_text(content, encoding="utf-8")
    assert read_file(path, "code" if suffix == ".py" else "regulatory") == content


def test_read_file_unsupported_extension_raises(tmp_path):
    path = tmp_path / "table.xlsx"
    path.write_text("binary-ish")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_file(path, "regulatory")


def test_pdf_without_docling_raises_operational_error(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    # docling is deliberately not installed in the test env; be deterministic
    # even if someone installs it: hide the module for this test.
    monkey = pytest.MonkeyPatch()
    monkey.setitem(sys.modules, "docling", None)
    monkey.setitem(sys.modules, "docling.document_converter", None)
    try:
        with pytest.raises(RuntimeError, match="docling is not installed"):
            pdf_to_text(path)
    finally:
        monkey.undo()


def test_pdf_with_docling_converts(tmp_path, monkeypatch):
    """Fake a docling install and check the markdown export round-trip."""
    class FakeResult:
        class document:  # noqa: N801 - mirrors docling's attribute shape
            @staticmethod
            def export_to_markdown():
                return "# converted"

        # document accessed as result.document
    class FakeConverter:
        def __init__(self):
            self.converted = []

        def convert(self, path):
            self.converted.append(path)
            return FakeResult()

    docling = types.ModuleType("docling")
    sub = types.ModuleType("docling.document_converter")
    sub.DocumentConverter = FakeConverter
    docling.document_converter = sub

    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setitem(sys.modules, "docling", docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", sub)

    assert pdf_to_text(path) == "# converted"


# ----------------------------------------------------------------- walk_data

def test_walk_data_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        walk_data(tmp_path / "does-not-exist")


def test_walk_data_order_and_classification(tmp_path):
    # create in scrambled order; walk must return doc-type order then sorted
    (tmp_path / "security").mkdir()
    (tmp_path / "security" / "b.md").write_text("s-b")
    (tmp_path / "security" / "a.md").write_text("s-a")
    (tmp_path / "regulatory").mkdir()
    (tmp_path / "regulatory" / "z.md").write_text("r-z")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "m.py").write_text("x = 1\n")
    (tmp_path / "architecture").mkdir()  # empty class dir: skipped silently

    found = walk_data(tmp_path)
    assert [(p.name, t) for p, t in found] == [
        ("z.md", "regulatory"), ("a.md", "security"), ("b.md", "security"),
        ("m.py", "code"),
    ]


def test_walk_data_skips_meta_hidden_unsupported(tmp_path):
    reg = tmp_path / "regulatory"
    reg.mkdir()
    (reg / "README.md").write_text("meta")
    (reg / ".hidden.md").write_text("hidden")
    (reg / "data.csv").write_text("a,b")
    (reg / "real.md").write_text("content")
    found = walk_data(tmp_path)
    assert [p.name for p, _ in found] == ["real.md"]


def test_walk_data_classifies_by_immediate_parent_only(tmp_path):
    (tmp_path / "regulatory" / "nested-deep").mkdir(parents=True)
    (tmp_path / "regulatory" / "nested-deep" / "doc.md").write_text("x")
    found = walk_data(tmp_path)
    # parent dir is not a class dir, but rglob runs under the class dir, so
    # the file is still found and classified by the class root's name
    assert [(p.name, t) for p, t in found] == [("doc.md", "regulatory")]


# ------------------------------------------------------- SourceDocument etc.

def test_source_document_from_path(tmp_path):
    data_root = tmp_path / "data"
    reg = data_root / "regulatory"
    reg.mkdir(parents=True)
    doc_file = reg / "cps-234.md"
    doc_file.write_text("# Title\n")
    doc = SourceDocument.from_path(
        doc_file, "regulatory", data_root, "# Title\n", "Title from arg")
    assert doc.doc_id == "cps-234"
    assert doc.doc_type == "regulatory"
    assert doc.title == "Title from arg"
    assert doc.source_path == "data/regulatory/cps-234.md"
    assert doc.abs_path == str(doc_file)
    assert doc.text == "# Title\n"


def test_from_path_empty_title_falls_back_to_stem(tmp_path):
    data_root = tmp_path / "data"
    doc_file = data_root / "regulatory" / "no-heading.md"
    doc_file.parent.mkdir(parents=True)
    doc_file.write_text("body only\n")
    doc = SourceDocument.from_path(doc_file, "regulatory", data_root, "body only\n", "")
    assert doc.title == "no-heading"


# ------------------------------------------------------------ load_documents

def test_load_documents_reads_corpus(tmp_path):
    (tmp_path / "regulatory").mkdir()
    (tmp_path / "regulatory" / "one.md").write_text("# One\n")
    (tmp_path / "security").mkdir()
    (tmp_path / "security" / "two.txt").write_text("Two body\n")
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["one", "two"]
    assert docs[0].title == "One"
    assert docs[0].doc_type == "regulatory"
    assert docs[1].doc_type == "security"


def test_load_documents_disambiguates_duplicate_stems(tmp_path):
    (tmp_path / "regulatory").mkdir()
    (tmp_path / "regulatory" / "same.md").write_text("# First\n")
    (tmp_path / "security").mkdir()
    (tmp_path / "security" / "same.md").write_text("# Second\n")
    docs = load_documents(tmp_path)
    ids = [d.doc_id for d in docs]
    assert ids[0] == "same"
    assert ids[1].startswith("same-") and ids[1] != "same"
    assert docs[1].title == "Second"