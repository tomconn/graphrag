"""Unit tests for ingest/pipeline/chunk.py.

Covers markdown heading-hierarchy chunking, regulatory Clause splitting and
tree-sitter code chunking. No Neo4j, no LLM, no embeddings.
"""
import hashlib

import pytest

from pipeline.chunk import (
    _chunk_id,
    _hard_split,
    chunk_code,
    chunk_document,
    chunk_markdown,
    chunk_params,
    chunk_regulatory,
    group_paragraphs,
)

ARCH_DOC = """# Main Title

intro paragraph

## Architecture Overview

architecture body

### Pattern Catalog

pattern body

## Second Section

second body
"""

REG_DOC = """# CPS 999 Test Standard

Objective and application notes.

## Part A - Operational Risk

### Clause 27

Institutions must manage operational risk.

#### Clause 27.1

Sub clause text.

## Part B

### Clause 12

Other requirement.

## Clause 99

Not a clause heading.

### clause 7

Lowercase clause body.
"""

CODE_SOURCE = '''"""Module docstring."""

import os

CONSTANT = 1

def alpha(x):
    """Alpha doc."""
    return x + 1

class Beta:
    def method(self):
        return 2

@decorator
def gamma(y):
    return y
'''


# ---------------------------------------------------------------- markdown

def test_heading_paths_h2_and_h3_h1_excluded():
    chunks = chunk_markdown("doc1", "architecture", ARCH_DOC)
    assert [c.section for c in chunks] == [
        "",  # body before the first H2 has no H2+ heading yet
        "Architecture Overview",
        "Architecture Overview > Pattern Catalog",
        "Second Section",
    ]


def test_chunk_texts_follow_their_sections():
    chunks = chunk_markdown("doc1", "architecture", ARCH_DOC)
    assert [c.text for c in chunks] == [
        "intro paragraph",
        "architecture body",
        "pattern body",
        "second body",
    ]


def test_bare_heading_is_kept_addressable_as_text():
    chunks = chunk_markdown("doc1", "architecture", "## Lonely Heading\n")
    assert len(chunks) == 1
    assert chunks[0].section == "Lonely Heading"
    assert chunks[0].text == "Lonely Heading"
    assert chunks[0].clause == "" and chunks[0].code_ref == ""


def test_size_grouping_and_overlap(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "70")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    doc = "\n\n".join(f"para{i} " + "x" * 30 for i in range(6))
    chunks = chunk_markdown("doc1", "architecture", "## Guide\n\n" + doc)
    assert len(chunks) == 6
    for chunk in chunks:
        assert len(chunk.text) <= 70
        assert chunk.section == "Guide"
    # every chunk after the first carries the tail of its predecessor
    assert chunks[1].text.startswith(chunks[0].text[-20:])


def test_group_paragraphs_groups_by_size():
    paras = [f"para{i} " + "x" * 30 for i in range(6)]
    chunks = group_paragraphs(paras, 70, 20)
    assert len(chunks) == 6
    assert all(len(c) <= 70 for c in chunks)
    assert chunks[1].startswith(chunks[0][-20:])


def test_group_paragraphs_without_overlap():
    paras = [f"para{i} " + "x" * 30 for i in range(3)]
    chunks = group_paragraphs(paras, 70, 0)
    assert chunks == paras  # one para per chunk, no overlap tails


def test_hard_split_windows_overlap():
    pieces = _hard_split("a" * 250, 100, 20)
    assert [len(p) for p in pieces] == [100, 100, 90]
    assert pieces[0][-20:] == pieces[1][:20]
    assert "".join(pieces) == "a" * 290  # 250 + 2 * 20 overlap


def test_indented_code_fence_comment_is_not_a_heading():
    doc = "## Guide\n\n```python\n    # indented comment stays in body\n```\n"
    chunks = chunk_markdown("doc1", "architecture", doc)
    assert len(chunks) == 1
    assert chunks[0].section == "Guide"
    assert "# indented comment stays in body" in chunks[0].text


def test_column0_hash_inside_fence_is_mistaken_for_heading():
    """Pinned actual behavior (a known limitation): chunk_markdown is not
    fence-aware, so a column-0 '#' line inside a code fence is treated as an
    H1 — it splits the chunk and the following body loses its section path."""
    doc = (
        "## Guide\n\n```python\n# looks like a heading\ndef f(): pass\n```\n\n"
        "more text\n"
    )
    chunks = chunk_markdown("doc1", "architecture", doc)
    assert [c.section for c in chunks] == ["Guide", ""]
    assert "def f(): pass" in chunks[1].text


# -------------------------------------------------------------- regulatory

def test_clause_chunks_carry_number_and_part_section():
    chunks = chunk_regulatory("doc1", "regulatory", REG_DOC)
    clause_chunks = [c for c in chunks if c.clause]
    by_number = {c.clause: c for c in clause_chunks}
    assert set(by_number) == {"27", "12", "7"}  # no "27.1", no "99"
    c27 = by_number["27"]
    assert c27.section == "Part A - Operational Risk > Clause 27"
    assert c27.text.startswith("### Clause 27")
    assert "Institutions must manage operational risk." in c27.text
    assert c27.code_ref == ""
    assert by_number["12"].section == "Part B > Clause 12"


def test_preamble_grouped_without_clause_metadata():
    chunks = chunk_regulatory("doc1", "regulatory", REG_DOC)
    preamble = [c for c in chunks if not c.clause]
    assert len(preamble) == 1
    assert preamble[0].section == ""
    text = preamble[0].text
    assert "# CPS 999 Test Standard" in text
    assert "Objective and application notes." in text
    # sub-clause numbering ("Clause 27.1") is not a clause boundary: its
    # heading and body stay in the preamble
    assert "#### Clause 27.1" in text
    assert "Sub clause text." in text


def test_part_heading_scopes_the_clause_section():
    chunks = chunk_regulatory("doc1", "regulatory", REG_DOC)
    c7 = next(c for c in chunks if c.clause == "7")
    assert c7.section == "Clause 99 > Clause 7"


def test_lowercase_clause_heading_matches():
    chunks = chunk_regulatory("doc1", "regulatory", REG_DOC)
    c7 = next(c for c in chunks if c.clause == "7")
    assert c7.text.startswith("### clause 7")


def test_clause_with_empty_body_keeps_its_heading_text():
    text = "## Part C\n\n### Clause 5\n\n## Part D\n"
    chunks = chunk_regulatory("doc1", "regulatory", text)
    c5 = next(c for c in chunks if c.clause == "5")
    assert c5.text == "### Clause 5"
    assert c5.section == "Part C > Clause 5"


def test_chunk_ids_are_deterministic_sha256_and_unique():
    chunks = chunk_regulatory("doc1", "regulatory", REG_DOC)
    for index, chunk in enumerate(chunks):
        assert chunk.id == hashlib.sha256(f"doc1:{index}".encode()).hexdigest()
    assert len({c.id for c in chunks}) == len(chunks)
    other = chunk_regulatory("doc2", "regulatory", REG_DOC)
    assert other[0].id != chunks[0].id


# -------------------------------------------------------------------- code

def test_code_top_level_symbols_become_chunks():
    chunks = chunk_code("c1", "code", "pkg/mod.py", CODE_SOURCE)
    assert [c.section for c in chunks] == [
        "pkg/mod.py#module",
        "pkg/mod.py#alpha",
        "pkg/mod.py#Beta",
        "pkg/mod.py#gamma",
    ]


def test_code_module_chunk_holds_preamble():
    chunks = chunk_code("c1", "code", "pkg/mod.py", CODE_SOURCE)
    module = chunks[0]
    assert module.code_ref == "pkg/mod.py#module"
    assert '"""Module docstring."""' in module.text
    assert "import os" in module.text
    assert "CONSTANT = 1" in module.text


def test_code_ref_shape_is_path_symbol_start_end():
    chunks = chunk_code("c1", "code", "pkg/mod.py", CODE_SOURCE)
    refs = {c.section.split("#")[-1]: c.code_ref for c in chunks}
    assert refs["alpha"] == "pkg/mod.py#alpha:7-9"
    assert refs["Beta"] == "pkg/mod.py#Beta:11-13"
    assert refs["gamma"] == "pkg/mod.py#gamma:15-17"
    assert refs["module"] == "pkg/mod.py#module"


def test_code_symbol_chunks_hold_their_own_body():
    chunks = chunk_code("c1", "code", "pkg/mod.py", CODE_SOURCE)
    alpha, beta = chunks[1], chunks[2]
    assert alpha.text == 'def alpha(x):\n    """Alpha doc."""\n    return x + 1'
    assert beta.text == "class Beta:\n    def method(self):\n        return 2"


def test_decorated_function_keeps_decorator_and_symbol():
    chunks = chunk_code("c1", "code", "pkg/mod.py", CODE_SOURCE)
    gamma = chunks[3]
    assert gamma.text.startswith("@decorator")
    assert gamma.code_ref == "pkg/mod.py#gamma:15-17"


def test_code_chunks_are_typed_and_identified():
    chunks = chunk_code("c1", "code", "pkg/mod.py", CODE_SOURCE)
    assert all(c.doc_type == "code" for c in chunks)
    assert all(c.clause == "" for c in chunks)
    for index, chunk in enumerate(chunks):
        assert chunk.id == _chunk_id("c1", index)


def test_imports_only_file_yields_single_module_chunk():
    chunks = chunk_code("c1", "code", "mod.py", "import os\nimport sys\n")
    assert len(chunks) == 1
    assert chunks[0].section == "mod.py#module"
    assert chunks[0].code_ref == "mod.py#module"


def test_empty_source_yields_no_chunks():
    assert chunk_code("c1", "code", "empty.py", "") == []
    assert chunk_code("c1", "code", "empty.py", "   \n") == []


# ---------------------------------------------------------------- dispatch

def test_chunk_document_dispatch(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "900")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    # code + .py -> tree-sitter chunker
    code_chunks = chunk_document("d", "code", "x/mod.py", CODE_SOURCE)
    assert code_chunks[0].code_ref == "x/mod.py#module"
    # code + non-.py -> markdown chunker (no code_ref)
    md_chunks = chunk_document("d", "code", "x/notes.md", "# Title\n\nbody\n")
    assert md_chunks[0].code_ref == ""
    assert md_chunks[0].section == ""
    # regulatory / security -> clause splitter
    reg_chunks = chunk_document("d", "security", "x/a.md", "### Clause 1\n\ntext\n")
    assert reg_chunks[0].clause == "1"
    # anything else -> markdown
    arch_chunks = chunk_document("d", "architecture", "x/a.md", "## H\n\nbody\n")
    assert arch_chunks[0].section == "H"


def test_chunk_params_read_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("CHUNK_OVERLAP", "64")
    assert chunk_params() == (512, 64)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    assert chunk_params() == (900, 150)  # module defaults