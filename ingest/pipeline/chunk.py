"""Class-specific chunking strategies.

Three strategies, selected by document class:

* code (``.py``)         — tree-sitter-python symbol-aware chunking: one chunk
                           per top-level function/class (docstring + signature
                           + body); leftover top-level statements form a module
                           chunk. ``code_ref = "relative/path.py#Symbol[:start-end]"``
* markdown (architecture)— heading-hierarchy chunking: split on ATX headings,
                           keep the heading path (H2+ levels, e.g.
                           "BIAN capability model > Business Areas and Service
                           Domains") in ``section``; paragraphs grouped until
                           CHUNK_SIZE with CHUNK_OVERLAP.
* regulatory / security  — split on "### Clause N" headings first: each clause
                           is its own chunk (``clause = "N"``, ``section =
                           "<part heading> > Clause N"``); non-clause preamble
                           text is grouped semantically with overlap.

Chunk ids are deterministic: sha256 of "{doc_id}:{index}".
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
CLAUSE_RE = re.compile(r"^#{3,4}\s+Clause\s+(\d+)\s*$", re.IGNORECASE)
FENCE_RE = re.compile(r"^(```|~~~)")


def _is_fence_toggle(line: str) -> bool:
    """True if the line opens or closes a fenced code block (``` or ~~~)."""
    return bool(FENCE_RE.match(line))

DEFAULT_CHUNK_SIZE = 900     # approx characters
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    id: str
    text: str
    embedding: list = field(default_factory=list)  # dense vector, set by the embedder
    section: str = ""    # heading path / "Clause N" / "path#symbol"
    clause: str = ""     # native clause number, regulatory only
    code_ref: str = ""   # "path#symbol[:lines]" for code chunks
    doc_type: str = ""   # "code" for code chunks, else the document class


def chunk_params() -> tuple[int, int]:
    """(CHUNK_SIZE, CHUNK_OVERLAP) from the environment."""
    return (int(os.environ.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            int(os.environ.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)))


def _chunk_id(doc_id: str, index: int) -> str:
    return hashlib.sha256(f"{doc_id}:{index}".encode("utf-8")).hexdigest()


def chunk_document(doc_id: str, doc_type: str, rel_path: str, text: str) -> list[Chunk]:
    """Dispatch to the class-specific chunker."""
    if doc_type == "code" and Path(rel_path).suffix == ".py":
        return chunk_code(doc_id, doc_type, rel_path, text)
    if doc_type in ("regulatory", "security"):
        return chunk_regulatory(doc_id, doc_type, text)
    return chunk_markdown(doc_id, doc_type, text)


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Split one over-long block into `size` windows with `overlap`."""
    step = max(1, size - overlap)
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += step
    return [p for p in (p.strip() for p in pieces) if p]


def group_paragraphs(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Group paragraphs into chunks of <= size chars, with the tail of the
    previous chunk (up to `overlap` chars) prepended to each subsequent chunk
    of the same group. Over-long paragraphs are hard-split with overlap.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("\n\n".join(buf).strip())
            buf.clear()

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            flush()
            out.extend(_hard_split(para, size, overlap))
            continue
        if buf and len("\n\n".join(buf + [para])) > size:
            flush()
            if overlap > 0 and out:
                buf.append(out[-1][-overlap:].strip())
        buf.append(para)
    flush()
    return [c for c in out if c]


# --------------------------------------------------------------------------
# Code — tree-sitter-python symbol-aware chunking
# --------------------------------------------------------------------------

def _make_parser():
    """tree-sitter parser for Python (handles both 0.21 and 0.22+ APIs)."""
    import tree_sitter_python
    from tree_sitter import Language, Parser

    raw = tree_sitter_python.language()
    try:
        language = Language(raw)              # tree-sitter >= 0.22 (capsule)
    except TypeError:
        language = Language(raw, "python")    # tree-sitter 0.21 (pointer)
    try:
        return Parser(language)               # tree-sitter >= 0.22
    except TypeError:
        parser = Parser()
        parser.set_language(language)         # tree-sitter 0.21
        return parser


def _symbol_name(node) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        for child in node.children:
            inner_type = child.type
            if inner_type in ("function_definition", "class_definition"):
                inner = child.child_by_field_name("name")
                if inner is not None:
                    return inner.text.decode("utf-8", "replace")
        return ""
    return name_node.text.decode("utf-8", "replace")


def chunk_code(doc_id: str, doc_type: str, rel_path: str, text: str) -> list[Chunk]:
    """One chunk per top-level function/class; everything else (imports,
    module docstring, assignments) is a single module chunk.
    """
    size, overlap = chunk_params()
    parser = _make_parser()
    source = text.encode("utf-8")
    tree = parser.parse(source)
    root = tree.root_node

    chunks: list[Chunk] = []
    preamble: list[str] = []

    def emit_module_chunk() -> None:
        module_text = "\n".join(preamble).strip()
        preamble.clear()
        if module_text:
            pieces = group_paragraphs(
                [block for block in re.split(r"\n\s*\n", module_text) if block.strip()],
                size, overlap) or ([module_text[:size]] if module_text else [])
            for piece in pieces:
                chunks.append(Chunk(id="", text=piece, section=f"{rel_path}#module",
                                    clause="", code_ref=f"{rel_path}#module",
                                    doc_type=doc_type))

    for node in root.children:
        symbol_type = node.type
        if symbol_type == "decorated_definition":
            inner = [c for c in node.children
                     if c.type in ("function_definition", "class_definition")]
            symbol_type = inner[0].type if inner else symbol_type
        if symbol_type in ("function_definition", "class_definition"):
            symbol = _symbol_name(node) or "<unnamed>"
            emit_module_chunk()
            body = source[node.start_byte:node.end_byte].decode("utf-8", "replace")
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            for piece in (group_paragraphs([body], size, overlap)
                          if len(body) > size else [body]):
                # split_lines keeps one symbol in possibly several chunks
                chunks.append(Chunk(
                    id="", text=piece,
                    section=f"{rel_path}#{symbol}",
                    clause="",
                    code_ref=f"{rel_path}#{symbol}:{start_line}-{end_line}",
                    doc_type=doc_type))
        else:
            preamble.append(source[node.start_byte:node.end_byte]
                            .decode("utf-8", "replace").strip())
    emit_module_chunk()

    for index, chunk in enumerate(chunks):
        chunk.id = _chunk_id(doc_id, index)
    if not chunks and text.strip():
        chunks.append(Chunk(id=_chunk_id(doc_id, 0), text=text.strip()[:size],
                            section=f"{rel_path}#module",
                            code_ref=f"{rel_path}#module", doc_type=doc_type))
    LOG.debug("Code chunking %s: %d chunks", rel_path, len(chunks))
    return chunks


# --------------------------------------------------------------------------
# Markdown — heading-hierarchy chunking (architecture docs)
# --------------------------------------------------------------------------

def chunk_markdown(doc_id: str, doc_type: str, text: str) -> list[Chunk]:
    """Split on ATX headings; `section` keeps the H2+ heading path
    ("BIAN capability model > Business Areas and Service Domains").
    """
    size, overlap = chunk_params()
    stack: list[tuple[int, str]] = []   # (level, title), level 1 = H1 (doc title)
    body_lines: list[str] = []
    chunks: list[Chunk] = []

    def section_path() -> str:
        return " > ".join(title for level, title in stack if level >= 2)

    def emit_section() -> None:
        section = section_path()
        body = "\n".join(body_lines).strip()
        body_lines.clear()
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        pieces = group_paragraphs(paragraphs, size, overlap)
        if not pieces and section:
            pieces = [section]   # keep bare headings addressable for citations
        for piece in pieces:
            chunks.append(Chunk(id="", text=piece, section=section,
                                clause="", code_ref="", doc_type=doc_type))

    in_fence = False
    for line in text.splitlines():
        if _is_fence_toggle(line):
            in_fence = not in_fence
            body_lines.append(line)
            continue
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            emit_section()
            level, title = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body_lines.append(line)
    emit_section()

    for index, chunk in enumerate(chunks):
        chunk.id = _chunk_id(doc_id, index)
    if not chunks and text.strip():
        chunks.append(Chunk(id=_chunk_id(doc_id, 0), text=text.strip()[:size],
                            doc_type=doc_type))
    LOG.debug("Markdown chunking %s: %d chunks", doc_id, len(chunks))
    return chunks


# --------------------------------------------------------------------------
# Regulatory / security — clause-aware chunking
# --------------------------------------------------------------------------

def chunk_regulatory(doc_id: str, doc_type: str, text: str) -> list[Chunk]:
    """Each "### Clause N" heading starts its own chunk (clause="N",
    section="<part heading> > Clause N"); non-clause preamble text (title,
    application notes, part headings without clauses) is grouped semantically
    with overlap.
    """
    size, overlap = chunk_params()
    part_heading = ""
    preamble: list[str] = []
    clause_blocks: list[tuple[str, str, str, str]] = []  # (number, heading, body, part)
    current: tuple[str, str, list[str], str] | None = None

    in_fence = False
    for line in text.splitlines():
        if _is_fence_toggle(line):
            in_fence = not in_fence
        heading = None if in_fence else HEADING_RE.match(line)
        clause = CLAUSE_RE.match(line) if heading else None
        if heading and not clause:
            level, title = len(heading.group(1)), heading.group(2).strip()
            if level <= 2:
                part_heading = title
            if current is not None:
                clause_blocks.append((current[0], current[1],
                                      "\n\n".join(current[2]).strip(),
                                      current[3]))
                current = None
            else:
                preamble.append("")  # blank line between heading groups
            if line.strip():
                preamble.append(line)
            continue
        if clause:
            if current is not None:
                clause_blocks.append((current[0], current[1],
                                      "\n\n".join(current[2]).strip(),
                                      current[3]))
            current = (clause.group(1), line.strip(), [], part_heading)
            continue
        if current is not None:
            current[2].append(line)
        else:
            preamble.append(line)
    if current is not None:
        clause_blocks.append((current[0], current[1],
                              "\n\n".join(current[2]).strip(), current[3]))

    chunks: list[Chunk] = []

    # Preamble: semantic paragraph grouping with overlap. The preamble spans
    # the whole document (title, warnings, clause-less part headings), so it
    # carries no single part heading — section stays empty.
    preamble_section = ""
    preamble_text = "\n\n".join(p for p in preamble if p.strip())
    paragraphs = [p for p in re.split(r"\n\s*\n", preamble_text) if p.strip()]
    for piece in group_paragraphs(paragraphs, size, overlap):
        chunks.append(Chunk(id="", text=piece, section=preamble_section,
                            clause="", code_ref="", doc_type=doc_type))

    # One chunk per clause (clause heading + its body).
    for number, heading, body, part in clause_blocks:
        section = f"{part} > Clause {number}".strip(" >")
        if body:
            pieces = group_paragraphs([body], size, overlap) or [body[:size]]
        else:
            pieces = [heading]
        for piece in pieces:
            chunk_text = f"{heading}\n\n{piece}" if piece != heading else heading
            chunks.append(Chunk(id="", text=chunk_text, section=section,
                                clause=number, code_ref="", doc_type=doc_type))

    for index, chunk in enumerate(chunks):
        chunk.id = _chunk_id(doc_id, index)
    LOG.debug("Regulatory chunking %s: %d chunks (%d clauses)",
              doc_id, len(chunks), len(clause_blocks))
    return chunks