"""Corpus walk and classification.

Walks the bind-mounted data root (/app/data), classifies each file by its
parent directory name (regulatory | security | architecture | code), skips
meta files (data/README.md, hidden files), and reads each document.

PDFs are handled only if `docling` is importable — docling is deliberately
NOT in requirements.txt (heavy); without it a PDF raises a clear error
telling the operator to convert manually.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

DOC_TYPES = ("regulatory", "security", "architecture", "code")
TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
PDF_EXTENSIONS = {".pdf"}
CODE_EXTENSIONS = {".py"}
SKIPPED_FILES = {"README.md"}  # meta documentation at the data root
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | CODE_EXTENSIONS


@dataclass(frozen=True)
class SourceDocument:
    """A single corpus file, classified and read."""

    doc_id: str          # stable id: file stem (collision-suffixed, see from_path)
    doc_type: str        # regulatory | security | architecture | code
    title: str           # first H1 heading text, else file stem
    source_path: str     # e.g. "data/regulatory/cps-234-information-security.md"
    abs_path: str
    text: str

    @staticmethod
    def from_path(abs_path: Path, doc_type: str, data_root: Path, text: str,
                  title: str) -> "SourceDocument":
        stem = abs_path.stem
        rel = abs_path.relative_to(data_root)
        return SourceDocument(
            doc_id=stem,
            doc_type=doc_type,
            title=title or stem,
            source_path=f"{data_root.name}/{rel.as_posix()}",
            abs_path=str(abs_path),
            text=text,
        )


def classify(parent_dir: str) -> str | None:
    """Document class from the file's parent directory name."""
    if parent_dir in DOC_TYPES:
        return parent_dir
    return None


def doc_title(text: str, fallback: str) -> str:
    """Title = first ATX level-1 heading text, else the file stem."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return fallback


def pdf_to_text(abs_path: Path) -> str:
    """Convert a PDF to markdown with docling, if available.

    Docling is optional and heavy (not in requirements.txt). Without it the
    operator must convert the PDF manually and drop the markdown next to it.
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot ingest PDF {abs_path}: docling is not installed in the "
            "ingest image (it is optional and heavy on purpose). Either "
            "convert the PDF to markdown manually and place the .md file in "
            "the same data/<class>/ folder, or install docling locally "
            "(pip install docling) and pre-convert before running ingest."
        ) from exc
    LOG.info("Converting PDF %s with docling ...", abs_path)
    converter = DocumentConverter()
    result = converter.convert(str(abs_path))
    return result.document.export_to_markdown()


def read_file(abs_path: Path, doc_type: str) -> str:
    """Read a document's text; PDFs are converted via docling when present."""
    suffix = abs_path.suffix.lower()
    if suffix in CODE_EXTENSIONS or suffix in TEXT_EXTENSIONS:
        return abs_path.read_text(encoding="utf-8", errors="replace")
    if suffix in PDF_EXTENSIONS:
        return pdf_to_text(abs_path)
    raise ValueError(f"Unsupported file extension {suffix!r}: {abs_path}")


def walk_data(data_root: str | os.PathLike[str]) -> list[tuple[Path, str]]:
    """Return (abs_path, doc_type) for every ingestible file, in a
    deterministic order (doc-type order, then sorted paths).

    Missing class directories are logged and skipped (e.g. an empty
    data/code/). Unknown subdirectory layouts: a file is classified by its
    immediate parent directory name only.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {root} (expected bind mount of data/)"
        )
    found: list[tuple[Path, str]] = []
    for doc_type in DOC_TYPES:
        class_dir = root / doc_type
        if not class_dir.is_dir():
            LOG.info("data/%s/ does not exist — nothing to ingest there.", doc_type)
            continue
        for abs_path in sorted(class_dir.rglob("*")):
            if not abs_path.is_file():
                continue
            if abs_path.name in SKIPPED_FILES or abs_path.name.startswith("."):
                LOG.info("Skipping meta/hidden file %s", abs_path)
                continue
            if abs_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                LOG.warning("Skipping unsupported file %s", abs_path)
                continue
            found.append((abs_path, doc_type))
    return found


def load_documents(data_root: str | os.PathLike[str]) -> list[SourceDocument]:
    """Read every ingestible file into SourceDocument records.

    Doc ids are the file stems; a collision within the corpus (same stem in
    two class directories) is disambiguated with a short hash suffix.
    """
    root = Path(data_root)
    documents: list[SourceDocument] = []
    seen_ids: dict[str, str] = {}
    for abs_path, doc_type in walk_data(root):
        text = read_file(abs_path, doc_type)
        doc = SourceDocument.from_path(abs_path, doc_type, root, text,
                                       doc_title(text, abs_path.stem))
        if doc.doc_id in seen_ids:
            import hashlib
            suffix = hashlib.sha256(doc.source_path.encode("utf-8")).hexdigest()[:8]
            LOG.warning("Duplicate doc id %r (%s vs %s) — suffixing %s",
                        doc.doc_id, seen_ids[doc.doc_id], doc.source_path, suffix)
            doc = SourceDocument(  # replace with disambiguated id
                doc_id=f"{doc.doc_id}-{suffix}",
                doc_type=doc.doc_type,
                title=doc.title,
                source_path=doc.source_path,
                abs_path=doc.abs_path,
                text=doc.text,
            )
        seen_ids[doc.doc_id] = doc.source_path
        documents.append(doc)
    return documents