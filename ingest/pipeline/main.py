"""Ingest orchestration: walk data/ -> parse -> chunk -> embed -> extract
-> write, per document, with per-file progress logs and a final summary.

Run as:  python -m pipeline.main   (container CMD)

Chunk-level LLM extraction runs concurrently, bounded by INGEST_CONCURRENCY
(default 6). Entity MERGEs are protected by uniqueness constraints on every
knowledge label's merge key (see pipeline/write.py), so concurrent writes
stay idempotent. Per-document writes remain sequential (batched).

Exit codes: 0 on success (per-file failures are logged and skipped),
non-zero on a fatal error (config, connectivity), 75 when the stall
watchdog force-exits a wedged run (see pipeline/logsetup.py).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

LOG = logging.getLogger("ingest")

DEFAULT_CONCURRENCY = 6


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name} is not set "
            "(see .env.example / docker-compose.yml).")
    return value


def process_document(doc, chunker, embedder, extractor, writer,
                     counters: Counter,
                     seen_entities: set | None = None,
                     seen_edges: set | None = None,
                     concurrency: int = 1,
                     timings: dict | None = None) -> dict:
    """Parse->chunk->embed->extract->write for one document.

    Extraction failures degrade gracefully: the retrieval layer (Document +
    Chunk nodes with embeddings) is still written, the extraction is skipped
    with a warning, and the failure is counted.

    seen_entities / seen_edges accumulate unique (label, key) / (type, source,
    target) tuples across documents for the final summary. Stage wall time is
    accumulated into `timings` {"embed": s, "extract": s, "write": s}.
    """
    stats = {"chunks": 0, "entities": 0, "edges": 0, "failed_chunks": 0}
    seen_entities = seen_entities if seen_entities is not None else set()
    seen_edges = seen_edges if seen_edges is not None else set()
    timings = timings if timings is not None else {"embed": 0.0, "extract": 0.0,
                                                   "write": 0.0}

    chunks = chunker(doc.doc_id, doc.doc_type, doc.source_path, doc.text)
    if not chunks:
        LOG.warning("[%s] produced no chunks — nothing written", doc.doc_id)
        return stats

    stage = time.monotonic()
    embedder.embed_chunks(chunks)
    timings["embed"] += time.monotonic() - stage

    stage = time.monotonic()
    writer.replace_document(doc, chunks)
    timings["write"] += time.monotonic() - stage
    stats["chunks"] = len(chunks)
    counters["documents"] += 1
    counters["chunks"] += len(chunks)

    # Extraction: LLM calls for this document's chunks run concurrently,
    # bounded by `concurrency` workers.
    results: dict[str, tuple] = {}
    stage = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(extractor.extract, chunk, doc): chunk
                   for chunk in chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                results[chunk.id] = future.result()
            except Exception as exc:  # noqa: BLE001 — keep the run alive
                stats["failed_chunks"] += 1
                counters["failed_chunks"] += 1
                LOG.warning("[%s] chunk %s extraction failed, skipping: %s",
                            doc.doc_id, chunk.id[:12], exc)
                results[chunk.id] = ([], [])
    timings["extract"] += time.monotonic() - stage

    # Writes stay in chunk order (edge provenance: last write wins, per the
    # contract) — fast batched MERGEs, protected by the uniqueness constraints.
    stage = time.monotonic()
    for chunk in chunks:
        entities, relations = results.get(chunk.id, ([], []))
        writer.write_entities(doc.doc_id, entities)
        writer.write_edges(doc.doc_id, chunk.id, relations)
        stats["entities"] += len(entities)
        stats["edges"] += len(relations)
        for entity in entities:
            seen_entities.add((entity.label, entity.key))
        for relation in relations:
            seen_edges.add((relation.rel_type, relation.source_label,
                            relation.source_key, relation.target_label,
                            relation.target_key))
    timings["write"] += time.monotonic() - stage

    LOG.info(
        "[%s] %s: %d chunks, %d entities, %d edges (%d failed extractions)",
        doc.doc_type, doc.doc_id, stats["chunks"], stats["entities"],
        stats["edges"], stats["failed_chunks"])
    return stats


def run() -> None:
    started = time.monotonic()
    from pipeline.logsetup import setup_logging
    setup_logging()

    data_dir = os.environ.get("DATA_DIR", "/app/data")
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = _require("NEO4J_PASSWORD")
    concurrency = max(1, _int_env("INGEST_CONCURRENCY", DEFAULT_CONCURRENCY))

    # Local imports so config errors surface before library load time.
    from pipeline.parse import load_documents
    from pipeline.chunk import chunk_document, chunk_params
    from pipeline.embed import Embedder
    from pipeline.extract import Extractor
    from pipeline.write import GraphWriter

    chunk_size, chunk_overlap = chunk_params()
    LOG.info("Ingest starting: data=%s neo4j=%s chunk_size=%d overlap=%d "
             "concurrency=%d", data_dir, neo4j_uri, chunk_size, chunk_overlap,
             concurrency)

    documents = load_documents(data_dir)
    LOG.info("Found %d ingestible documents", len(documents))
    if not documents:
        LOG.warning("Nothing to ingest — data/ is empty or unsupported.")

    extractor = Extractor()
    knowledge_labels = {label: spec["key"]
                        for label, spec in extractor.node_labels.items()}
    writer = GraphWriter(neo4j_uri, neo4j_user, neo4j_password,
                         _int_env("EMBEDDING_DIM", 1024),
                         knowledge_labels=knowledge_labels)
    embedder = Embedder()
    counters: Counter = Counter()
    seen_entities: set = set()   # unique (label, key) merged in the graph
    seen_edges: set = set()      # unique (type, source, target) merged
    timings = {"embed": 0.0, "extract": 0.0, "write": 0.0}

    try:
        for doc in documents:
            process_document(doc, chunk_document, embedder, extractor,
                             writer, counters, seen_entities, seen_edges,
                             concurrency=concurrency, timings=timings)
    finally:
        writer.close()

    elapsed = time.monotonic() - started
    entities_by_label = sorted(
        (label, sum(1 for l, _ in seen_entities if l == label))
        for label in {label for label, _ in seen_entities})
    edges_by_type = sorted(
        (rel_type, sum(1 for t, *_ in seen_edges if t == rel_type))
        for rel_type in {t for t, *_ in seen_edges})
    LOG.info("=" * 60)
    LOG.info("INGEST SUMMARY")
    LOG.info("  documents : %d", counters["documents"])
    LOG.info("  chunks    : %d", counters["chunks"])
    LOG.info("  entities  : %d unique  %s", len(seen_entities),
             ", ".join(f"{label}={n}" for label, n in entities_by_label))
    LOG.info("  edges     : %d unique  %s", len(seen_edges),
             ", ".join(f"{rel}={n}" for rel, n in edges_by_type))
    LOG.info("  extraction failures (chunks skipped): %d",
             counters["failed_chunks"])
    LOG.info("  wall time : %.1fs (concurrency=%d)", elapsed, concurrency)
    LOG.info("  stages    : embed %.1fs, extract %.1fs, write %.1fs",
             timings["embed"], timings["extract"], timings["write"])
    LOG.info("=" * 60)


def main() -> None:
    try:
        run()
    except Exception as exc:  # noqa: BLE001 — fatal, report and exit non-zero
        LOG.exception("Fatal error during ingest: %s", exc)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()