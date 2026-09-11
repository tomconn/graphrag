"""Chunk-layer retrieval: dense (vector index), sparse (BM25 fulltext) and a
reciprocal-rank-fusion hybrid. Built on neo4j-graphrag retrievers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import neo4j
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResultItem

_log = logging.getLogger(__name__)

VECTOR_INDEX = "chunk_embeddings"
FULLTEXT_INDEX = "chunk_text_ft"
DEFAULT_TOP_K = 8
RRF_K = 60

CHUNK_PROPS = ("id", "text", "section", "clause", "code_ref")

SPARSE_CYPHER = (
    "CALL db.index.fulltext.queryNodes($index_name, $query) "
    "YIELD node, score "
    "RETURN node { .id, .text, .section, .clause, .code_ref } AS node, score "
    "ORDER BY score DESC LIMIT $top_k"
)


def chunk_formatter(record: neo4j.Record) -> RetrieverResultItem:
    """Shared formatter: record['node'] is a map projection of chunk props."""
    node = dict(record.get("node") or {})
    metadata = {prop: node.get(prop) for prop in CHUNK_PROPS}
    metadata["score"] = record.get("score")
    return RetrieverResultItem(content=node.get("text", ""), metadata=metadata)


def _to_chunks(result) -> list[dict[str, Any]]:
    """Normalize a neo4j-graphrag search() result into plain chunk dicts."""
    chunks: list[dict[str, Any]] = []
    for item in result.items:
        meta = item.metadata or {}
        chunks.append(
            {
                "id": meta.get("id") or "",
                "text": item.content or "",
                "section": meta.get("section") or "",
                "clause": meta.get("clause") or "",
                "code_ref": meta.get("code_ref") or "",
                "score": float(meta.get("score") or 0.0),
                "kind": "chunk",
            }
        )
    return chunks


class BM25Retriever(Retriever):
    """Sparse retrieval over the Chunk fulltext index (BM25), implementing the
    neo4j-graphrag retriever interface so it composes with the dense side."""

    index_name: str = FULLTEXT_INDEX

    def __init__(
        self, driver: neo4j.Driver, index_name: str = FULLTEXT_INDEX
    ) -> None:
        super().__init__(driver)
        self.index_name = index_name
        self.result_formatter = chunk_formatter

    def get_search_results(
        self, query_text: str, top_k: int = 5
    ) -> RawSearchResult:
        query = _sanitize_fulltext(query_text)
        if not query:
            return RawSearchResult(records=[])

        def work(tx: neo4j.Transaction) -> list[neo4j.Record]:
            return list(
                tx.run(
                    SPARSE_CYPHER,
                    index_name=self.index_name,
                    query=query,
                    top_k=top_k,
                )
            )

        with self.driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
            records = session.execute_read(work)
        return RawSearchResult(records=records)


def _sanitize_fulltext(query: str) -> str:
    """Strip Lucene special characters that would break the fulltext query."""
    cleaned = re.sub(r"[^\w\s]", " ", query)
    return re.sub(r"\s+", " ", cleaned).strip()


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]], k: int = RRF_K
) -> list[dict[str, Any]]:
    """Fuse ranked chunk lists by reciprocal rank (k=60), deduped by id."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            chunk_id = chunk.get("id") or ""
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(chunk_id, chunk)
    fused = sorted(
        (by_id[cid] for cid in scores), key=lambda c: scores[c["id"]], reverse=True
    )
    return fused


class RetrievalService:
    """Dense / sparse / hybrid retrieval against the Chunk layer."""

    def __init__(self, driver: neo4j.Driver) -> None:
        from app.embedder import FastEmbedEmbedder

        embedder = FastEmbedEmbedder()
        self.dense = VectorRetriever(
            driver,
            VECTOR_INDEX,
            embedder,
            result_formatter=chunk_formatter,
        )
        self.sparse = BM25Retriever(driver, FULLTEXT_INDEX)

    def search(
        self, query: str, mode: str = "hybrid", top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:
        """mode: 'hybrid' (RRF of dense+sparse) or 'vector' (dense only)."""
        if mode == "vector":
            chunks = _to_chunks(self.dense.search(query_text=query, top_k=top_k))
            _log.info(
                "dense retrieval (mode=vector) query=%r chunks=%d", query, len(chunks)
            )
            return chunks
        dense = _to_chunks(self.dense.search(query_text=query, top_k=top_k))
        sparse = _to_chunks(self.sparse.search(query_text=query, top_k=top_k))
        fused = reciprocal_rank_fusion(dense, sparse)[:top_k]
        _log.info(
            "hybrid retrieval query=%r dense=%d sparse=%d fused=%d",
            query,
            len(dense),
            len(sparse),
            len(fused),
        )
        return fused