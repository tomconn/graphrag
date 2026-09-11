"""FastEmbed dense embeddings (int8-quantized ONNX, CPU only).

The model must be identical to the agent's (EMBEDDING_MODEL env) so chunks
and queries share one vector space. EMBEDDING_DIM is asserted against the
model's actual output dimensionality at startup.
"""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger(__name__)

EMBED_BATCH = 64


class Embedder:
    """Dense-only embedder on top of fastembed.TextEmbedding."""

    def __init__(self, model_name: str | None = None, expected_dim: int | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
        self.expected_dim = int(expected_dim if expected_dim is not None
                                else os.environ.get("EMBEDDING_DIM", "1024"))
        LOG.info("Loading FastEmbed model %s (CPU, ONNX) ...", self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)

        probe = list(self._model.embed(["dimension probe"]))
        actual_dim = len(probe[0])
        if actual_dim != self.expected_dim:
            raise ValueError(
                f"EMBEDDING_DIM={self.expected_dim} does not match the "
                f"{self.model_name} output dimension ({actual_dim}). Fix "
                "EMBEDDING_DIM (must also match the agent container and the "
                "chunk_embeddings vector index).")
        self.dim = actual_dim
        LOG.info("FastEmbed ready: %s, %d dims", self.model_name, self.dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Dense vectors for a batch of texts (order preserved)."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            for vector in self._model.embed(batch):
                vectors.append([float(x) for x in vector])
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"FastEmbed returned {len(vectors)} vectors for "
                f"{len(texts)} texts")
        return vectors

    def embed_chunks(self, chunks: list) -> None:
        """Fill chunk.embedding in place for a list of chunk.Chunk."""
        if not chunks:
            return
        vectors = self.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector