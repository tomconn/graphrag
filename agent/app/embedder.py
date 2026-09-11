"""FastEmbed adapter implementing the neo4j-graphrag ``Embedder`` interface.

The ingest job must use the identical model (``EMBEDDING_MODEL``) so queries
and chunks share one vector space.
"""

from __future__ import annotations

import logging
import os
import threading

from neo4j_graphrag.embeddings.base import Embedder

_log = logging.getLogger(__name__)

_DEFAULT_EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"


class FastEmbedEmbedder(Embedder):
    """neo4j-graphrag ``Embedder`` backed by FastEmbed (ONNX, CPU).

    The model is loaded lazily on first embed so /health and startup stay
    fast and an unused agent never pays the model load.
    """

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self._model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL
        )
        self._text_embedding = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._text_embedding is None:
            with self._lock:
                if self._text_embedding is None:
                    from fastembed import TextEmbedding  # deferred: heavy import

                    _log.info("loading FastEmbed model %s", self._model_name)
                    self._text_embedding = TextEmbedding(
                        model_name=self._model_name, lazy_load=True
                    )
        return self._text_embedding

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = next(model.query_embed(text))
        return [float(x) for x in vector]