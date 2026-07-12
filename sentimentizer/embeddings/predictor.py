"""Dense embedding predictor for the legacy vectorizer wire contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from sentimentizer import logger
from sentimentizer.device import resolve_device

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    SENTENCE_TRANSFORMERS_AVAILABLE = False

DENSE_DIM = 768
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "


class DenseEmbeddingPredictor:
    """Serve nomic dense embeddings with explicit query/document modes."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str = "auto",
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        factory = model_factory or SentenceTransformer
        if factory is None:
            raise RuntimeError(
                "sentence-transformers is not installed; install the embeddings extra"
            )
        self.model_id = model_id
        self.revision = revision
        self.device = resolve_device(device)
        self._model = factory(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device=self.device,
        )

    def encode(self, texts: Sequence[str], mode: str) -> list[list[float]]:
        """Encode texts in query or document mode while preserving order."""
        if not texts:
            return []
        if mode not in {"query", "document"}:
            raise ValueError(f"unsupported embedding mode {mode!r}")
        started = perf_counter()
        logger.debug(
            "dense_embedding_started",
            input_count=len(texts),
            character_count=sum(len(text) for text in texts),
            mode=mode,
            model=self.model_id,
            device=self.device,
        )
        prefix = QUERY_PREFIX if mode == "query" else DOCUMENT_PREFIX
        vectors = self._model.encode(
            [prefix + text for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        result = [[float(value) for value in vector] for vector in vectors]
        for vector in result:
            if len(vector) != DENSE_DIM:
                raise RuntimeError(
                    f"dense embedder returned {len(vector)} dimensions; expected {DENSE_DIM}"
                )
        logger.debug(
            "dense_embedding_completed",
            input_count=len(result),
            dimensions=DENSE_DIM,
            mode=mode,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
        )
        return result
