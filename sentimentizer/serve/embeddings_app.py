"""Ray Serve deployment that owns optional dense and BGE-M3 predictors."""

from __future__ import annotations

from typing import Any

from sentimentizer import logger
from sentimentizer.embeddings import BGEM3Predictor, DenseEmbeddingPredictor
from sentimentizer.serve.base import serve
from sentimentizer.serve.config import cfg


@serve.deployment(
    num_replicas=cfg.bge_m3_num_replicas,
    max_ongoing_requests=cfg.bge_m3_max_ongoing_requests,
    ray_actor_options={
        "num_cpus": cfg.bge_m3_num_cpus,
        "num_gpus": cfg.bge_m3_num_gpus,
    },
)
class EmbeddingsDeployment:
    """Own embedding model lifecycles and expose methods to the HTTP ingress."""

    def __init__(self) -> None:
        self._dense = (
            DenseEmbeddingPredictor(
                model_id=cfg.dense_embedding_model_id,
                revision=cfg.dense_embedding_revision,
                device=cfg.embeddings_device,
            )
            if cfg.embeddings_enabled
            else None
        )
        self._bge_m3 = (
            BGEM3Predictor(
                model_id=cfg.bge_m3_model_id,
                device=cfg.embeddings_device,
                use_fp16=cfg.bge_m3_use_fp16,
                batch_size=cfg.bge_m3_batch_size,
            )
            if cfg.bge_m3_enabled
            else None
        )

    def dense(self, texts: list[str], mode: str) -> list[list[float]]:
        """Return dense embeddings for a query or document batch."""
        if self._dense is None:
            raise RuntimeError("dense embeddings are disabled")
        logger.debug("dense_embedding_request", input_count=len(texts), mode=mode)
        return self._dense.encode(texts, mode)

    @serve.batch(
        max_batch_size=cfg.bge_m3_batch_size,
        batch_wait_timeout_s=cfg.bge_m3_batch_wait_s,
    )
    async def bge_m3(self, texts: list[str]) -> list[dict[str, Any]]:
        """Auto-batch single-text BGE-M3 calls across HTTP requests."""
        if self._bge_m3 is None:
            raise RuntimeError("BGE-M3 embeddings are disabled")
        logger.debug("bge_m3_embedding_request", input_count=len(texts))
        return self._bge_m3.encode(texts)
