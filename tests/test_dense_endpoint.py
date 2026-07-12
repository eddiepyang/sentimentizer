"""Tests for the JSON nomic dense-embeddings endpoint."""

import asyncio
from types import SimpleNamespace

import pytest

from sentimentizer.serve.app import SentimentizerDeployment as _Deployment
from sentimentizer.serve.config import cfg
from sentimentizer.serve.embeddings_models import DenseEmbeddingsRequest

_SentimentizerDeployment = _Deployment.func_or_class


class _DenseHandle:
    async def remote(self, texts: list[str], mode: str) -> list[list[float]]:
        assert texts == ["first", "second"]
        assert mode == "document"
        return [[0.1, 0.2], [0.3, 0.4]]


def test_dense_embeddings_returns_model_and_vectors() -> None:
    deployment = SimpleNamespace(
        _require_embeddings=lambda: None,
        _embeddings_handle=SimpleNamespace(dense=_DenseHandle()),
    )

    result = asyncio.run(
        _SentimentizerDeployment.dense_embeddings(
            deployment,
            DenseEmbeddingsRequest(texts=["first", "second"], mode="document"),
        )
    )

    assert result == {
        "model": cfg.dense_embedding_model_id,
        "vectors": [[0.1, 0.2], [0.3, 0.4]],
    }


def test_dense_embeddings_request_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        DenseEmbeddingsRequest(texts=[])
