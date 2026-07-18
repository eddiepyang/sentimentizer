"""Tests for BGE-M3 predictor conversion and optional serving schema."""

import os

import numpy as np
import pytest

from sentimentizer.embeddings.bge_m3 import DENSE_DIM, SPARSE_VOCAB, BGEM3Predictor
from sentimentizer.serve.embeddings_models import EmbeddingsRequest


class _FakeBGEModel:
    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs

    def encode(self, texts, **_kwargs):
        return {
            "dense_vecs": np.zeros((len(texts), DENSE_DIM), dtype=np.float32),
            "lexical_weights": [
                {"12": 0.25, "2": 0.5} if text != "empty" else {} for text in texts
            ],
        }


def test_bge_m3_sorts_sparse_and_preserves_batch_order():
    predictor = BGEM3Predictor(device="cpu", use_fp16=True, model_factory=_FakeBGEModel)

    result = predictor.encode(["first", "empty", "third"])

    assert [item["sparse_indices"] for item in result] == [[2, 12], [], [2, 12]]
    assert result[0]["sparse_values"] == [0.5, 0.25]
    assert all(len(item["dense"]) == DENSE_DIM for item in result)
    assert predictor.use_fp16 is False


def test_embeddings_request_limits():
    with pytest.raises(ValueError):
        EmbeddingsRequest(texts=[])
    with pytest.raises(ValueError):
        EmbeddingsRequest(texts=["x"] * 65)


@pytest.mark.skipif(
    os.environ.get("SENTIMENTIZER_BGE_M3_IT") != "1",
    reason="set SENTIMENTIZER_BGE_M3_IT=1 to load the real BGE-M3 model",
)
def test_real_bge_m3_shape():
    predictor = BGEM3Predictor(device="auto", batch_size=1)

    vector = predictor.encode(["margherita pizza"])[0]

    assert len(vector["dense"]) == DENSE_DIM
    assert vector["sparse_indices"] == sorted(vector["sparse_indices"])
    assert vector["sparse_indices"]
    assert len(vector["sparse_indices"]) < 16000
    assert max(vector["sparse_indices"]) < SPARSE_VOCAB
