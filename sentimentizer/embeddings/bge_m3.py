"""BGE-M3 dense and learned-sparse embedding predictor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sentimentizer import logger
from sentimentizer.device import resolve_device

try:
    from FlagEmbedding import BGEM3FlagModel

    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    BGEM3FlagModel = None  # type: ignore[assignment,misc]
    FLAG_EMBEDDING_AVAILABLE = False

DENSE_DIM = 1024
SPARSE_VOCAB = 250002
MAX_SPARSE_NNZ = 16000


class BGEM3Predictor:
    """Generate BGE-M3 dense and learned-sparse vectors."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "auto",
        use_fp16: bool = False,
        batch_size: int = 12,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        resolved_device = resolve_device(device)
        self.model_id = model_id
        self.device = resolved_device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and resolved_device.startswith("cuda")
        if use_fp16 and not self.use_fp16:
            logger.info(
                "bge_m3_fp16_disabled",
                device=resolved_device,
                reason="fp16 is enabled only on CUDA",
            )

        factory = model_factory or BGEM3FlagModel
        if factory is None:
            raise RuntimeError("FlagEmbedding is not installed; install the embeddings extra")
        self._model = factory(
            model_id,
            use_fp16=self.use_fp16,
            devices=[resolved_device],
        )

    def encode(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """Encode texts in input order into JSON-serializable hybrid vectors."""
        if not texts:
            return []
        output = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]
        if len(dense_vectors) != len(texts) or len(lexical_weights) != len(texts):
            raise RuntimeError("BGE-M3 returned a different number of vectors than input texts")

        vectors: list[dict[str, Any]] = []
        for dense, sparse in zip(dense_vectors, lexical_weights, strict=True):
            dense_values = [float(value) for value in dense]
            if len(dense_values) != DENSE_DIM:
                raise RuntimeError(
                    f"BGE-M3 returned {len(dense_values)} dense dimensions; expected {DENSE_DIM}"
                )
            indices, values = _sorted_sparse(sparse)
            if len(indices) > MAX_SPARSE_NNZ:
                raise RuntimeError(
                    f"BGE-M3 sparse vector has {len(indices)} non-zero values; "
                    f"pgvector permits at most {MAX_SPARSE_NNZ}"
                )
            vectors.append(
                {
                    "dense": dense_values,
                    "sparse_indices": indices,
                    "sparse_values": values,
                }
            )
        return vectors


def _sorted_sparse(weights: Mapping[Any, Any]) -> tuple[list[int], list[float]]:
    """Convert lexical weights into sorted parallel index and value arrays."""
    pairs = sorted((int(index), float(value)) for index, value in weights.items())
    return [index for index, _ in pairs], [value for _, value in pairs]
