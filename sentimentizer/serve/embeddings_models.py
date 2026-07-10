"""Request and response models for embedding endpoints."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from sentimentizer.serve.config import cfg


class VectorRequest(BaseModel):
    """Single dense vector request."""

    text: Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]
    mode: Literal["query", "document"] = "query"


class VectorBatchRequest(BaseModel):
    """Batch dense vector request."""

    texts: list[Annotated[str, Field(min_length=1, max_length=cfg.max_text_length)]] = Field(
        min_length=1,
        max_length=cfg.max_batch_size,
    )
    normalize: bool = True
    mode: Literal["query", "document"] = "document"


class EmbeddingsRequest(BaseModel):
    """Batch BGE-M3 hybrid embedding request."""

    texts: list[Annotated[str, Field(min_length=1, max_length=10000)]] = Field(
        min_length=1,
        max_length=64,
    )


class EmbeddingVector(BaseModel):
    """Dense and learned-sparse representation for one text."""

    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class EmbeddingsResult(BaseModel):
    """BGE-M3 embedding response."""

    backend: str
    model: str
    vectors: list[EmbeddingVector]
