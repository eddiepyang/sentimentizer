"""Embedding predictors used by the Ray Serve embeddings deployment."""

from sentimentizer.embeddings.bge_m3 import BGEM3Predictor
from sentimentizer.embeddings.predictor import DenseEmbeddingPredictor

__all__ = ["BGEM3Predictor", "DenseEmbeddingPredictor"]
