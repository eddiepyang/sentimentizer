"""RouterModel: sentence-transformer backbone + LogisticRegression head.

Drop-in replacement for SetFitModel with equivalent API:
- from_pretrained() / save_pretrained() / push_to_hub()
- model_encode() / predict() / predict_proba()

The model consists of:
- backbone: SentenceTransformer for encoding text to embeddings
- head: LogisticRegression for classification on embeddings
- labels: List of category names (e.g., ["dietary", "service", "general"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — these are optional dependencies
_sentence_transformers: Any = None
_joblib: Any = None


def _get_sentence_transformers() -> Any:
    """Lazy import sentence_transformers."""
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer

        _sentence_transformers = SentenceTransformer
    return _sentence_transformers


def _get_joblib() -> Any:
    """Lazy import joblib."""
    global _joblib
    if _joblib is None:
        import joblib

        _joblib = joblib
    return _joblib


class RouterModel:
    """Sentence-transformer backbone + LogisticRegression head.

    Drop-in replacement for SetFitModel with equivalent API:
    - from_pretrained() / save_pretrained() / push_to_hub()
    - model_encode() / predict() / predict_proba()
    """

    def __init__(
        self,
        backbone: Any,
        head: Any | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Initialize RouterModel.

        Args:
            backbone: SentenceTransformer model for encoding.
            head: Sklearn classifier (e.g., LogisticRegression). If None,
                  predict/predict_proba will raise RuntimeError.
            labels: List of label names in index order.
        """
        self.backbone = backbone
        self.head = head
        self.labels = labels or []

    # ------------------------------------------------------------------
    # Class methods: loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str | Path) -> RouterModel:
        """Load a RouterModel from a local directory.

        Expects the directory to contain:
        - SentenceTransformer backbone files (model.safetensors, config.json, etc.)
        - router_head.joblib (serialized sklearn classifier)
        - router_config.json (metadata with labels, model_type, etc.)

        Falls back to legacy SetFit model loading if router_head.joblib
        is not found (for backward compatibility with previously saved models).

        Args:
            path: Path to the saved model directory.

        Returns:
            Loaded RouterModel.

        Raises:
            FileNotFoundError: If the path doesn't exist.
            RuntimeError: If the model cannot be loaded.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        SentenceTransformer = _get_sentence_transformers()
        joblib = _get_joblib()

        # Load backbone
        logger.info(f"Loading backbone from {path}")
        backbone = SentenceTransformer(str(path))

        # Load head
        head_path = path / "router_head.joblib"
        if head_path.exists():
            logger.info(f"Loading classification head from {head_path}")
            head = joblib.load(head_path)
        else:
            logger.warning(
                f"No router_head.joblib found at {path}. Attempting legacy SetFit model migration."
            )
            return cls._migrate_legacy_setfit_model(path, backbone)

        # Load config
        config_path = path / "router_config.json"
        labels = []
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            labels = config.get("labels", [])
            logger.info(f"Loaded router config: labels={labels}")

        return cls(backbone=backbone, head=head, labels=labels)

    @classmethod
    def _migrate_legacy_setfit_model(cls, path: Path, backbone: Any) -> RouterModel:
        """Migrate a legacy SetFit-format model to RouterModel.

        Legacy SetFit models store the head as model_head.pkl and
        labels in config_setfit.json. This method loads those files
        and returns a RouterModel without the setfit dependency.

        Args:
            path: Path to the SetFit model directory.
            backbone: Already-loaded SentenceTransformer backbone.

        Returns:
            RouterModel with migrated head and labels.

        Raises:
            RuntimeError: If migration fails.
        """
        joblib = _get_joblib()

        # Try loading SetFit's head format (model_head.pkl or model_head.pickle)
        head = None
        for head_name in ("model_head.pkl", "model_head.pickle"):
            head_path = path / head_name
            if head_path.exists():
                try:
                    head = joblib.load(head_path)
                    logger.info(f"Migrated SetFit head from {head_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load SetFit head from {head_path}: {e}")

        # Try loading labels from config_setfit.json
        labels = []
        config_path = path / "config_setfit.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                labels = config.get("labels", [])
                if isinstance(labels, dict):
                    # SetFit stores labels as {idx: name}, convert to list
                    labels = [labels[str(i)] for i in range(len(labels))]
                logger.info(f"Migrated SetFit labels: {labels}")
            except Exception as e:
                logger.warning(f"Failed to load SetFit config from {config_path}: {e}")

        if head is None:
            raise RuntimeError(
                f"Cannot migrate SetFit model at {path}: "
                f"no model_head.pkl or model_head.pickle found. "
                f"Please retrain the router model using the updated training pipeline."
            )

        return cls(backbone=backbone, head=head, labels=labels)

    # ------------------------------------------------------------------
    # Save / push
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str | Path) -> None:
        """Save the RouterModel to a local directory.

        Saves:
        - SentenceTransformer backbone files
        - router_head.joblib (serialized sklearn classifier)
        - router_config.json (metadata)

        Args:
            path: Directory to save the model to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib = _get_joblib()

        # Save backbone
        logger.info(f"Saving backbone to {path}")
        self.backbone.save(str(path))

        # Save head
        if self.head is not None:
            head_path = path / "router_head.joblib"
            joblib.dump(self.head, head_path)
            logger.info(f"Saved classification head to {head_path}")

        # Save config
        config = {
            "model_type": "router",
            "labels": self.labels,
            "head_type": type(self.head).__name__ if self.head else None,
        }
        config_path = path / "router_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved router config to {config_path}")

    def push_to_hub(self, repo_id: str, **kwargs: Any) -> None:
        """Push the RouterModel to Hugging Face Hub.

        Saves the model to a temporary directory first, then uploads.

        Args:
            repo_id: Hugging Face repository ID (e.g., "user/model").
            **kwargs: Additional arguments passed to huggingface_hub.
        """
        import tempfile

        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api = HfApi()
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                **kwargs,
            )
        logger.info(f"Pushed RouterModel to {repo_id}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def model_encode(self, texts: list[str] | str) -> np.ndarray:
        """Encode texts to embeddings using the backbone.

        Args:
            texts: Single text or list of texts to encode.

        Returns:
            Numpy array of shape (N, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.backbone.encode(texts, convert_to_numpy=True)

    def predict(self, texts: list[str] | str) -> list | np.ndarray:
        """Predict class labels for texts.

        Args:
            texts: Single text or list of texts.

        Returns:
            List of predicted labels (ints or strings depending on head).
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model_encode(texts)
        return self.head.predict(embeddings)

    def predict_proba(self, texts: list[str] | str) -> np.ndarray:
        """Predict class probabilities for texts.

        Args:
            texts: Single text or list of texts.

        Returns:
            Numpy array of shape (N, num_classes) with class probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model_encode(texts)
        return self.head.predict_proba(embeddings)
