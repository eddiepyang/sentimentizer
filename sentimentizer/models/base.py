"""Base class and registry for sentiment analysis models.

Provides:
- BaseSentimentModel: Shared predict() and predict_text() methods for inference
- MODEL_REGISTRY: Dict mapping model_type → (model_class, config_class, new_model_func)
- get_trained_model: Unified model loading from registry (replaces per-file duplicates)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from sentimentizer.config import VALID_DEVICES

if TYPE_CHECKING:
    from sentimentizer.tokenizer import Tokenizer


class BaseSentimentModel(nn.Module):
    """Base class for all sentiment analysis models.

    Provides the shared predict() and predict_text() methods that were
    previously duplicated across RNN, Encoder, and Decoder. Subclasses
    must implement forward().

    Output contract (3-class):
    - forward(): returns (B, num_classes) raw logits — never squeeze
    - predict(): returns (B, num_classes) softmax probabilities
    - predict_text(): returns dict[str, float] mapping label names to
      probabilities, e.g. {"negative": 0.05, "neutral": 0.12, "positive": 0.83}
    """

    LABEL_NAMES: list[str] = ["negative", "neutral", "positive"]
    NUM_CLASSES: int = 3

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer  # Tokenizer will be set externally after model creation

    def predict(self, converted_text: np.ndarray) -> torch.Tensor:
        """Run inference with softmax activation.

        Args:
            converted_text: Token IDs as numpy array, shape (B, seq_len)

        Returns:
            Probability matrix of shape (B, num_classes) with softmax
            probabilities for each class [negative, neutral, positive].
        """
        with torch.no_grad():
            self.eval()
            device = next(self.parameters()).device
            output = torch.from_numpy(converted_text).to(device)
            return torch.softmax(self.forward(output), dim=-1)

    def predict_text(self, text: str) -> dict[str, float]:
        """Tokenize raw text and run sentiment prediction in one call.

        Combines tokenization and model inference, eliminating the need
        to call tokenizer.tokenize_text() and model.predict() separately.

        Args:
            text: Raw text string to classify.

        Returns:
            Dict mapping label names to probabilities, e.g.
            {"negative": 0.05, "neutral": 0.12, "positive": 0.83}.
        """
        token_ids = self.tokenizer.tokenize_text(text)
        probs = self.predict(token_ids)  # (1, num_classes)
        return {label: probs[0, i].item() for i, label in enumerate(self.LABEL_NAMES)}


# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────

# Registry maps model_type → (model_class, config_class, new_model_func)
# Populated lazily to avoid circular imports at module load time.
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def _ensure_registry() -> None:
    """Populate the model registry if not yet populated."""
    if _MODEL_REGISTRY:
        return

    from sentimentizer.config import DecoderConfig, EncoderConfig, RNNConfig
    from sentimentizer.models.decoder import Decoder
    from sentimentizer.models.decoder import new_model as _decoder_new
    from sentimentizer.models.encoder import Encoder
    from sentimentizer.models.encoder import new_model as _encoder_new
    from sentimentizer.models.rnn import RNN
    from sentimentizer.models.rnn import new_model as _rnn_new

    _MODEL_REGISTRY["rnn"] = {
        "model_class": RNN,
        "config_class": RNNConfig,
        "new_model": _rnn_new,
        "weights_key": "rnn",
    }
    _MODEL_REGISTRY["encoder"] = {
        "model_class": Encoder,
        "config_class": EncoderConfig,
        "new_model": _encoder_new,
        "weights_key": "encoder",
    }
    _MODEL_REGISTRY["decoder"] = {
        "model_class": Decoder,
        "config_class": DecoderConfig,
        "new_model": _decoder_new,
        "weights_key": "decoder",
    }


def get_model_registry() -> dict[str, dict[str, Any]]:
    """Return the model registry, populating it lazily if needed."""
    _ensure_registry()
    return dict(_MODEL_REGISTRY)


def get_trained_model(model_type: str, device: str) -> nn.Module:
    """Load a pre-trained model from saved weights using the model registry.

    This unified function replaces the separate get_trained_model() functions
    that were duplicated across rnn.py, encoder.py, and decoder.py.

    If local weights are not found, attempts to download from Hugging Face Hub.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        device: Device to load weights onto ("cpu", "cuda", or "mps").

    Returns:
        Model with loaded weights on the specified device.

    Raises:
        ValueError: If device is invalid or model_type is unknown.
        FileNotFoundError: If weights cannot be found locally or downloaded.
        RuntimeError: If weights are incompatible with the current architecture.
    """
    _ensure_registry()

    if device not in VALID_DEVICES:
        raise ValueError("device must be cpu, cuda, or mps")

    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type!r}")

    entry = _MODEL_REGISTRY[model_type]
    model_class = entry["model_class"]

    from sentimentizer.config import DriverConfig, weights_path_for

    weights_path = weights_path_for(model_type)

    # Try local file first; if missing, download from Hugging Face Hub
    if not Path(weights_path).exists():
        from sentimentizer.hf import download_weights

        downloaded = download_weights(
            model_type, weights_path, dict_path=DriverConfig.files.dictionary_file_path
        )
        if downloaded is None:
            raise FileNotFoundError(
                f"Weights file not found at {weights_path} and could not be "
                "downloaded from Hugging Face Hub. Please train the model "
                f"first: python workflows/driver.py --device cuda --model {model_type} "
                "--type new --save True"
            )

    weights = torch.load(weights_path, map_location=torch.device(device=device), weights_only=True)

    # Check if weights are from the new architecture (has 'classifier' keys)
    if "classifier.0.weight" not in weights:
        raise RuntimeError(
            "Saved weights are from a previous model architecture and are incompatible. "
            "Please retrain the model: "
            f"python workflows/driver.py --device cuda --model {model_type} --type new --save True"
        )

    # Use the model-specific new_model function to create an empty model
    # with correct architecture, then load the trained weights
    config_class = entry["config_class"]
    config = config_class()

    # Each model has different architecture inference logic from weights,
    # so delegate to the model-specific new_model and then load_state_dict

    emb_shape = weights["embed_layer.weight"].shape
    empty_embeddings = torch.zeros(emb_shape)

    # Build model kwargs based on model type
    # Infer num_classes from the final linear layer output dimension
    num_classes = weights["classifier.3.weight"].shape[0]
    if num_classes == 1:
        raise RuntimeError(
            f"Saved weights are from a binary classification model (num_classes=1). "
            f"3-class migration requires retraining: "
            f"python workflows/driver.py --device cuda --model {model_type} --type new --save True"
        )
    # Also check _metadata if available for robust detection
    if "_metadata" in weights and "num_classes" in weights["_metadata"]:
        num_classes = weights["_metadata"]["num_classes"]

    if model_type == "rnn":
        hidden_size = weights["classifier.0.weight"].shape[1] // 2
        model = model_class(
            emb_weights=empty_embeddings,
            hidden_size=hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=num_classes,
        )
    elif model_type == "encoder":
        d_model = weights["proj.weight"].shape[0]
        model = model_class(
            d_model=d_model,
            n_heads=config.n_heads,
            emb_weights=empty_embeddings,
            n_layers=config.n_layers,
            dropout=config.dropout,
            ff_multiplier=config.ff_multiplier,
            num_classes=num_classes,
        )
    elif model_type == "decoder":
        d_model = weights["proj.weight"].shape[0]
        model = model_class(
            d_model=d_model,
            n_heads=config.n_heads,
            emb_weights=empty_embeddings,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            dropout=config.dropout,
            ff_multiplier=config.ff_multiplier,
            num_classes=num_classes,
        )

    try:
        model.load_state_dict(weights)
    except RuntimeError as e:
        raise RuntimeError(
            "Saved weights are incompatible with the current model architecture. "
            "Please retrain the model: "
            f"python workflows/driver.py --device cuda --model {model_type} --type new --save True"
        ) from e

    return model
