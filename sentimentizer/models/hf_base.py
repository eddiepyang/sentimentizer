"""Base class for Hugging Face transformer models in sentimentizer."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn

from sentimentizer import new_logger
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    ModernBERTOptimizationParams,
    ModernBERTSchedulerParams,
)
from sentimentizer.models.base import BaseSentimentModel

logger = new_logger(DEFAULT_LOG_LEVEL)

try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HFTransformerModel(BaseSentimentModel):
    """Base class for Hugging Face transformer models.

    Manages transformer backbone loading, mean-pooling over non-padding tokens,
    layer-specific training parameter freeze/unfreeze, and directory-based
    checkpointing (saving backbone configs and weights separately from classifier
    head to allow dynamic/offline serving).
    """

    OPT_PARAMS_CLS: ClassVar[type] = ModernBERTOptimizationParams
    SCHED_PARAMS_CLS: ClassVar[type] = ModernBERTSchedulerParams
    NEEDS_TOKENIZE_STAGE: ClassVar[bool] = False
    DDP_FIND_UNUSED_PARAMS: ClassVar[bool] = True
    SUPPORTS_ONNX: ClassVar[bool] = False
    HF_MODEL_NAME: ClassVar[str] = ""

    def __init__(self, config: Any, tokenizer: Any = None) -> None:
        super().__init__(tokenizer=tokenizer)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Hugging Face transformers package is required to use HFTransformerModel. "
                "Please install it using: pip install sentimentizer[transformers]"
            )
        self.config = config

        # Load backbone: use local directory if it exists, else load from HF hub
        # attn_implementation="eager" ensures deterministic behavior and avoids
        # silently using Flash Attention when flash-attn is installed.
        backbone_name_or_path = config.model_name
        self.backbone = AutoModel.from_pretrained(
            backbone_name_or_path, attn_implementation="eager"
        )

        # Enable gradient checkpointing to reduce activation memory during
        # training.  This trades ~30% more compute for ~60% less activation
        # VRAM, which is critical for 149M-param models on 16 GiB GPUs.
        # HuggingFace's implementation recomputes intermediate activations
        # during the backward pass instead of caching them.
        if getattr(config, "gradient_checkpointing", True):
            self.backbone.gradient_checkpointing_enable()

        # Define classifier head
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden = outputs.last_hidden_state  # (B, L, D)

        if attention_mask is not None:
            # Expand attention mask to shape (B, L, D)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            # Sum embeddings along seq_len dimension, weighted by attention mask
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            # Sum attention mask along seq_len dimension
            # (clamped to min=1e-9 to avoid division by zero)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = last_hidden.mean(dim=1)

        return self.classifier(pooled)

    def prepare_batch(self, batch: dict, device: str) -> tuple[dict, torch.Tensor]:
        """Convert a raw batch dict into (model_inputs, target)."""
        target = batch["target"].long().to(device)

        inputs = {}
        if "input_ids" in batch:
            inputs["input_ids"] = batch["input_ids"].long().to(device)
        elif "data" in batch:  # Ray dataset might map text/ids to 'data'
            inputs["input_ids"] = batch["data"].long().to(device)

        if "attention_mask" in batch:
            inputs["attention_mask"] = batch["attention_mask"].long().to(device)

        return inputs, target

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone parameters.

        Handles DDP unwrap internally so callers don't need to know
        if the model is DDP-wrapped.
        """
        inner = self.module if hasattr(self, "module") else self
        for param in inner.backbone.parameters():
            param.requires_grad = True

    def save_to_checkpoint_dir(self, ckpt_dir: Path, tokenizer: Any = None) -> dict:
        """Save model state to checkpoint directory.

        Saves the transformer backbone and tokenizer into a backbone/ subdirectory,
        and returns the classifier head's state dict and configuration metadata.

        Config is serialized as a plain dict (via ``dataclasses.asdict``) so the
        resulting ``.pth`` file contains only tensors and primitives and can be
        loaded with ``weights_only=True``.
        """
        backbone_dir = ckpt_dir / "backbone"
        backbone_dir.mkdir(parents=True, exist_ok=True)

        # Save backbone using Hugging Face's save_pretrained
        self.backbone.save_pretrained(str(backbone_dir))

        # Save tokenizer
        active_tokenizer = tokenizer or self.tokenizer
        if active_tokenizer is not None:
            active_tokenizer.save_pretrained(str(backbone_dir))

        return {
            "classifier_state_dict": self.classifier.state_dict(),
            # Serialize config as a plain dict so weights_only=True is preserved.
            # load_from_checkpoint_dir reconstructs the dataclass from this dict.
            "config_dict": dataclasses.asdict(self.config),
            "config_class": type(self.config).__name__,
            "num_classes": self.config.num_classes,
            "dropout": self.config.dropout,
            "backbone_dir": "backbone",  # relative path for checkpoint portability
            "hf_model_name": self.HF_MODEL_NAME,
        }

    @classmethod
    def load_from_checkpoint_dir(
        cls, ckpt_dir: Path, metadata: dict, device: str
    ) -> HFTransformerModel:
        """Construct an instance and restore weights from checkpoint directory."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Hugging Face transformers package is required to load HFTransformerModel. "
                "Please install it using: pip install sentimentizer[transformers]"
            )

        # Validate checkpoint format — pre-refactor checkpoints won't have this key
        if "backbone_dir" not in metadata:
            raise ValueError(
                "Checkpoint missing 'backbone_dir' — saved with pre-refactor format. "
                "Retrain the model or use a newer checkpoint."
            )

        backbone_dir = ckpt_dir / metadata["backbone_dir"]

        # Issue #3: explicit error if backbone directory is missing
        if not backbone_dir.exists():
            logger.error(
                f"backbone directory not found in checkpoint: {backbone_dir} — "
                f"checkpoint may be corrupt or incomplete"
            )
            raise FileNotFoundError(
                f"Backbone directory not found at {backbone_dir}. "
                f"The checkpoint may be corrupt or saved with an older format."
            )

        # Load backbone from local checkpoint directory with eager attention
        # (same as __init__ — avoids silent Flash Attention usage)
        backbone = AutoModel.from_pretrained(str(backbone_dir), attn_implementation="eager")

        # Reconstruct config from the serialized plain dict.
        # New checkpoints store "config_dict" + "config_class" (plain primitives, weights_only=True
        # compatible). Legacy checkpoints stored the dataclass object directly under "config".
        if "config_dict" in metadata:
            # New format: reconstruct dataclass from plain dict via registry
            from sentimentizer.models.base import _MODEL_REGISTRY, _ensure_registry

            _ensure_registry()
            # Find the config class by name from the registry
            config_class_name = metadata.get("config_class", "ModernBERTConfig")
            config_class = None
            for entry in _MODEL_REGISTRY.values():
                if entry["config_class"].__name__ == config_class_name:
                    config_class = entry["config_class"]
                    break
            if config_class is None:
                from sentimentizer.config import ModernBERTConfig

                config_class = ModernBERTConfig
            config_dict = dict(metadata["config_dict"])
            config_dict["model_name"] = str(backbone_dir)
            config = config_class(**config_dict)
        else:
            # Legacy format: config was stored as a dataclass object directly.
            # weights_only=False was required for this path.
            legacy_config = metadata["config"]
            from sentimentizer.config import ModernBERTConfig

            config = ModernBERTConfig(
                model_name=str(backbone_dir),
                dropout=metadata.get(
                    "dropout", legacy_config.dropout if hasattr(legacy_config, "dropout") else 0.1
                ),
                num_classes=metadata.get(
                    "num_classes",
                    legacy_config.num_classes if hasattr(legacy_config, "num_classes") else 3,
                ),
                max_seq_length=(
                    legacy_config.max_seq_length
                    if hasattr(legacy_config, "max_seq_length")
                    else 512
                ),
                freeze_backbone_epochs=(
                    legacy_config.freeze_backbone_epochs
                    if hasattr(legacy_config, "freeze_backbone_epochs")
                    else 2
                ),
            )

        # Create model instance with loaded backbone (bypasses __init__'s from_pretrained)
        model = cls.__new__(cls)
        super(HFTransformerModel, model).__init__(tokenizer=None)
        model.config = config
        model.backbone = backbone

        # Re-enable gradient checkpointing for loaded models (same as __init__)
        if getattr(config, "gradient_checkpointing", True):
            model.backbone.gradient_checkpointing_enable()

        hidden_size = backbone.config.hidden_size
        model.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes),
        )

        # Load tokenizer from checkpoint if available
        tokenizer = None
        if (backbone_dir / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(backbone_dir))
        model.tokenizer = tokenizer

        # Load classifier weights from metadata
        if "classifier_state_dict" in metadata:
            model.classifier.load_state_dict(metadata["classifier_state_dict"])
        elif "model_state_dict" in metadata:
            state_dict = metadata["model_state_dict"]
            classifier_keys = {k: v for k, v in state_dict.items() if k.startswith("classifier.")}
            if classifier_keys:
                stripped_keys = {k[len("classifier.") :]: v for k, v in classifier_keys.items()}
                model.classifier.load_state_dict(stripped_keys)
            else:
                model.load_state_dict(state_dict, strict=False)

        return model.to(device)
