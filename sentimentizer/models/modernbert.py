"""ModernBERT sentiment classification model and factory."""

from __future__ import annotations

from typing import Any, ClassVar

from sentimentizer.models.hf_base import TRANSFORMERS_AVAILABLE, HFTransformerModel


class ModernBERT(HFTransformerModel):
    """ModernBERT model for 3-class sentiment classification."""

    MODEL_TYPE: str = "modernbert"
    HF_MODEL_NAME: ClassVar[str] = "answerdotai/ModernBERT-base"

    def __init__(self, config: Any = None, tokenizer: Any = None) -> None:
        if config is None:
            from sentimentizer.config import ModernBERTConfig

            config = ModernBERTConfig()
        super().__init__(config=config, tokenizer=tokenizer)


def new_modernbert_model(
    dict_path: Any = None,
    embeddings_config: Any = None,
    freeze_embeddings: bool = True,
) -> ModernBERT:
    """Factory function to create a new ModernBERT model.

    Args:
        dict_path: Ignored. Included for signature compatibility.
        embeddings_config: Ignored. Included for signature compatibility.
        freeze_embeddings: If True and freeze_backbone_epochs > 0, the backbone
            is frozen initially and unfrozen at the configured epoch. If
            freeze_backbone_epochs == 0 (default), the backbone is trainable
            from the start regardless of this flag.

    Returns:
        A ModernBERT model instance.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Hugging Face transformers package is required to use ModernBERT. "
            "Please install it using: pip install sentimentizer[transformers]"
        )

    from transformers import AutoTokenizer

    from sentimentizer.config import ModernBERTConfig

    config = ModernBERTConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    model = ModernBERT(config=config, tokenizer=tokenizer)

    # Only freeze the backbone if freeze_backbone_epochs > 0, meaning the
    # user wants a warmup phase with only the classifier head trainable.
    # When freeze_backbone_epochs == 0 (default), the entire model is
    # trainable from the start — freezing and never unfreezing would
    # prevent the backbone from ever learning.
    if freeze_embeddings and config.freeze_backbone_epochs > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
