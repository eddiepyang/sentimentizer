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
        freeze_embeddings: If True, initial training steps freeze the transformer backbone.

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

    if freeze_embeddings:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model
