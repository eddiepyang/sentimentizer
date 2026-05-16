"""Train the SetFit router model for Yelp review categorization.

Uses CosineSimilarityLoss for contrastive learning on the embedding
space, then fits a classification head on top of the embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from setfit import SetFitModel, Trainer, TrainingArguments

# Apply transformers compatibility shim BEFORE importing setfit,
# which imports default_logdir from transformers.training_args
# (removed in transformers 5.x).
import sentimentizer.compat  # noqa: F401
from sentimentizer.router.config import RouteLabels, SetFitConfig

logger = logging.getLogger(__name__)


def _load_setfit_model(model_id: str, num_classes: int = 3) -> SetFitModel:
    """Load a SetFit model, handling missing config_setfit.json gracefully.

    Sentence-transformer models like BAAI/bge-base-en-v1.5 don't have
    a config_setfit.json on HuggingFace Hub. When using huggingface_hub>=1.0,
    the 404 is raised as a hard error before SetFit can fall back. This
    function catches that error and creates the model from the
    sentence-transformer backbone directly, with a classification head
    for the specified number of classes.
    """
    try:
        return SetFitModel.from_pretrained(model_id)
    except Exception as e:
        error_str = str(e)
        if "404" in error_str and "config_setfit" in error_str:
            logger.info(
                f"No SetFit config found for {model_id}, "
                f"loading as sentence-transformer backbone "
                f"with {num_classes}-class classification head"
            )
            from sentence_transformers import SentenceTransformer
            from sklearn.linear_model import LogisticRegression

            model_body = SentenceTransformer(model_id)
            model = SetFitModel(model_body=model_body)
            # SetFitModel(model_body=...) does not auto-create a
            # classification head. We must create a LogisticRegression
            # head explicitly so that Trainer.train_classifier() can
            # call model_head.fit(embeddings, labels).
            model.model_head = LogisticRegression(max_iter=1000, solver="lbfgs")
            model.labels = list(RouteLabels.label_names().values())[:num_classes]
            logger.info(
                f"Created {num_classes}-class LogisticRegression head "
                f"for sentence-transformer backbone"
            )
            return model
        raise


def train_router(
    config: SetFitConfig,
    train_dataset: Any,
    eval_dataset: Any = None,
) -> SetFitModel:
    """Train the SetFit router model.

    Uses CosineSimilarityLoss for contrastive learning on the embedding
    space, then fits a classification head on top of the embeddings.

    Args:
        config: SetFitConfig with model parameters.
        train_dataset: Training dataset with 'text' and 'label' columns.
        eval_dataset: Optional evaluation dataset.

    Returns:
        Trained SetFitModel.
    """
    logger.info(f"Loading SetFit base model: {config.base_model}")
    model = _load_setfit_model(config.base_model, num_classes=RouteLabels.num_classes())

    args = TrainingArguments(
        batch_size=config.batch_size,
        num_iterations=config.num_iterations,
        num_epochs=config.num_epochs,
        seed=config.seed,
        output_dir=str(config.output_dir),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        column_mapping={"text": "text", "label": "label"},
    )

    logger.info("Starting SetFit router training...")
    trainer.train()

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    logger.info(f"Router model saved to {output_path}")
    return model
