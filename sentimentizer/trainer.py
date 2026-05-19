from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

try:
    import ray
    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.torch import TorchTrainer, prepare_model
except ImportError:
    pass

from sentimentizer import new_logger
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    DecoderOptimizationParams,
    DecoderSchedulerParams,
    DriverConfig,
    EncoderOptimizationParams,
    EncoderSchedulerParams,
    OptimizationParams,
    SchedulerParams,
    TrainerConfig,
    default_dataloader_workers,
    default_epochs,
)
from sentimentizer.loader import CorpusDataset
from sentimentizer.metrics import ClassificationMetrics
from sentimentizer.metrics_publisher import publish_epoch_metrics, write_epoch_metrics_to_file

# Backward-compatible alias — existing code imports _write_epoch_metrics_to_file
# from this module.  The canonical location is now metrics_publisher.
_write_epoch_metrics_to_file = write_epoch_metrics_to_file


# ---------------------------------------------------------------------------
# Ray custom metrics (lazy initialization)
#
# In Ray 2.55+, ray.util.metrics.Gauge objects must be created inside a
# Ray worker context (task or actor) to be exported via the Ray metrics
# endpoint.  Creating them at module-import time in the driver process
# silently fails — the gauges are never pushed to the metrics agent.
#
# We therefore lazily create gauge instances on first use and cache them
# per model_type so that each worker gets its own gauge that is properly
# registered with that worker's metrics agent.
# ---------------------------------------------------------------------------

_RAY_GAUGES: dict[str, dict[str, Any]] = {}
"""Cache of Ray Gauge dicts keyed by model_type."""


def _get_ray_gauges(model_type: str) -> dict[str, Any] | None:
    """Return (and lazily create) Ray Gauge instances for *model_type*.

    Gauges are created once per model_type and cached.  On the first call
    this imports ``ray.util.metrics.Gauge`` and builds all the gauge
    objects.  If Ray is not available, returns ``None``.
    """
    if model_type in _RAY_GAUGES:
        return _RAY_GAUGES[model_type]

    try:
        from ray.util.metrics import Gauge
    except ImportError:
        return None

    gauges: dict[str, Any] = {
        "train_loss": Gauge(
            "sentimentizer_live_train_loss",
            description="Live training loss",
            tag_keys=("model_type",),
        ),
        "val_loss": Gauge(
            "sentimentizer_live_val_loss",
            description="Live validation loss",
            tag_keys=("model_type",),
        ),
        "val_accuracy": Gauge(
            "sentimentizer_live_val_accuracy",
            description="Live validation accuracy",
            tag_keys=("model_type",),
        ),
        "val_balanced_accuracy": Gauge(
            "sentimentizer_live_val_balanced_accuracy",
            description="Live validation balanced accuracy",
            tag_keys=("model_type",),
        ),
        "val_negative_precision": Gauge(
            "sentimentizer_live_val_negative_precision",
            description="Live validation negative-class precision",
            tag_keys=("model_type",),
        ),
        "val_negative_recall": Gauge(
            "sentimentizer_live_val_negative_recall",
            description="Live validation negative-class recall",
            tag_keys=("model_type",),
        ),
        "val_negative_f1": Gauge(
            "sentimentizer_live_val_negative_f1",
            description="Live validation negative-class F1",
            tag_keys=("model_type",),
        ),
        "val_neutral_precision": Gauge(
            "sentimentizer_live_val_neutral_precision",
            description="Live validation neutral-class precision",
            tag_keys=("model_type",),
        ),
        "val_neutral_recall": Gauge(
            "sentimentizer_live_val_neutral_recall",
            description="Live validation neutral-class recall",
            tag_keys=("model_type",),
        ),
        "val_neutral_f1": Gauge(
            "sentimentizer_live_val_neutral_f1",
            description="Live validation neutral-class F1",
            tag_keys=("model_type",),
        ),
        "val_positive_precision": Gauge(
            "sentimentizer_live_val_positive_precision",
            description="Live validation positive-class precision",
            tag_keys=("model_type",),
        ),
        "val_positive_recall": Gauge(
            "sentimentizer_live_val_positive_recall",
            description="Live validation positive-class recall",
            tag_keys=("model_type",),
        ),
        "val_positive_f1": Gauge(
            "sentimentizer_live_val_positive_f1",
            description="Live validation positive-class F1",
            tag_keys=("model_type",),
        ),
        "val_cohen_kappa": Gauge(
            "sentimentizer_live_val_cohen_kappa",
            description="Live validation Cohen's kappa",
            tag_keys=("model_type",),
        ),
        "val_mcc": Gauge(
            "sentimentizer_live_val_mcc",
            description="Live validation Matthews correlation coefficient",
            tag_keys=("model_type",),
        ),
        "val_macro_f1": Gauge(
            "sentimentizer_live_val_macro_f1",
            description="Live validation macro-averaged F1 score",
            tag_keys=("model_type",),
        ),
        "val_weighted_f1": Gauge(
            "sentimentizer_live_val_weighted_f1",
            description="Live validation weighted-averaged F1 score",
            tag_keys=("model_type",),
        ),
        "val_neutral_to_positive_rate": Gauge(
            "sentimentizer_live_val_neutral_to_positive_rate",
            description="Live validation neutral-to-positive misclassification rate",
            tag_keys=("model_type",),
        ),
        "val_neutral_to_negative_rate": Gauge(
            "sentimentizer_live_val_neutral_to_negative_rate",
            description="Live validation neutral-to-negative misclassification rate",
            tag_keys=("model_type",),
        ),
        "val_pred_neutral_frac": Gauge(
            "sentimentizer_live_val_pred_neutral_frac",
            description="Live validation fraction of neutral predictions",
            tag_keys=("model_type",),
        ),
        "val_neutral_auc_roc": Gauge(
            "sentimentizer_live_val_neutral_auc_roc",
            description="Live validation neutral-class AUC-ROC",
            tag_keys=("model_type",),
        ),
        "val_neutral_avg_precision": Gauge(
            "sentimentizer_live_val_neutral_avg_precision",
            description="Live validation neutral-class average precision",
            tag_keys=("model_type",),
        ),
        "epoch": Gauge(
            "sentimentizer_live_epoch",
            description="Live training epoch",
            tag_keys=("model_type",),
        ),
        "lr": Gauge(
            "sentimentizer_live_lr",
            description="Live learning rate",
            tag_keys=("model_type",),
        ),
    }
    # Set default tag so callers can do gauge.set(value) without
    # repeating the tag on every call.
    for g in gauges.values():
        g.set_default_tags({"model_type": model_type})

    _RAY_GAUGES[model_type] = gauges
    return gauges


logger = new_logger(DEFAULT_LOG_LEVEL)


# ──────────────────────────────────────────────
# Shared training primitives
# ──────────────────────────────────────────────


def train_step(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    max_grad_norm: float = 1.0,
) -> float:
    """Single training step: forward, loss, backward, clip, step.

    Args:
        model: The model to train.
        data: Input tensor (already on the correct device).
        target: Target tensor (already on the correct device).
        optimizer: Optimizer to update parameters.
        loss_function: Loss function to compute loss.
        max_grad_norm: Max gradient norm for clipping.

    Returns:
        The scalar loss value for this step.
    """
    optimizer.zero_grad()
    output = model(data)
    loss = loss_function(output, target)
    loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm, norm_type=2
    )
    optimizer.step()
    return loss.item()


def val_step(
    model: torch.nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    loss_function: Callable,
) -> float:
    """Single validation step: forward, loss (no grad).

    Args:
        model: The model to evaluate.
        data: Input tensor (already on the correct device).
        target: Target tensor (already on the correct device).
        loss_function: Loss function to compute loss.

    Returns:
        The scalar loss value for this step.
    """
    with torch.no_grad():
        output = model(data)
        loss = loss_function(output, target)
    return loss.item()


@dataclass
class EpochResult:
    """Per-epoch training result container."""

    epoch: int
    train_loss: float
    val_loss: float
    metrics: ClassificationMetrics
    lr: float


@dataclass
class TrainingState:
    """Mutable training state container."""

    val_loss: float = float("inf")
    latest_train_loss: float = 0.0
    latest_epoch: int = 0
    latest_metrics: ClassificationMetrics | None = None
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    running_loss_mean: float = 0.0
    steps: int = 0


class TrainingCallback:
    """Protocol for hooking into the training loop."""

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        """Called after each epoch. Return True to stop training.

        Args:
            result: The current epoch's results.
            state: Mutable training state.

        Returns:
            True if training should stop, False otherwise.
        """
        return False

    def on_train_begin(self, model_type: str, device: str, epochs: int) -> None:
        """Called before training starts."""

    def on_train_end(self, state: TrainingState) -> None:
        """Called after training ends."""


class MetricsCallback(TrainingCallback):
    """Publishes metrics to Prometheus/JSON. Only active on rank 0."""

    def __init__(
        self,
        model_type: str,
        rank: int = 0,
        ray_gauges: dict[str, Any] | None = None,
    ) -> None:
        self.model_type = model_type
        self.rank = rank
        self.ray_gauges = ray_gauges

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if self.rank != 0:
            return False
        publish_epoch_metrics(
            model_type=self.model_type,
            epoch=result.epoch,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
            metrics=result.metrics,
            lr=result.lr,
            ray_gauges=self.ray_gauges,
        )
        return False


class CheckpointCallback(TrainingCallback):
    """Saves periodic and best model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        checkpoint_every: int,
        checkpoint_best: bool,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.checkpoint_best = checkpoint_best
        self._best_val_loss = float("inf")

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if self.checkpoint_dir is None:
            return False

        # Periodic checkpoint
        if self.checkpoint_every > 0 and result.epoch % self.checkpoint_every == 0:
            ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{result.epoch}.pth"
            # Note: actual save happens in _run_training_loop which has model/optimizer refs
            state._pending_checkpoint_path = str(ckpt_path)  # type: ignore[attr-defined]

        # Best model checkpoint
        if self.checkpoint_best and result.val_loss < self._best_val_loss:
            self._best_val_loss = result.val_loss
            best_path = self.checkpoint_dir / "best_model.pth"
            state._pending_best_checkpoint_path = str(best_path)  # type: ignore[attr-defined]

        return False


class EarlyStoppingCallback(TrainingCallback):
    """Stops training when validation loss doesn't improve."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._best_val_loss = float("inf")
        self.patience_counter = 0

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if result.val_loss < self._best_val_loss:
            self._best_val_loss = result.val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(
                    f"early stopping at epoch {result.epoch}, "
                    f"val_loss hasn't improved for {self.patience_counter} epochs"
                )
                return True
        return False


class LoggingCallback(TrainingCallback):
    """Structured logging per epoch. Only active on rank 0."""

    def __init__(self, model_type: str, rank: int = 0) -> None:
        self.model_type = model_type
        self.rank = rank

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        if self.rank != 0:
            return False
        logger.info(
            f"[{self.model_type}] [epoch {result.epoch}] completed, "
            f"val_loss={result.val_loss:.4f}, train_loss={result.train_loss:.4f}, "
            f"accuracy={result.metrics.accuracy:.4f}, lr={result.lr:.6f}"
        )
        return False

    def on_train_begin(self, model_type: str, device: str, epochs: int) -> None:
        if self.rank != 0:
            return
        logger.info(f"[{model_type}] fitting model...")
        logger.info(f"[{model_type}] epochs={epochs}, device={device}")


class RayReportCallback(TrainingCallback):
    """Reports metrics and checkpoints to Ray Train. All ranks must call."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
        try:
            import tempfile

            import ray.cloudpickle as pickle
            from ray import train
            from ray.train import Checkpoint

            # Note: checkpoint data must be populated by the loop
            checkpoint_data = getattr(state, "_ray_checkpoint_data", None)
            if checkpoint_data is not None:
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, "data.pkl"), "wb") as fp:
                        pickle.dump(checkpoint_data, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(
                        {
                            "train_loss": result.train_loss,
                            "val_loss": result.val_loss,
                            "accuracy": result.metrics.accuracy,
                            "balanced_accuracy": result.metrics.balanced_accuracy,
                            "negative_precision": result.metrics.negative_precision,
                            "negative_recall": result.metrics.negative_recall,
                            "negative_f1": result.metrics.negative_f1,
                            "neutral_precision": result.metrics.neutral_precision,
                            "neutral_recall": result.metrics.neutral_recall,
                            "neutral_f1": result.metrics.neutral_f1,
                            "positive_precision": result.metrics.positive_precision,
                            "positive_recall": result.metrics.positive_recall,
                            "positive_f1": result.metrics.positive_f1,
                            "macro_f1": result.metrics.macro_f1,
                            "weighted_f1": result.metrics.weighted_f1,
                            "cohen_kappa": result.metrics.cohen_kappa,
                            "mcc": result.metrics.mcc,
                            "total": result.metrics.total,
                            "epoch": result.epoch,
                            "lr": result.lr,
                        },
                        checkpoint=checkpoint,
                    )
            else:
                train.report(
                    {
                        "train_loss": result.train_loss,
                        "val_loss": result.val_loss,
                        "accuracy": result.metrics.accuracy,
                        "balanced_accuracy": result.metrics.balanced_accuracy,
                        "macro_f1": result.metrics.macro_f1,
                        "epoch": result.epoch,
                        "lr": result.lr,
                    }
                )
        except (ImportError, RuntimeError):
            pass
        return False


def compute_epoch_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    model_type: str,
    num_classes: int = 3,
) -> ClassificationMetrics:
    """Compute classification metrics from probabilities and targets.

    Handles NaN replacement in probabilities before computing metrics.
    This is the shared post-processing logic that was previously duplicated
    in both Trainer.evaluate() and _train_func().

    Args:
        probabilities: Softmax probability matrix, shape (N, num_classes).
            May contain NaN values.
        targets: Ground truth class indices, shape (N,).
        model_type: Model type label for logging.
        num_classes: Number of classification classes (default 3).

    Returns:
        ClassificationMetrics with all standard classification metrics.
    """
    from sentimentizer.config import NUM_CLASSES as _NUM_CLASSES

    if num_classes <= 1:
        num_classes = _NUM_CLASSES

    # Replace NaN probabilities with uniform distribution
    nan_mask = np.isnan(probabilities)
    if nan_mask.any():
        nan_count = int(nan_mask.sum())
        logger.warning(
            f"[{model_type}] nan_in_probabilities",
            message=f"Found {nan_count} NaN values in probabilities, replacing with uniform",
        )
        probabilities = np.where(nan_mask, 1.0 / num_classes, probabilities)

    predictions = probabilities.argmax(axis=1)

    from sentimentizer.metrics import compute_classification_metrics

    return compute_classification_metrics(
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
        num_classes=num_classes,
    )


def _iter_batches(
    data_source: DataLoader | Any,
    batch_size: int,
    device: str,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (data, target) tensors regardless of source type.

    Normalizes iteration over both PyTorch DataLoader and Ray dataset shards,
    ensuring tensors are moved to the correct device with appropriate dtypes.

    Args:
        data_source: Either a DataLoader or a Ray dataset shard.
        batch_size: Batch size for Ray dataset iteration.
        device: Device to move tensors to.

    Yields:
        Tuples of (data, target) tensors.
    """
    if isinstance(data_source, DataLoader):
        for sent, target in data_source:
            yield sent.to(device), target.to(device)
    else:
        # Ray dataset shard
        for batch in data_source.iter_torch_batches(batch_size=batch_size):
            yield batch["data"].long().to(device), batch["target"].long().to(device)


def create_model_from_registry(
    model_type: str,
    dict_path: str,
    embeddings_config: Any,
    model_config: Any | None = None,
) -> torch.nn.Module:
    """Create a model using the MODEL_REGISTRY.

    Replaces inline if/elif blocks for model creation in distributed
    and tuning paths with a unified registry lookup.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        dict_path: Path to the dictionary file.
        embeddings_config: EmbeddingsConfig instance.
        model_config: Optional model-specific config (for tuning).

    Returns:
        The created model instance.

    Raises:
        ValueError: If model_type is not in the registry.
    """
    from sentimentizer.models.base import get_model_registry

    registry = get_model_registry()
    if model_type not in registry:
        raise ValueError(f"no matching model for {model_type}")

    entry = registry[model_type]
    new_model_func = entry["new_model"]

    kwargs: dict[str, Any] = {
        "dict_path": dict_path,
        "embeddings_config": embeddings_config,
    }
    if model_config is not None:
        kwargs["model_config"] = model_config

    return new_model_func(**kwargs)


def _create_training_components(
    model: torch.nn.Module,
    model_type: str,
    device: str,
    cfg: TrainerConfig | None = None,
    lr: float | None = None,
    betas: tuple[float, float] | None = None,
    weight_decay: float | None = None,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, Callable]:
    """Create optimizer, scheduler, and loss function for a model.

    Uses CrossEntropyLoss (or FocalCrossEntropyLoss) for 3-class
    classification, replacing the binary BCEWithLogitsLoss.

    Args:
        model: The model to create components for.
        model_type: Model type string.
        device: Device to place tensors on.
        cfg: TrainerConfig for loss_type/label_smoothing/focal_gamma settings.
        lr: Optional learning rate override.
        betas: Optional AdamW betas override.
        weight_decay: Optional weight decay override.
        class_weights: Optional per-class weight tensor for CrossEntropyLoss.

    Returns:
        Tuple of (optimizer, scheduler, loss_function).
    """
    from sentimentizer.losses import FocalCrossEntropyLoss

    opt_params = _get_opt_params(model_type)
    sched_params = _get_sched_params(model_type)

    # Use overrides if provided, otherwise use defaults
    lr = lr if lr is not None else opt_params.lr
    betas = betas if betas is not None else opt_params.betas
    weight_decay = weight_decay if weight_decay is not None else opt_params.weight_decay

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    # Use warmup+cosine for transformer models, simple cosine for RNN
    is_transformer = isinstance(sched_params, (EncoderSchedulerParams, DecoderSchedulerParams))
    if is_transformer and sched_params.warmup_epochs > 0:
        scheduler = _LinearWarmupCosineScheduler(
            optimizer,
            warmup_steps=sched_params.warmup_epochs,
            total_steps=sched_params.T_max,
            eta_min=sched_params.eta_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_params.T_max,
            eta_min=sched_params.eta_min,
            last_epoch=sched_params.last_epoch,
        )

    # Determine loss type and label smoothing from config
    loss_type = cfg.loss_type if cfg else "cross_entropy"
    label_smoothing = cfg.label_smoothing if cfg else 0.1
    focal_gamma = cfg.focal_gamma if cfg else 2.0

    # Move class weights to device if provided
    weights_on_device = class_weights.to(device) if class_weights is not None else None

    if loss_type == "focal":
        loss_function = FocalCrossEntropyLoss(
            weight=weights_on_device,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    else:
        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights_on_device,
            label_smoothing=label_smoothing,
        )

    return optimizer, scheduler, loss_function


def _new_loaders(
    train_data: CorpusDataset, val_data: CorpusDataset, cfg: TrainerConfig
) -> tuple[DataLoader, DataLoader]:
    # Resolve auto-detect dataloader_workers (-1 means auto)
    workers = cfg.dataloader_workers
    if workers == -1:
        workers = default_dataloader_workers(cfg.device)

    pin_mem = cfg.memory if cfg.device != "mps" else False

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=pin_mem,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader


@dataclass
class Trainer:
    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""

    loss_function: Callable
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    cfg: TrainerConfig
    model_type: str
    val_loss: float = float("inf")
    latest_train_loss: float = 0.0
    latest_epoch: int = 0
    latest_metrics: ClassificationMetrics | None = None

    def _train_epoch(self, model: torch.nn.Module, train_loader: DataLoader, epoch: int) -> None:
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        loss_ema = 0.0
        update_every = 50  # update tqdm postfix every N batches

        with logging_redirect_tqdm():
            pbar = tqdm(train_loader, desc=f"[{self.model_type}] epoch {epoch}", leave=False)
            for i, (sent, target) in enumerate(pbar):
                batch_size = target.size(0)
                loss_val = train_step(
                    model,
                    data=sent.to(self.cfg.device),
                    target=target.to(self.cfg.device),
                    optimizer=self.optimizer,
                    loss_function=self.loss_function,
                )
                epoch_loss_sum += loss_val * batch_size
                epoch_sample_count += batch_size
                loss_ema = 0.9 * loss_ema + 0.1 * loss_val if i > 0 else loss_val
                if i % update_every == 0:
                    pbar.set_postfix(
                        loss=f"{loss_ema:.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    )

        # Store per-epoch weighted average for evaluate() to consume
        self._epoch_loss_sum = epoch_loss_sum  # type: ignore[attr-defined]
        self._epoch_sample_count = epoch_sample_count  # type: ignore[attr-defined]

    def fit(
        self, model: torch.nn.Module, train_data: CorpusDataset, val_data: CorpusDataset
    ) -> None:
        train_loader, val_loader = _new_loaders(train_data, val_data, self.cfg)
        model.to(self.cfg.device)
        start = time.time()

        # Resolve default epochs if auto (-1)
        epochs = self.cfg.epochs
        if epochs == -1:
            epochs = default_epochs(self.model_type)

        logger.info(
            f"[{self.model_type}] fitting model...",
        )
        logger.info(
            f"[{self.model_type}] config: model={model.__class__.__name__}, "
            f"epochs={epochs}, device={self.cfg.device}"
        )

        best_val_loss = float("inf")
        patience_counter = 0
        checkpoint_dir = self.cfg.checkpoint_dir
        checkpoint_every = self.cfg.checkpoint_every
        checkpoint_best = self.cfg.checkpoint_best

        # Create checkpoint directory if checkpointing is enabled
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        try:
            for epoch in range(1, epochs + 1):
                self._train_epoch(model, train_loader, epoch)
                self.evaluate(model, val_loader, epoch)

                if self.scheduler:
                    self.scheduler.step()

                # Save periodic checkpoint
                if checkpoint_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                    ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
                    save_checkpoint(model, self.optimizer, epoch, ckpt_path)
                    logger.info(f"[{self.model_type}] saved checkpoint: {ckpt_path}")

                # Save best model checkpoint
                if checkpoint_dir and checkpoint_best and self.val_loss < best_val_loss:
                    best_path = Path(checkpoint_dir) / "best_model.pth"
                    save_checkpoint(model, self.optimizer, epoch, best_path)
                    logger.info(
                        f"[{self.model_type}] saved best model checkpoint "
                        f"(val_loss={self.val_loss:.4f}): {best_path}"
                    )

                # Early stopping based on validation loss
                if self.cfg.early_stopping_patience > 0:
                    if self.val_loss < best_val_loss:
                        best_val_loss = self.val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.cfg.early_stopping_patience:
                            logger.info(
                                f"[{self.model_type}] early stopping at epoch {epoch}, "
                                f"val_loss hasn't improved for {patience_counter} epochs"
                            )
                            break

                logger.info(
                    f"[{self.model_type}] [epoch {epoch}] completed, val_loss={self.val_loss:.4f}"
                )
        finally:
            # Release CUDA resources even on Ctrl-C or exception.
            # Prevents error 804 ("forward compatibility was attempted on
            # non supported HW") caused by orphaned GPU contexts.
            if self.cfg.device in ("cuda", "mps") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        logger.info(
            f"[{self.model_type}] model fitting completed, {time.time() - start:.0f} seconds passed"
        )

    def evaluate(self, model: torch.nn.Module, val_loader: DataLoader, epoch: int) -> None:
        logger.info(f"[{self.model_type}] [epoch {epoch}] evaluating predictions...")
        model.to(self.cfg.device)
        model.eval()

        all_probs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        val_loss_sum = 0.0
        val_sample_count = 0

        with logging_redirect_tqdm():
            pbar = tqdm(val_loader, desc=f"[{self.model_type}] eval {epoch}", leave=False)
            with torch.no_grad():
                for sent, target in pbar:
                    batch_size = target.size(0)
                    sent = sent.to(self.cfg.device)
                    target = target.to(self.cfg.device)
                    logits = model(sent)
                    loss_val = self.loss_function(logits, target)
                    val_loss_sum += loss_val.item() * batch_size
                    val_sample_count += batch_size

                    all_probs.append(torch.softmax(logits, dim=-1).cpu())
                    all_targets.append(target.cpu())

        probabilities = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        metrics = compute_epoch_metrics(probabilities, targets, self.model_type)

        self.val_loss = val_loss_sum / val_sample_count if val_sample_count > 0 else 0.0
        # Per-epoch weighted train loss from _train_epoch
        train_loss_sum = getattr(self, "_epoch_loss_sum", 0.0)
        train_sample_count = getattr(self, "_epoch_sample_count", 0)
        self.latest_train_loss = (
            train_loss_sum / train_sample_count if train_sample_count > 0 else 0.0
        )
        self.latest_epoch = epoch
        self.latest_metrics = metrics

        # Publish metrics to all backends (Ray gauges, Prometheus, JSON, logger)
        publish_epoch_metrics(
            model_type=self.model_type,
            epoch=epoch,
            train_loss=self.latest_train_loss,
            val_loss=self.val_loss,
            metrics=metrics,
            lr=self.optimizer.param_groups[0]["lr"],
            ray_gauges=_get_ray_gauges(self.model_type),
        )


def _run_training_loop(
    model: torch.nn.Module,
    train_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    val_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_function: Callable,
    callbacks: list[TrainingCallback],
    model_type: str,
    device: str,
) -> TrainingState:
    """Unified training loop used by all three paths (single-node, distributed, tuning).

    Args:
        model: The model to train.
        train_iter: Iterator yielding (data, target) batches for training.
        val_iter: Iterator yielding (data, target) batches for validation.
        epochs: Number of epochs to train.
        optimizer: Optimizer for training.
        scheduler: Optional LR scheduler.
        loss_function: Loss function.
        callbacks: List of TrainingCallback instances.
        model_type: Model type string for logging.
        device: Device string.

    Returns:
        TrainingState with final training state.
    """
    state = TrainingState()

    for cb in callbacks:
        cb.on_train_begin(model_type, device, epochs)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        loss_ema = 0.0
        update_every = 50

        with logging_redirect_tqdm():
            pbar = tqdm(train_iter, desc=f"[{model_type}] epoch {epoch}", leave=False)
            for i, (data, target) in enumerate(pbar):
                batch_size = target.size(0)
                loss_val = train_step(
                    model,
                    data=data,
                    target=target,
                    optimizer=optimizer,
                    loss_function=loss_function,
                )
                epoch_loss_sum += loss_val * batch_size
                epoch_sample_count += batch_size
                state.running_loss_mean = (
                    (state.running_loss_mean * state.steps + loss_val) / (state.steps + 1)
                    if state.steps > 0
                    else loss_val
                )
                state.steps += 1
                loss_ema = 0.9 * loss_ema + 0.1 * loss_val if i > 0 else loss_val
                if i % update_every == 0:
                    pbar.set_postfix(
                        loss=f"{loss_ema:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                    )

        train_loss = epoch_loss_sum / epoch_sample_count if epoch_sample_count > 0 else 0.0
        state.latest_train_loss = train_loss

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for data, target in val_iter:
                batch_size = target.size(0)
                output = model(data)
                loss_val = loss_function(output, target)
                val_loss_sum += loss_val.item() * batch_size
                val_sample_count += batch_size

                all_probs.append(torch.softmax(output, dim=-1).cpu())
                all_targets.append(target.cpu())

        val_loss = val_loss_sum / val_sample_count if val_sample_count > 0 else 0.0
        probabilities = torch.cat(all_probs).numpy() if all_probs else np.array([])
        targets = torch.cat(all_targets).numpy() if all_targets else np.array([])

        metrics = compute_epoch_metrics(probabilities, targets, model_type)

        lr = optimizer.param_groups[0]["lr"]
        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            lr=lr,
        )

        state.val_loss = val_loss
        state.latest_epoch = epoch
        state.latest_metrics = metrics

        if scheduler is not None:
            scheduler.step()

        # Callbacks
        should_stop = False
        for cb in callbacks:
            if cb.on_epoch_end(result, state):
                should_stop = True
                break

        # Handle checkpointing signaled by CheckpointCallback
        if hasattr(state, "_pending_checkpoint_path"):
            save_checkpoint(model, optimizer, epoch, state._pending_checkpoint_path)
            delattr(state, "_pending_checkpoint_path")

        if hasattr(state, "_pending_best_checkpoint_path"):
            save_checkpoint(model, optimizer, epoch, state._pending_best_checkpoint_path)
            delattr(state, "_pending_best_checkpoint_path")

        if should_stop:
            break

    for cb in callbacks:
        cb.on_train_end(state)

    return state


def _get_opt_params(
    model_type: str,
) -> OptimizationParams | EncoderOptimizationParams | DecoderOptimizationParams:
    """Return optimization params appropriate for the model type."""
    if model_type == "decoder":
        return DecoderOptimizationParams()
    if model_type == "encoder":
        return EncoderOptimizationParams()
    if model_type == "rnn":
        return OptimizationParams()
    raise ValueError(f"no matching model: {model_type}")


def _get_sched_params(
    model_type: str,
) -> SchedulerParams | EncoderSchedulerParams | DecoderSchedulerParams:
    """Return scheduler params appropriate for the model type."""
    if model_type == "encoder":
        return EncoderSchedulerParams()
    if model_type == "decoder":
        return DecoderSchedulerParams()
    if model_type == "rnn":
        return SchedulerParams()
    raise ValueError(f"no matching model: {model_type}")


class _LinearWarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay.

    During warmup, LR increases linearly from base_lr/warmup_steps to base_lr.
    After warmup, follows CosineAnnealing decay to eta_min.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ) -> None:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return eta_min + (1.0 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


def new_trainer(
    model: torch.nn.Module,
    cfg: TrainerConfig,
    model_type: str,
) -> Trainer:
    # Build class_weights tensor from config if provided
    class_weights_tensor = None
    if cfg.class_weights is not None:
        class_weights_tensor = torch.tensor(cfg.class_weights, dtype=torch.float32)

    optimizer, scheduler, loss_function = _create_training_components(
        model=model,
        model_type=model_type,
        device=cfg.device,
        cfg=cfg,
        class_weights=class_weights_tensor,
    )

    trainer = Trainer(
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        model_type=model_type,
    )

    return trainer


# ──────────────────────────────────────────────
# Ray Train distributed training
# ──────────────────────────────────────────────


def _train_func(config: dict) -> None:
    """Training function executed on each Ray Train worker.

    Creates the model on the worker, wraps it with DDP via prepare_model(),
    and runs the training loop with dataset shards from Ray Data.
    Reports metrics and checkpoints back to Ray Train.
    """

    # Suppress smart_open verbose logging in Ray workers.
    # smart_open submodules (s3, http, etc.) log INFO messages during
    # initialization which flood Ray worker output. Setting the parent
    # logger suppresses all children that propagate.
    import logging

    logging.getLogger("smart_open").setLevel(logging.WARNING)
    # Also suppress any child loggers that may already exist
    for _name in list(logging.root.manager.loggerDict):
        if _name.startswith("smart_open"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    # Unpack training config (resolve -1 to model-specific default)
    epochs = config["epochs"]
    if epochs == -1:
        epochs = default_epochs(config["model_type"])
    batch_size = config["batch_size"]
    lr = config["lr"]
    betas = tuple(config["betas"])
    weight_decay = config["weight_decay"]
    use_warmup = config.get("use_warmup", False)
    warmup_steps = config.get("warmup_steps", 0)
    total_steps = config.get("total_steps", 0)
    scheduler_eta_min = config.get("scheduler_eta_min", 1e-6)

    # Unpack model config
    model_type = config["model_type"]
    dict_path = config["dict_path"]
    embeddings_model_name = config["embeddings_model_name"]
    embeddings_emb_length = config["embeddings_emb_length"]
    # input_len is no longer needed by new_model() — removed from config

    # Create model on this worker
    from sentimentizer.config import EmbeddingsConfig

    embeddings_config = EmbeddingsConfig(
        model_name=embeddings_model_name,
        emb_length=embeddings_emb_length,
    )

    if model_type == "rnn":
        from sentimentizer.models.rnn import new_model
    elif model_type == "encoder":
        from sentimentizer.models.encoder import new_model
    elif model_type == "decoder":
        from sentimentizer.models.decoder import new_model
    else:
        raise ValueError(f"no matching model for {model_type}")

    model = new_model(
        dict_path=dict_path,
        embeddings_config=embeddings_config,
        # input_len removed from new_model() signatures
    )

    # Prepare model for distributed training (DDP) and move to correct device
    model = prepare_model(model)

    # Determine device from model (set by prepare_model)
    device = next(model.parameters()).device

    # Set up optimizer (AdamW for all models)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    # Set up scheduler (warmup+cosine for transformers, cosine for RNN)
    if use_warmup and warmup_steps > 0:
        scheduler = _LinearWarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=scheduler_eta_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_eta_min,
        )
    # Use CrossEntropyLoss for 3-class classification
    loss_type = config.get("loss_type", "cross_entropy")
    label_smoothing = config.get("label_smoothing", 0.1)

    # Build class weights tensor from config if provided
    class_weights_list = config.get("class_weights")
    class_weights_tensor = None
    if class_weights_list is not None:
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)

    if loss_type == "focal":
        from sentimentizer.losses import FocalCrossEntropyLoss

        focal_gamma = config.get("focal_gamma", 2.0)
        loss_function = FocalCrossEntropyLoss(
            weight=class_weights_tensor,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    else:
        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor,
            label_smoothing=label_smoothing,
        )

    # Get dataset shards for this worker
    # In Ray 2.55+, get_dataset_shard is a standalone function, not a method on get_context()
    # See https://docs.ray.io/en/2.55.1/train/api/doc/ray.train.get_dataset_shard.html
    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("val")

    start = time.time()
    logger.info(
        f"[{model_type}] fitting model (distributed)...",
    )
    logger.info(
        f"[{model_type}] config: model={model.__class__.__name__}, "
        f"epochs={epochs}, lr={lr}, device={device}, "
        f"use_warmup={use_warmup}, warmup_steps={warmup_steps}, total_steps={total_steps}"
    )

    is_rank_0 = train.get_context().get_world_rank() == 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        ray_loss_ema = 0.0
        ray_update_every = 50

        with logging_redirect_tqdm():
            train_pbar = tqdm(
                train_shard.iter_torch_batches(batch_size=batch_size),
                desc=f"[{model_type}] epoch {epoch}",
                leave=False,
                disable=not is_rank_0,
            )
            for i, batch in enumerate(train_pbar):
                data = batch["data"].long().to(device)
                target = batch["target"].long().to(device)
                bs = target.size(0)
                loss_val = train_step(
                    model,
                    data=data,
                    target=target,
                    optimizer=optimizer,
                    loss_function=loss_function,
                )
                epoch_loss_sum += loss_val * bs
                epoch_sample_count += bs
                ray_loss_ema = 0.9 * ray_loss_ema + 0.1 * loss_val if i > 0 else loss_val
                if i % ray_update_every == 0:
                    train_pbar.set_postfix(
                        loss=f"{ray_loss_ema:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                    )

        if scheduler:
            scheduler.step()

        # Validation
        logger.info(f"[{model_type}] [epoch {epoch}] evaluating predictions...")
        val_loss_sum = 0.0
        val_sample_count = 0
        all_probs = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for batch in val_shard.iter_torch_batches(batch_size=batch_size):
                data = batch["data"].long().to(device)
                target = batch["target"].long().to(device)
                bs = target.size(0)
                logits = model(data)
                loss_val = loss_function(logits, target)
                val_loss_sum += loss_val.item() * bs
                val_sample_count += bs

                all_probs.append(torch.softmax(logits, dim=-1).cpu())
                all_targets.append(target.cpu())

        val_loss = val_loss_sum / val_sample_count if val_sample_count > 0 else 0.0
        train_loss = epoch_loss_sum / epoch_sample_count if epoch_sample_count > 0 else 0.0

        probabilities = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        metrics = compute_epoch_metrics(probabilities, targets, model_type)

        # Publish metrics from rank 0 to prevent worker collisions
        if train.get_context().get_world_rank() == 0:
            publish_epoch_metrics(
                model_type=model_type,
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                metrics=metrics,
                lr=optimizer.param_groups[0]["lr"],
                ray_gauges=_get_ray_gauges(model_type),
            )

        # Report metrics and checkpoint to Ray Train
        # Ray 2.55+ requires directory-based checkpoints (from_dict removed)
        import tempfile

        import ray.cloudpickle as pickle

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        # NOTE: train.report() with checkpoint= reads the checkpoint files
        # from disk when called, so the temp directory must still exist.
        # Keep train.report() INSIDE the with-block to avoid FileNotFoundError
        # from pyarrow.fs.copy_files when the directory is auto-deleted.
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "data.pkl"), "wb") as fp:
                pickle.dump(checkpoint_data, fp)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": metrics.accuracy,
                    "balanced_accuracy": metrics.balanced_accuracy,
                    "negative_precision": metrics.negative_precision,
                    "negative_recall": metrics.negative_recall,
                    "negative_f1": metrics.negative_f1,
                    "neutral_precision": metrics.neutral_precision,
                    "neutral_recall": metrics.neutral_recall,
                    "neutral_f1": metrics.neutral_f1,
                    "positive_precision": metrics.positive_precision,
                    "positive_recall": metrics.positive_recall,
                    "positive_f1": metrics.positive_f1,
                    "macro_f1": metrics.macro_f1,
                    "weighted_f1": metrics.weighted_f1,
                    "cohen_kappa": metrics.cohen_kappa,
                    "mcc": metrics.mcc,
                    "total": metrics.total,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                checkpoint=checkpoint,
            )

    logger.info(f"[{model_type}] model fitting completed, {time.time() - start:.0f} seconds passed")


# ──────────────────────────────────────────────
# Checkpoint save / load
# ──────────────────────────────────────────────


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str | Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    val_loss: float | None = None,
) -> None:
    """Save a training checkpoint to disk.

    Saves model state_dict, optimizer state_dict, scheduler state_dict
    (if provided), epoch number, and optionally val_loss.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer to checkpoint.
        epoch: Current epoch number (1-based).
        path: File path to save the checkpoint (.pth).
        scheduler: Optional LR scheduler to include in the checkpoint.
        val_loss: Optional validation loss to include in the checkpoint.
    """
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if val_loss is not None:
        checkpoint["val_loss"] = val_loss

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"checkpoint saved: {path} (epoch={epoch})")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint from disk.

    Restores model weights, optimizer state, and scheduler state
    (if present in the checkpoint). Returns the checkpoint dict
    with metadata (epoch, val_loss, etc.).

    Args:
        path: File path to the checkpoint (.pth).
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map tensors to when loading.

    Returns:
        Dict with checkpoint metadata (epoch, val_loss, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"checkpoint loaded: {path} (epoch={checkpoint.get('epoch', '?')})")

    return checkpoint


def list_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """List all checkpoint files in a directory, sorted by epoch.

    Returns:
        List of Path objects for each .pth checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


# ──────────────────────────────────────────────
# Ray Train distributed training
# ──────────────────────────────────────────────


def new_ray_trainer(
    train_ds: ray.data.Dataset,
    val_ds: ray.data.Dataset,
    cfg: TrainerConfig,
    model_type: str,
    driver_config: type[DriverConfig] = DriverConfig,
) -> TorchTrainer:
    """Factory function to create a Ray Train TorchTrainer for distributed training.

    See Ray Train docs:
    https://docs.ray.io/en/2.55.1/train/api/doc/ray.train.torch.TorchTrainer.html

    Args:
        train_ds: Ray Dataset for training
        val_ds: Ray Dataset for validation
        cfg: TrainerConfig with training hyperparameters
        model_type: Model type ("rnn", "encoder", or "decoder")
        driver_config: DriverConfig class with file paths and model config

    Returns:
        Configured TorchTrainer ready to call .fit()
    """

    # Get model-specific optimization and scheduler params
    opt = _get_opt_params(model_type)
    sched = _get_sched_params(model_type)

    warmup_sched = isinstance(sched, (EncoderSchedulerParams, DecoderSchedulerParams))
    use_warmup = warmup_sched and sched.warmup_epochs > 0
    if warmup_sched:
        warmup_steps = sched.warmup_epochs
        total_steps = sched.T_max
    else:
        warmup_steps = 0
        total_steps = 0

    # Build train_loop_config with everything the worker needs to
    # create the model and run training (models cannot be passed across workers)
    train_loop_config = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": opt.lr,
        "betas": list(opt.betas),
        "weight_decay": opt.weight_decay,
        "use_warmup": use_warmup,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "scheduler_eta_min": sched.eta_min,
        "model_type": model_type,
        "dict_path": driver_config.files.dictionary_file_path,
        "embeddings_model_name": driver_config.embeddings.model_name,
        "embeddings_emb_length": driver_config.embeddings.emb_length,
        "loss_type": cfg.loss_type,
        "label_smoothing": cfg.label_smoothing,
        "focal_gamma": cfg.focal_gamma,
        "class_weights": cfg.class_weights,
    }

    use_gpu = cfg.device in ("cuda", "mps")

    trainer = TorchTrainer(
        train_loop_per_worker=_train_func,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(
            num_workers=cfg.ray_workers,
            use_gpu=use_gpu,
        ),
        datasets={"train": train_ds, "val": val_ds},
    )

    return trainer
