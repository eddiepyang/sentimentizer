from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

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

# ---------------------------------------------------------------------------
# Shared metrics persistence helpers
# ---------------------------------------------------------------------------


def _write_epoch_metrics_to_file(
    *,
    model_type: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float | None = None,
) -> None:
    """Write current epoch metrics to the per-model JSON file.

    Each model type writes to its own file
    (``/tmp/sentimentizer_metrics/{model_type}_metrics.json``) so concurrent
    training processes never race on a shared JSON file.  The standalone
    exporter discovers the file directly by model type.
    """
    import contextlib
    import json
    import time
    from pathlib import Path

    metrics_dir = Path("/tmp/sentimentizer_metrics")
    with contextlib.suppress(OSError):
        metrics_dir.mkdir(parents=True, exist_ok=True)

    path = metrics_dir / f"{model_type}_metrics.json"

    auc_roc = metrics.auc_roc
    data = {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "cohen_kappa": float(metrics.cohen_kappa),
        "auc_roc": float(auc_roc) if auc_roc is not None else None,
        "positive_accuracy": float(metrics.positive_accuracy),
        "negative_accuracy": float(metrics.negative_accuracy),
        "epoch": int(epoch),
        "lr": float(lr) if lr is not None else None,
        "_written_by": model_type,
        "_written_at": time.time(),
    }

    with contextlib.suppress(OSError):
        path.write_text(json.dumps(data, indent=2))


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
        "val_precision": Gauge(
            "sentimentizer_live_val_precision",
            description="Live validation precision (positive class)",
            tag_keys=("model_type",),
        ),
        "val_recall": Gauge(
            "sentimentizer_live_val_recall",
            description="Live validation recall (positive class)",
            tag_keys=("model_type",),
        ),
        "val_f1": Gauge(
            "sentimentizer_live_val_f1",
            description="Live validation F1 score",
            tag_keys=("model_type",),
        ),
        "val_cohen_kappa": Gauge(
            "sentimentizer_live_val_cohen_kappa",
            description="Live validation Cohen's kappa",
            tag_keys=("model_type",),
        ),
        "val_auc_roc": Gauge(
            "sentimentizer_live_val_auc_roc",
            description="Live validation AUC-ROC",
            tag_keys=("model_type",),
        ),
        "val_positive_accuracy": Gauge(
            "sentimentizer_live_val_positive_accuracy",
            description="Live validation positive-class accuracy",
            tag_keys=("model_type",),
        ),
        "val_negative_accuracy": Gauge(
            "sentimentizer_live_val_negative_accuracy",
            description="Live validation negative-class accuracy",
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
    losses: list[float] = field(default_factory=list)
    val_loss: float = float("inf")
    latest_train_loss: float = 0.0
    latest_epoch: int = 0
    latest_metrics: ClassificationMetrics | None = None

    def _train_epoch(self, model: torch.nn.Module, train_loader: DataLoader, epoch: int) -> None:
        i = 0
        n = len(train_loader.dataset)  # type: ignore[arg-type]
        model.train()

        for j, (sent, target) in enumerate(train_loader):
            loss_val = train_step(
                model,
                data=sent.to(self.cfg.device),
                target=target.to(self.cfg.device),
                optimizer=self.optimizer,
                loss_function=self.loss_function,
            )

            i += len(target)
            self.losses.append(loss_val)
            if i % (self.cfg.batch_size * 100) == 0:
                current_loss = float(np.mean(self.losses[-60:]))
                gauges = _get_ray_gauges(self.model_type)
                if gauges is not None:
                    gauges["train_loss"].set(current_loss)
                try:
                    from sentimentizer.exporter import TRAINING_TRAIN_LOSS

                    TRAINING_TRAIN_LOSS.labels(model_type=self.model_type).set(current_loss)
                except ImportError:
                    pass
                logger.info(
                    f"[{self.model_type}] [epoch {epoch}] {i / n:.2f} of rows completed in "
                    f"{j + 1} cycles, current loss at {np.mean(self.losses[-60:]):.4f}"
                )
                logger.info(
                    f"[{self.model_type}] [epoch {epoch}] current learning rate at {self.optimizer.param_groups[0]['lr']:.4f}"  # noqa: E501
                )

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
            for epoch in range(epochs):
                self._train_epoch(model, train_loader, epoch)
                self.evaluate(model, val_loader, epoch)

                if self.scheduler:
                    self.scheduler.step()

                # Save periodic checkpoint
                if checkpoint_dir and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                    ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
                    save_checkpoint(model, self.optimizer, epoch + 1, ckpt_path)
                    logger.info(f"[{self.model_type}] saved checkpoint: {ckpt_path}")

                # Save best model checkpoint
                if checkpoint_dir and checkpoint_best and self.val_loss < best_val_loss:
                    best_path = Path(checkpoint_dir) / "best_model.pth"
                    save_checkpoint(model, self.optimizer, epoch + 1, best_path)
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
        losses = []

        with torch.no_grad():
            for sent, target in val_loader:
                sent = sent.to(self.cfg.device)
                target = target.to(self.cfg.device)
                logits = model(sent)
                loss_val = self.loss_function(logits, target)
                losses.append(loss_val.item())

                all_probs.append(torch.sigmoid(logits).cpu())
                all_targets.append(target.cpu())

        probabilities = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        # Replace NaN probabilities with 0.5 (can result from extreme logit values)
        nan_mask = np.isnan(probabilities)
        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            logger.warning(
                f"[{self.model_type}] nan_in_probabilities",
                message=f"Found {nan_count} NaN values in probabilities, replacing with 0.5",
            )
            probabilities = np.where(nan_mask, 0.5, probabilities)

        predictions = (probabilities >= 0.5).astype(int)

        from sentimentizer.metrics import compute_classification_metrics

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
        )

        self.val_loss = float(np.mean(losses))
        self.latest_train_loss = float(np.mean(self.losses)) if self.losses else 0.0
        self.latest_epoch = epoch
        self.latest_metrics = metrics
        gauges = _get_ray_gauges(self.model_type)
        if gauges is not None:
            gauges["val_loss"].set(self.val_loss)
            gauges["val_accuracy"].set(float(metrics.accuracy))
            gauges["val_precision"].set(float(metrics.precision))
            gauges["val_recall"].set(float(metrics.recall))
            gauges["val_f1"].set(float(metrics.f1))
            gauges["val_cohen_kappa"].set(float(metrics.cohen_kappa))
            if metrics.auc_roc is not None:
                gauges["val_auc_roc"].set(float(metrics.auc_roc))
            gauges["val_positive_accuracy"].set(float(metrics.positive_accuracy))
            gauges["val_negative_accuracy"].set(float(metrics.negative_accuracy))
            gauges["epoch"].set(epoch)
            gauges["lr"].set(float(self.optimizer.param_groups[0]["lr"]))

        # Also push to the standalone Prometheus exporter gauges
        try:
            from sentimentizer.exporter import (
                TRAINING_EPOCH,
                TRAINING_LR,
                TRAINING_VAL_ACCURACY,
                TRAINING_VAL_AUC_ROC,
                TRAINING_VAL_COHEN_KAPPA,
                TRAINING_VAL_F1,
                TRAINING_VAL_LOSS,
                TRAINING_VAL_NEGATIVE_ACCURACY,
                TRAINING_VAL_POSITIVE_ACCURACY,
                TRAINING_VAL_PRECISION,
                TRAINING_VAL_RECALL,
            )

            lbl = {"model_type": self.model_type}
            TRAINING_VAL_LOSS.labels(**lbl).set(self.val_loss)
            TRAINING_VAL_ACCURACY.labels(**lbl).set(float(metrics.accuracy))
            TRAINING_VAL_PRECISION.labels(**lbl).set(float(metrics.precision))
            TRAINING_VAL_RECALL.labels(**lbl).set(float(metrics.recall))
            TRAINING_VAL_F1.labels(**lbl).set(float(metrics.f1))
            TRAINING_VAL_COHEN_KAPPA.labels(**lbl).set(float(metrics.cohen_kappa))
            if metrics.auc_roc is not None:
                TRAINING_VAL_AUC_ROC.labels(**lbl).set(float(metrics.auc_roc))
            TRAINING_VAL_POSITIVE_ACCURACY.labels(**lbl).set(float(metrics.positive_accuracy))
            TRAINING_VAL_NEGATIVE_ACCURACY.labels(**lbl).set(float(metrics.negative_accuracy))
            TRAINING_EPOCH.labels(**lbl).set(epoch)
            TRAINING_LR.labels(**lbl).set(float(self.optimizer.param_groups[0]["lr"]))
        except ImportError:
            pass

        # Write current metrics to the JSON file so the standalone exporter
        # serves up-to-date data in the dashboard table panel.
        _write_epoch_metrics_to_file(
            model_type=self.model_type,
            epoch=epoch,
            train_loss=self.latest_train_loss,
            val_loss=self.val_loss,
            metrics=metrics,
            lr=self.optimizer.param_groups[0]["lr"],
        )

        logger.info(  # type: ignore[call-arg]
            f"[{self.model_type}] [epoch {epoch}] evaluation complete",
            model_type=self.model_type,
            val_loss=round(self.val_loss, 4),
            accuracy=round(metrics.accuracy, 4),
            precision=round(metrics.precision, 4),
            recall=round(metrics.recall, 4),
            f1=round(metrics.f1, 4),
            cohen_kappa=round(metrics.cohen_kappa, 4),
            auc_roc=round(metrics.auc_roc, 4) if metrics.auc_roc is not None else None,
            pos_acc=round(metrics.positive_accuracy, 4),
            neg_acc=round(metrics.negative_accuracy, 4),
        )


def _get_opt_params(model_type: str) -> OptimizationParams | EncoderOptimizationParams:
    """Return optimization params appropriate for the model type."""
    if model_type in ("encoder", "decoder"):
        return EncoderOptimizationParams()
    elif model_type == "rnn":
        return OptimizationParams()
    raise ValueError(f"no matching model: {model_type}")


def _get_sched_params(model_type: str) -> SchedulerParams | EncoderSchedulerParams:
    """Return scheduler params appropriate for the model type."""
    if model_type in ("encoder", "decoder"):
        return EncoderSchedulerParams()
    elif model_type == "rnn":
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
    opt = _get_opt_params(model_type)
    sched = _get_sched_params(model_type)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        betas=opt.betas,
        weight_decay=opt.weight_decay,
    )

    # Use warmup+cosine for transformer models, simple cosine for RNN
    if isinstance(sched, EncoderSchedulerParams) and sched.warmup_epochs > 0:
        # Estimate total steps: epochs * batches_per_epoch
        # We'll use a rough estimate; the scheduler steps per optimizer call
        warmup_steps = sched.warmup_epochs  # number of epoch-level steps for warmup
        total_steps = sched.T_max  # total epoch-level steps
        scheduler = _LinearWarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=sched.eta_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched.T_max,
            eta_min=sched.eta_min,
            last_epoch=sched.last_epoch,
        )

    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.pos_weight]).to(cfg.device)
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
    input_len = config["input_len"]

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
        input_len=input_len,
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
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.get("pos_weight", 1.0)]).to(device)
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

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for i, batch in enumerate(train_shard.iter_torch_batches(batch_size=batch_size)):
            loss_val = train_step(
                model,
                data=batch["data"].long().to(device),
                target=batch["target"].float().to(device),
                optimizer=optimizer,
                loss_function=loss_function,
            )
            epoch_losses.append(loss_val)

            # Update Ray custom metrics gauge from rank 0 to prevent worker collisions
            if i % 100 == 0 and train.get_context().get_world_rank() == 0:
                gauges = _get_ray_gauges(model_type)
                if gauges is not None:
                    gauges["train_loss"].set(float(np.mean(epoch_losses[-60:])))
                try:
                    from sentimentizer.exporter import TRAINING_TRAIN_LOSS

                    TRAINING_TRAIN_LOSS.labels(model_type=model_type).set(
                        float(np.mean(epoch_losses[-60:]))
                    )
                except ImportError:
                    pass

        if scheduler:
            scheduler.step()

        # Validation
        logger.info(f"[{model_type}] [epoch {epoch}] evaluating predictions...")
        val_losses = []
        all_probs = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for batch in val_shard.iter_torch_batches(batch_size=batch_size):
                data = batch["data"].long().to(device)
                target = batch["target"].float().to(device)
                logits = model(data)
                loss_val = loss_function(logits, target)
                val_losses.append(loss_val.item())

                all_probs.append(torch.sigmoid(logits).cpu())
                all_targets.append(target.cpu())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        from sentimentizer.metrics import compute_classification_metrics

        probabilities = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()

        # Replace NaN probabilities with 0.5 (can result from extreme logit values)
        nan_mask = np.isnan(probabilities)
        if nan_mask.any():
            nan_count = int(nan_mask.sum())
            logger.warning(
                f"[{model_type}] nan_in_probabilities",
                message=f"Found {nan_count} NaN values in probabilities, replacing with 0.5",
            )
            probabilities = np.where(nan_mask, 0.5, probabilities)

        predictions = (probabilities >= 0.5).astype(int)

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
        )

        # Update Ray custom metrics gauges from rank 0 to prevent worker collisions
        if train.get_context().get_world_rank() == 0:
            gauges = _get_ray_gauges(model_type)
            if gauges is not None:
                gauges["val_loss"].set(float(val_loss))
                gauges["val_accuracy"].set(float(metrics.accuracy))
                gauges["val_precision"].set(float(metrics.precision))
                gauges["val_recall"].set(float(metrics.recall))
                gauges["val_f1"].set(float(metrics.f1))
                gauges["val_cohen_kappa"].set(float(metrics.cohen_kappa))
                if metrics.auc_roc is not None:
                    gauges["val_auc_roc"].set(float(metrics.auc_roc))
                gauges["val_positive_accuracy"].set(float(metrics.positive_accuracy))
                gauges["val_negative_accuracy"].set(float(metrics.negative_accuracy))
                gauges["epoch"].set(epoch)
                gauges["lr"].set(float(optimizer.param_groups[0]["lr"]))

        # Also push to standalone Prometheus exporter gauges from rank 0
        try:
            from sentimentizer.exporter import (  # noqa: E402
                TRAINING_EPOCH,
                TRAINING_LR,
                TRAINING_TRAIN_LOSS,
                TRAINING_VAL_ACCURACY,
                TRAINING_VAL_AUC_ROC,
                TRAINING_VAL_COHEN_KAPPA,
                TRAINING_VAL_F1,
                TRAINING_VAL_LOSS,
                TRAINING_VAL_NEGATIVE_ACCURACY,
                TRAINING_VAL_POSITIVE_ACCURACY,
                TRAINING_VAL_PRECISION,
                TRAINING_VAL_RECALL,
            )

            if train.get_context().get_world_rank() == 0:
                lbl = {"model_type": model_type}
                TRAINING_TRAIN_LOSS.labels(**lbl).set(float(train_loss))
                TRAINING_VAL_LOSS.labels(**lbl).set(float(val_loss))
                TRAINING_VAL_ACCURACY.labels(**lbl).set(float(metrics.accuracy))
                TRAINING_VAL_PRECISION.labels(**lbl).set(float(metrics.precision))
                TRAINING_VAL_RECALL.labels(**lbl).set(float(metrics.recall))
                TRAINING_VAL_F1.labels(**lbl).set(float(metrics.f1))
                TRAINING_VAL_COHEN_KAPPA.labels(**lbl).set(float(metrics.cohen_kappa))
                if metrics.auc_roc is not None:
                    TRAINING_VAL_AUC_ROC.labels(**lbl).set(float(metrics.auc_roc))
                TRAINING_VAL_POSITIVE_ACCURACY.labels(**lbl).set(float(metrics.positive_accuracy))
                TRAINING_VAL_NEGATIVE_ACCURACY.labels(**lbl).set(float(metrics.negative_accuracy))
                TRAINING_EPOCH.labels(**lbl).set(epoch)
                TRAINING_LR.labels(**lbl).set(float(optimizer.param_groups[0]["lr"]))

                # Write current epoch metrics to the JSON file so the standalone
                # exporter serves up-to-date data in the dashboard table panel.
                _write_epoch_metrics_to_file(
                    model_type=model_type,
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    metrics=metrics,
                    lr=optimizer.param_groups[0]["lr"],
                )
        except ImportError:
            pass

        logger.info(  # type: ignore[call-arg]
            f"[{model_type}] [epoch {epoch}] completed",
            model_type=model_type,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            accuracy=round(metrics.accuracy, 4),
            precision=round(metrics.precision, 4),
            recall=round(metrics.recall, 4),
            f1=round(metrics.f1, 4),
            cohen_kappa=round(metrics.cohen_kappa, 4),
            auc_roc=round(metrics.auc_roc, 4) if metrics.auc_roc is not None else None,
            pos_acc=round(metrics.positive_accuracy, 4),
            neg_acc=round(metrics.negative_accuracy, 4),
        )

        # Report metrics and checkpoint to Ray Train
        # Ray 2.55+ requires directory-based checkpoints (from_dict removed)
        import os
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
                    "pos_acc": metrics.positive_accuracy,
                    "neg_acc": metrics.negative_accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "cohen_kappa": metrics.cohen_kappa,
                    "auc_roc": metrics.auc_roc,
                    "tp": metrics.tp,
                    "tn": metrics.tn,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
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

    use_warmup = isinstance(sched, EncoderSchedulerParams) and sched.warmup_epochs > 0
    if isinstance(sched, EncoderSchedulerParams):
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
        "input_len": driver_config.tokenizer.max_len,
        "pos_weight": cfg.pos_weight,
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
