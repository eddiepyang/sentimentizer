"""Centralized metrics publishing for training and evaluation.

Consolidates the gauge-setting, JSON persistence, and logging logic that
was previously duplicated across Trainer.evaluate() and _train_func().

Instead of 4 near-identical blocks of gauge-setting code, both callers
now use a single publish_epoch_metrics() function.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from sentimentizer.metrics import ClassificationMetrics

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Metric field definitions
# ──────────────────────────────────────────────

# Mapping from ClassificationMetrics attribute name → Prometheus/Ray gauge key name.
# This single source of truth replaces the 4× duplicated field-mapping blocks.
_METRIC_GAUGE_KEYS: list[tuple[str, str, bool]] = [
    # (metrics_attr, gauge_key, is_optional_none)
    # gauge_key matches both Ray gauge dict keys and Prometheus gauge variable prefixes
    ("accuracy", "val_accuracy", False),
    ("precision", "val_precision", False),
    ("recall", "val_recall", False),
    ("f1", "val_f1", False),
    ("cohen_kappa", "val_cohen_kappa", False),
    ("mcc", "val_mcc", False),
    ("npv", "val_npv", False),
    ("macro_f1", "val_macro_f1", False),
    ("auc_roc", "val_auc_roc", True),
    ("avg_precision", "val_avg_precision", True),
    ("positive_accuracy", "val_positive_accuracy", False),
    ("negative_accuracy", "val_negative_accuracy", False),
]

# ──────────────────────────────────────────────
# JSON metrics persistence
# ──────────────────────────────────────────────

_METRICS_DIR = Path("/tmp/sentimentizer_metrics")


def get_metrics_dir() -> Path:
    """Return the directory used for per-model metrics JSON files."""
    return _METRICS_DIR


def write_epoch_metrics_to_file(
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
    training processes never race on a shared JSON file.
    """
    import contextlib

    metrics_dir = get_metrics_dir()
    with contextlib.suppress(OSError):
        metrics_dir.mkdir(parents=True, exist_ok=True)

    path = metrics_dir / f"{model_type}_metrics.json"

    data: dict[str, Any] = {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "cohen_kappa": float(metrics.cohen_kappa),
        "mcc": float(metrics.mcc),
        "npv": float(metrics.npv),
        "macro_f1": float(metrics.macro_f1),
        "auc_roc": float(metrics.auc_roc) if metrics.auc_roc is not None else None,
        "avg_precision": (
            float(metrics.avg_precision) if metrics.avg_precision is not None else None
        ),
        "positive_accuracy": float(metrics.positive_accuracy),
        "negative_accuracy": float(metrics.negative_accuracy),
        "epoch": int(epoch),
        "lr": float(lr) if lr is not None else None,
        "_written_by": model_type,
        "_written_at": time.time(),
    }

    with contextlib.suppress(OSError):
        path.write_text(json.dumps(data, indent=2))


# ──────────────────────────────────────────────
# Ray gauge publishing
# ──────────────────────────────────────────────


def _set_ray_gauges(
    gauges: dict[str, Any],
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float,
) -> None:
    """Set all Ray custom metric gauges for a training epoch.

    Args:
        gauges: Dict of gauge name → Ray Gauge object (from _get_ray_gauges).
        epoch: Current epoch number.
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch.
        metrics: Computed classification metrics.
        lr: Current learning rate.
    """
    gauges["train_loss"].set(float(train_loss))
    gauges["val_loss"].set(float(val_loss))
    gauges["lr"].set(float(lr))
    gauges["epoch"].set(epoch)

    for attr, key, is_optional_none in _METRIC_GAUGE_KEYS:
        value = getattr(metrics, attr)
        if is_optional_none and value is None:
            continue
        gauges[key].set(float(value))


# ──────────────────────────────────────────────
# Prometheus exporter gauge publishing
# ──────────────────────────────────────────────


def _set_prometheus_gauges(
    model_type: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float,
) -> None:
    """Set all standalone Prometheus exporter gauges for a training epoch.

    Imports the gauge objects lazily since the exporter module may not be
    available in all contexts (e.g., Ray workers).
    """
    try:
        from sentimentizer.exporter import (
            TRAINING_EPOCH,
            TRAINING_LR,
            TRAINING_TRAIN_LOSS,
            TRAINING_VAL_ACCURACY,
            TRAINING_VAL_AUC_ROC,
            TRAINING_VAL_AVG_PRECISION,
            TRAINING_VAL_COHEN_KAPPA,
            TRAINING_VAL_F1,
            TRAINING_VAL_LOSS,
            TRAINING_VAL_MACRO_F1,
            TRAINING_VAL_MCC,
            TRAINING_VAL_NEGATIVE_ACCURACY,
            TRAINING_VAL_NPV,
            TRAINING_VAL_POSITIVE_ACCURACY,
            TRAINING_VAL_PRECISION,
            TRAINING_VAL_RECALL,
        )
    except ImportError:
        return

    lbl = {"model_type": model_type}
    TRAINING_TRAIN_LOSS.labels(**lbl).set(float(train_loss))
    TRAINING_VAL_LOSS.labels(**lbl).set(float(val_loss))
    TRAINING_EPOCH.labels(**lbl).set(epoch)
    TRAINING_LR.labels(**lbl).set(float(lr))

    # Use the same field mapping table for consistency
    _PROM_GAUGE_MAP: dict[str, Any] = {
        "accuracy": TRAINING_VAL_ACCURACY,
        "precision": TRAINING_VAL_PRECISION,
        "recall": TRAINING_VAL_RECALL,
        "f1": TRAINING_VAL_F1,
        "cohen_kappa": TRAINING_VAL_COHEN_KAPPA,
        "mcc": TRAINING_VAL_MCC,
        "npv": TRAINING_VAL_NPV,
        "macro_f1": TRAINING_VAL_MACRO_F1,
        "auc_roc": TRAINING_VAL_AUC_ROC,
        "avg_precision": TRAINING_VAL_AVG_PRECISION,
        "positive_accuracy": TRAINING_VAL_POSITIVE_ACCURACY,
        "negative_accuracy": TRAINING_VAL_NEGATIVE_ACCURACY,
    }

    for attr, gauge in _PROM_GAUGE_MAP.items():
        value = getattr(metrics, attr)
        if value is None:
            continue
        gauge.labels(**lbl).set(float(value))


# ──────────────────────────────────────────────
# Unified publishing entry point
# ──────────────────────────────────────────────


def publish_epoch_metrics(
    *,
    model_type: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: ClassificationMetrics,
    lr: float,
    ray_gauges: dict[str, Any] | None = None,
) -> None:
    """Publish training metrics to all backends: Ray gauges, Prometheus, JSON file, and logger.

    This is the single entry point that replaces the 4 duplicated metric-publishing
    blocks that were previously in Trainer.evaluate() and _train_func().

    Args:
        model_type: Model type label (e.g. "rnn", "encoder", "decoder").
        epoch: Current training epoch number.
        train_loss: Average training loss for the epoch.
        val_loss: Average validation loss for the epoch.
        metrics: Computed classification metrics.
        lr: Current learning rate.
        ray_gauges: Optional dict of Ray Gauge objects (from _get_ray_gauges).
            If None, Ray gauge publishing is skipped (e.g., in single-node mode
            where Ray is not running).
    """
    # 1. Ray custom metrics (only if gauges are provided — skip if not in Ray context)
    if ray_gauges is not None:
        _set_ray_gauges(
            gauges=ray_gauges,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            lr=lr,
        )

    # 2. Standalone Prometheus exporter gauges (best-effort, skip if not importable)
    _set_prometheus_gauges(
        model_type=model_type,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        metrics=metrics,
        lr=lr,
    )

    # 3. Persist to JSON for the standalone exporter
    write_epoch_metrics_to_file(
        model_type=model_type,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        metrics=metrics,
        lr=lr,
    )

    # 4. Structured logging — format metrics into message string since this
    #    module uses logging.getLogger (not structlog), which does not accept
    #    arbitrary keyword arguments.
    auc_roc_str = f"{metrics.auc_roc:.4f}" if metrics.auc_roc is not None else "N/A"
    avg_precision_str = (
        f"{metrics.avg_precision:.4f}" if metrics.avg_precision is not None else "N/A"
    )
    logger.info(
        f"[{model_type}] [epoch {epoch}] evaluation complete — "
        f"val_loss={val_loss:.4f} accuracy={metrics.accuracy:.4f} "
        f"precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
        f"f1={metrics.f1:.4f} cohen_kappa={metrics.cohen_kappa:.4f} "
        f"mcc={metrics.mcc:.4f} npv={metrics.npv:.4f} "
        f"macro_f1={metrics.macro_f1:.4f} "
        f"auc_roc={auc_roc_str} avg_precision={avg_precision_str} "
        f"pos_acc={metrics.positive_accuracy:.4f} "
        f"neg_acc={metrics.negative_accuracy:.4f}"
    )
