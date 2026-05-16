"""Classification metrics for sentiment model evaluation.

Provides comprehensive metrics beyond simple accuracy, including
per-class accuracy, precision/recall/F1, Cohen's kappa, and AUC-ROC.

Uses torchmetrics for metric computation, which provides GPU-native
tensor operations and batch accumulation. Edge cases (NaN probabilities,
single-class targets, empty arrays) are handled with explicit guards
to maintain backward-compatible behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCohenKappa,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryNegativePredictiveValue,
    BinaryPrecision,
    BinaryRecall,
    MulticlassF1Score,
)

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


@dataclass
class ClassificationMetrics:
    """Comprehensive classification metrics for binary sentiment models.

    Attributes:
        accuracy: Overall accuracy (correct / total).
        positive_accuracy: Accuracy on positive samples only.
        negative_accuracy: Accuracy on negative samples only.
        precision: Positive-class precision (TP / (TP + FP)).
        recall: Positive-class recall (TP / (TP + FN)).
        f1: Positive-class F1 score (harmonic mean of precision and recall).
        cohen_kappa: Cohen's kappa coefficient (agreement beyond chance).
        mcc: Matthews correlation coefficient (-1 to 1, robust to imbalance).
        npv: Negative predictive value (TN / (TN + FN)).
        macro_f1: Macro-averaged F1 (mean of per-class F1, weights both classes equally).
        auc_roc: Area under the ROC curve (None if not computable).
        avg_precision: Average precision (area under PR curve, None if not computable).
        tp: True positive count.
        tn: True negative count.
        fp: False positive count.
        fn: False negative count.
        total: Total number of samples.
    """

    accuracy: float = 0.0
    positive_accuracy: float = 0.0
    negative_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    cohen_kappa: float = 0.0
    mcc: float = 0.0
    npv: float = 0.0
    macro_f1: float = 0.0
    auc_roc: float | None = None
    avg_precision: float | None = None
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary suitable for JSON serialization."""
        return {
            "accuracy": round(self.accuracy, 4),
            "positive_accuracy": round(self.positive_accuracy, 4),
            "negative_accuracy": round(self.negative_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "mcc": round(self.mcc, 4),
            "npv": round(self.npv, 4),
            "macro_f1": round(self.macro_f1, 4),
            "auc_roc": round(self.auc_roc, 4) if self.auc_roc is not None else None,
            "avg_precision": (
                round(self.avg_precision, 4) if self.avg_precision is not None else None
            ),
            "confusion_matrix": {
                "tp": self.tp,
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn,
            },
            "total": self.total,
        }


def _to_long_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert predictions or targets to a 1-D long tensor on CPU."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().long().flatten()
    return torch.as_tensor(arr, dtype=torch.long).flatten()


def _to_float_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert probabilities to a 1-D float32 tensor on CPU."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().float().flatten()
    return torch.as_tensor(arr, dtype=torch.float32).flatten()


def _replace_nan_probs(probabilities: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Replace NaN values in probabilities with 0.5 and return the count replaced."""
    nan_mask = torch.isnan(probabilities)
    nan_count = int(nan_mask.sum().item())
    if nan_count > 0:
        logger.warning(
            "nan_in_probabilities",
            message=f"Found {nan_count} NaN values in probabilities, replacing with 0.5",
        )
        probabilities = torch.where(
            nan_mask, torch.tensor(0.5, dtype=probabilities.dtype), probabilities
        )
    return probabilities, nan_count


def _safe_item(value: torch.Tensor | float) -> float:
    """Extract a Python float from a tensor, converting NaN to 0.0."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if math.isnan(value):
        return 0.0
    return value


def compute_classification_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    probabilities: np.ndarray | torch.Tensor | None = None,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics from predictions and targets.

    Args:
        predictions: Binary predictions (0 or 1), shape (N,).
        targets: Binary ground truth labels (0 or 1), shape (N,).
        probabilities: Raw probability scores for the positive class, shape (N,).
            Required for AUC-ROC computation. If None, auc_roc will be None.

    Returns:
        ClassificationMetrics with all computed metrics.
    """
    # Convert to tensors
    preds_t = _to_long_tensor(predictions)
    targets_t = _to_long_tensor(targets)

    total = targets_t.shape[0]
    if total == 0:
        return ClassificationMetrics(total=0)

    # Compute confusion matrix counts directly (fast, no torchmetrics overhead)
    tp = int(((preds_t == 1) & (targets_t == 1)).sum().item())
    tn = int(((preds_t == 0) & (targets_t == 0)).sum().item())
    fp = int(((preds_t == 1) & (targets_t == 0)).sum().item())
    fn = int(((preds_t == 0) & (targets_t == 1)).sum().item())

    # Overall accuracy
    accuracy = (tp + tn) / max(total, 1)

    # Per-class accuracy
    positive_total = tp + fn  # actual positives
    negative_total = tn + fp  # actual negatives
    positive_accuracy = tp / max(positive_total, 1)
    negative_accuracy = tn / max(negative_total, 1)

    # Use torchmetrics for precision, recall, F1, Cohen's kappa, MCC, NPV
    precision = _safe_item(BinaryPrecision()(preds_t, targets_t))
    recall = _safe_item(BinaryRecall()(preds_t, targets_t))
    f1 = _safe_item(BinaryF1Score()(preds_t, targets_t))
    cohen_kappa = _safe_item(BinaryCohenKappa()(preds_t, targets_t))
    mcc = _safe_item(BinaryMatthewsCorrCoef()(preds_t, targets_t))
    npv = _safe_item(BinaryNegativePredictiveValue()(preds_t, targets_t))
    macro_f1 = _safe_item(MulticlassF1Score(num_classes=2, average="macro")(preds_t, targets_t))

    # AUC-ROC and Average Precision (optional, require probability scores)
    auc_roc: float | None = None
    avg_precision: float | None = None
    if probabilities is not None:
        probs_t = _to_float_tensor(probabilities)
        probs_t, _nan_count = _replace_nan_probs(probs_t)

        # Check if both classes are present
        if len(torch.unique(targets_t)) < 2:
            logger.warning("auc_roc_only_one_class", message="AUC-ROC requires both classes")
            auc_roc = 0.0
            avg_precision = 0.0
        else:
            auc_roc = _safe_item(BinaryAUROC()(probs_t, targets_t))
            avg_precision = _safe_item(BinaryAveragePrecision()(probs_t, targets_t))

    return ClassificationMetrics(
        accuracy=accuracy,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cohen_kappa=cohen_kappa,
        mcc=mcc,
        npv=npv,
        macro_f1=macro_f1,
        auc_roc=auc_roc,
        avg_precision=avg_precision,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
    )


def compute_metrics_from_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> ClassificationMetrics:
    """Compute classification metrics from a model and dataloader.

    Runs inference on the entire dataset and computes comprehensive metrics
    including per-class accuracy, precision/recall/F1, Cohen's kappa, and
    AUC-ROC.

    Args:
        model: The model to evaluate (should output logits).
        dataloader: DataLoader yielding (inputs, targets) batches.
        device: Device to run inference on.

    Returns:
        ClassificationMetrics with all computed metrics.
    """
    model.to(device)
    model.eval()

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze(-1)

            # Replace NaN probabilities with 0.5 (can occur from extreme logit values)
            nan_mask = torch.isnan(probs)
            if nan_mask.any():
                probs = torch.where(nan_mask, torch.tensor(0.5, device=probs.device), probs)

            preds = (probs >= 0.5).long()

            # Binary targets: convert from float to int
            targets_binary = (targets >= 0.5).long()

            all_preds.append(preds.cpu())
            all_targets.append(targets_binary.cpu())
            all_probs.append(probs.cpu())

    predictions = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    probabilities = torch.cat(all_probs).numpy()

    return compute_classification_metrics(
        predictions=predictions,
        targets=targets,
        probabilities=probabilities,
    )


def compute_metrics_from_examples(
    results: list[dict[str, Any]],
) -> ClassificationMetrics:
    """Compute classification metrics from a list of validation example results.

    Each result dict should have 'expected' ('positive' or 'negative')
    and 'correct' (bool) keys, as produced by the TuningRun validation.

    Args:
        results: List of dicts with 'expected' and 'correct' keys.

    Returns:
        ClassificationMetrics with all computed metrics.
    """
    total = len(results)
    if total == 0:
        return ClassificationMetrics(total=0)

    tp = sum(1 for r in results if r.get("expected") == "positive" and r.get("correct"))
    tn = sum(1 for r in results if r.get("expected") == "negative" and r.get("correct"))
    fp = sum(1 for r in results if r.get("expected") == "negative" and not r.get("correct"))
    fn = sum(1 for r in results if r.get("expected") == "positive" and not r.get("correct"))

    accuracy = (tp + tn) / max(total, 1)

    positive_total = tp + fn
    negative_total = tn + fp
    positive_accuracy = tp / max(positive_total, 1)
    negative_accuracy = tn / max(negative_total, 1)

    # Build tensors for torchmetrics: correct predictions match the expected label,
    # incorrect predictions flip it (positive→0, negative→1).
    targets_list: list[int] = []
    preds_list: list[int] = []
    for r in results:
        expected = r.get("expected")
        correct = r.get("correct", False)
        target = 1 if expected == "positive" else 0
        pred = target if correct else (1 - target)
        targets_list.append(target)
        preds_list.append(pred)

    targets_t = torch.tensor(targets_list, dtype=torch.long)
    preds_t = torch.tensor(preds_list, dtype=torch.long)

    precision = _safe_item(BinaryPrecision()(preds_t, targets_t))
    recall = _safe_item(BinaryRecall()(preds_t, targets_t))
    f1 = _safe_item(BinaryF1Score()(preds_t, targets_t))
    cohen_kappa = _safe_item(BinaryCohenKappa()(preds_t, targets_t))
    mcc = _safe_item(BinaryMatthewsCorrCoef()(preds_t, targets_t))
    npv = _safe_item(BinaryNegativePredictiveValue()(preds_t, targets_t))
    macro_f1 = _safe_item(MulticlassF1Score(num_classes=2, average="macro")(preds_t, targets_t))

    # AUC-ROC and Average Precision require probability scores ('score')
    probabilities = np.array([r.get("score", 0.5) for r in results])
    probs_t = _to_float_tensor(probabilities)
    probs_t, _nan_count = _replace_nan_probs(probs_t)

    if len(torch.unique(targets_t)) < 2:
        auc_roc = 0.0
        avg_precision = 0.0
    else:
        auc_roc = _safe_item(BinaryAUROC()(probs_t, targets_t))
        avg_precision = _safe_item(BinaryAveragePrecision()(probs_t, targets_t))

    return ClassificationMetrics(
        accuracy=accuracy,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cohen_kappa=cohen_kappa,
        mcc=mcc,
        npv=npv,
        macro_f1=macro_f1,
        auc_roc=auc_roc,
        avg_precision=avg_precision,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
    )
