"""Classification metrics for sentiment model evaluation.

Provides comprehensive metrics beyond simple accuracy, including
per-class accuracy, precision/recall/F1, Cohen's kappa, and AUC-ROC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

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
        auc_roc: Area under the ROC curve (None if sklearn is unavailable).
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
    auc_roc: float | None = None
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
            "auc_roc": round(self.auc_roc, 4) if self.auc_roc is not None else None,
            "confusion_matrix": {
                "tp": self.tp,
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn,
            },
            "total": self.total,
        }


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
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if probabilities is not None and isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    predictions = np.asarray(predictions, dtype=np.int64).flatten()
    targets = np.asarray(targets, dtype=np.int64).flatten()

    total = len(targets)
    if total == 0:
        return ClassificationMetrics(total=0)

    # Confusion matrix components
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))

    # Overall accuracy
    accuracy = (tp + tn) / max(total, 1)

    # Per-class accuracy
    positive_total = tp + fn  # actual positives
    negative_total = tn + fp  # actual negatives
    positive_accuracy = tp / max(positive_total, 1)
    negative_accuracy = tn / max(negative_total, 1)

    # Precision, recall, F1 (positive class)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-10)

    # Cohen's kappa
    cohen_kappa = _cohen_kappa(tp, tn, fp, fn, total)

    # AUC-ROC (optional, requires probability scores)
    auc_roc: float | None = None
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64).flatten()
        auc_roc = _auc_roc(probabilities, targets)

    return ClassificationMetrics(
        accuracy=accuracy,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cohen_kappa=cohen_kappa,
        auc_roc=auc_roc,
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

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-10)

    cohen_kappa = _cohen_kappa(tp, tn, fp, fn, total)

    # AUC-ROC requires probability scores, which we have in 'score'
    probabilities = np.array([r.get("score", 0.5) for r in results])
    targets = np.array([1 if r.get("expected") == "positive" else 0 for r in results])

    auc_roc = _auc_roc(probabilities, targets)

    return ClassificationMetrics(
        accuracy=accuracy,
        positive_accuracy=positive_accuracy,
        negative_accuracy=negative_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        cohen_kappa=cohen_kappa,
        auc_roc=auc_roc,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cohen_kappa(tp: int, tn: int, fp: int, fn: int, total: int) -> float:
    """Compute Cohen's kappa coefficient from confusion matrix counts.

    Cohen's kappa measures inter-rater agreement, correcting for
    agreement occurring by chance. Range: -1 to 1 (1 = perfect agreement).

    Formula: kappa = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.
    """
    if total == 0:
        return 0.0

    # Observed agreement
    p_o = (tp + tn) / total

    # Expected agreement by chance
    # P(predicted positive) * P(actual positive) + P(predicted negative) * P(actual negative)
    p_pred_pos = (tp + fp) / total
    p_actual_pos = (tp + fn) / total
    p_pred_neg = (tn + fn) / total
    p_actual_neg = (tn + fp) / total
    p_e = p_pred_pos * p_actual_pos + p_pred_neg * p_actual_neg

    if p_e == 1.0:
        return 1.0  # Perfect agreement edge case

    return (p_o - p_e) / (1.0 - p_e)


def _auc_roc(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Compute Area Under the ROC Curve using the trapezoidal rule.

    Falls back to a simple implementation if scikit-learn is unavailable.
    """
    try:
        from sklearn.metrics import roc_auc_score

        # Check if both classes are present
        if len(np.unique(targets)) < 2:
            logger.warning("auc_roc_only_one_class", message="AUC-ROC requires both classes")
            return 0.0

        return float(roc_auc_score(targets, probabilities))
    except ImportError:
        logger.warning("sklearn_not_available", message="Computing AUC-ROC with trapezoidal rule")
        return _auc_roc_manual(probabilities, targets)


def _auc_roc_manual(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Manual AUC-ROC computation using the trapezoidal rule.

    Sorts by probability descending and computes the ROC curve,
    then integrates using the trapezoidal rule.
    """
    sorted_indices = np.argsort(-probabilities)
    sorted_targets = targets[sorted_indices]

    total_pos = max(np.sum(targets == 1), 1)
    total_neg = max(np.sum(targets == 0), 1)

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_count = 0
    fp_count = 0

    for i in range(len(sorted_targets)):
        if sorted_targets[i] == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_list.append(tp_count / total_pos)
        fpr_list.append(fp_count / total_neg)

    # Trapezoidal rule for AUC
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    auc = float(np.trapz(tpr_arr, fpr_arr))

    return max(0.0, min(1.0, auc))
