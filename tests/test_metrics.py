"""Tests for the sentimentizer.metrics module.

Tests cover:
- ClassificationMetrics dataclass defaults and to_dict()
- compute_classification_metrics from arrays
- compute_metrics_from_examples from validation results
- torchmetrics-based precision/recall/F1/Cohen's kappa/AUC-ROC
- Edge cases (empty, single class, NaN probabilities)
"""

import numpy as np
import pytest
import torch

from sentimentizer.metrics import (
    ClassificationMetrics,
    _replace_nan_probs,
    _safe_item,
    _safe_item_list,
    _to_float_tensor,
    _to_long_tensor,
    compute_classification_metrics,
    compute_metrics_from_examples,
)


class TestClassificationMetrics:
    """Test ClassificationMetrics dataclass."""

    def test_defaults(self) -> None:
        """ClassificationMetrics should have sensible defaults."""
        m = ClassificationMetrics()
        assert m.accuracy == 0.0
        assert m.balanced_accuracy == 0.0
        assert m.negative_precision == 0.0
        assert m.negative_recall == 0.0
        assert m.negative_f1 == 0.0
        assert m.neutral_precision == 0.0
        assert m.neutral_recall == 0.0
        assert m.neutral_f1 == 0.0
        assert m.positive_precision == 0.0
        assert m.positive_recall == 0.0
        assert m.positive_f1 == 0.0
        assert m.macro_f1 == 0.0
        assert m.weighted_f1 == 0.0
        assert m.confusion_matrix is None
        assert m.neutral_to_positive_rate == 0.0
        assert m.neutral_to_negative_rate == 0.0
        assert m.pred_negative_frac == 0.0
        assert m.pred_neutral_frac == 0.0
        assert m.pred_positive_frac == 0.0
        assert m.cohen_kappa == 0.0
        assert m.mcc == 0.0
        assert m.negative_auc_roc is None
        assert m.neutral_auc_roc is None
        assert m.positive_auc_roc is None
        assert m.negative_avg_precision is None
        assert m.neutral_avg_precision is None
        assert m.positive_avg_precision is None
        assert m.total == 0

    def test_to_dict(self) -> None:
        """to_dict should produce a flat dictionary with rounded values."""
        m = ClassificationMetrics(
            accuracy=0.875,
            balanced_accuracy=0.85,
            negative_precision=0.8888,
            negative_recall=0.9,
            negative_f1=0.8947,
            neutral_precision=0.75,
            neutral_recall=0.8,
            neutral_f1=0.7746,
            positive_precision=0.9,
            positive_recall=0.85,
            positive_f1=0.8744,
            macro_f1=0.78,
            weighted_f1=0.82,
            confusion_matrix=[[5, 1, 0], [0, 8, 2], [1, 0, 12]],
            neutral_to_positive_rate=0.2,
            neutral_to_negative_rate=0.0,
            pred_negative_frac=0.2,
            pred_neutral_frac=0.3,
            pred_positive_frac=0.5,
            cohen_kappa=0.75,
            mcc=0.6,
            negative_auc_roc=0.92,
            negative_avg_precision=0.88,
            neutral_auc_roc=0.85,
            neutral_avg_precision=0.82,
            positive_auc_roc=0.95,
            positive_avg_precision=0.90,
            total=29,
        )
        d = m.to_dict()
        assert d["accuracy"] == 0.875
        assert d["balanced_accuracy"] == 0.85
        assert d["negative_precision"] == 0.8888
        assert d["negative_recall"] == 0.9
        assert d["negative_f1"] == 0.8947
        assert d["neutral_precision"] == 0.75
        assert d["neutral_recall"] == 0.8
        assert d["neutral_f1"] == 0.7746
        assert d["positive_precision"] == 0.9
        assert d["positive_recall"] == 0.85
        assert d["positive_f1"] == 0.8744
        assert d["macro_f1"] == 0.78
        assert d["weighted_f1"] == 0.82
        assert d["confusion_matrix"] == [[5, 1, 0], [0, 8, 2], [1, 0, 12]]
        assert d["cohen_kappa"] == 0.75
        assert d["mcc"] == 0.6
        assert d["negative_auc_roc"] == 0.92
        assert d["negative_avg_precision"] == 0.88
        assert d["neutral_auc_roc"] == 0.85
        assert d["neutral_avg_precision"] == 0.82
        assert d["positive_auc_roc"] == 0.95
        assert d["positive_avg_precision"] == 0.90
        assert d["total"] == 29

    def test_to_dict_none_auc(self) -> None:
        """to_dict should handle None auc_roc and avg_precision fields."""
        m = ClassificationMetrics(accuracy=0.8)
        d = m.to_dict()
        assert d["negative_auc_roc"] is None
        assert d["neutral_auc_roc"] is None
        assert d["positive_auc_roc"] is None
        assert d["negative_avg_precision"] is None
        assert d["neutral_avg_precision"] is None
        assert d["positive_avg_precision"] is None


class TestComputeClassificationMetrics:
    """Test compute_classification_metrics from numpy/torch arrays."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should give accuracy=1.0 and per-class metrics=1.0."""
        predictions = np.array([2, 2, 0, 0, 2, 0, 1, 1])
        targets = np.array([2, 2, 0, 0, 2, 0, 1, 1])
        probabilities = np.array(
            [
                [0.05, 0.05, 0.9],
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.9, 0.05],
            ]
        )

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.accuracy == 1.0
        assert m.balanced_accuracy == 1.0
        assert m.negative_precision == 1.0
        assert m.negative_recall == 1.0
        assert m.negative_f1 == 1.0
        assert m.neutral_precision == 1.0
        assert m.neutral_recall == 1.0
        assert m.neutral_f1 == 1.0
        assert m.positive_precision == 1.0
        assert m.positive_recall == 1.0
        assert m.positive_f1 == 1.0
        assert m.cohen_kappa == 1.0
        assert m.mcc == 1.0
        assert m.macro_f1 == 1.0
        assert m.positive_auc_roc == 1.0
        assert m.positive_avg_precision == 1.0
        assert m.total == 8

    def test_all_wrong_predictions(self) -> None:
        """All wrong predictions should give accuracy=0.0."""
        predictions = np.array([0, 0, 2, 2])
        targets = np.array([2, 2, 0, 0])

        m = compute_classification_metrics(predictions, targets)
        assert m.accuracy == 0.0
        assert m.total == 4

    def test_mixed_predictions(self) -> None:
        """Mixed predictions should compute correct metrics."""
        # 3-class: 0=negative, 1=neutral, 2=positive
        predictions = np.array([2, 2, 0, 0, 2, 0, 0, 2])
        targets = np.array([2, 2, 0, 0, 2, 2, 0, 0])

        m = compute_classification_metrics(predictions, targets)
        assert m.total == 8
        assert m.accuracy == 6 / 8  # (2+2+2) correct out of 8
        assert m.balanced_accuracy > 0
        assert m.macro_f1 > 0

    def test_torch_tensors(self) -> None:
        """Should work with torch tensors."""
        predictions = torch.tensor([2, 0, 2, 0])
        targets = torch.tensor([2, 0, 2, 0])

        m = compute_classification_metrics(predictions, targets)
        assert m.accuracy == 1.0
        assert m.total == 4

    def test_empty_arrays(self) -> None:
        """Empty arrays should return zeroed metrics."""
        predictions = np.array([], dtype=np.int64)
        targets = np.array([], dtype=np.int64)

        m = compute_classification_metrics(predictions, targets)
        assert m.total == 0
        assert m.accuracy == 0.0

    def test_auc_roc_with_probabilities(self) -> None:
        """AUC-ROC should be computed when probabilities are provided."""
        predictions = np.array([2, 2, 0, 0])
        targets = np.array([2, 2, 0, 0])
        probabilities = np.array(
            [
                [0.05, 0.05, 0.9],
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [0.9, 0.05, 0.05],
            ]
        )

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.positive_auc_roc is not None
        assert m.negative_auc_roc is not None

    def test_auc_roc_none_without_probabilities(self) -> None:
        """AUC-ROC fields should be None when probabilities are not provided."""
        predictions = np.array([2, 0, 2, 0])
        targets = np.array([2, 0, 2, 0])

        m = compute_classification_metrics(predictions, targets)
        assert m.negative_auc_roc is None
        assert m.neutral_auc_roc is None
        assert m.positive_auc_roc is None


class TestCohenKappa:
    """Test Cohen's kappa calculation via torchmetrics."""

    def test_perfect_agreement(self) -> None:
        """Perfect agreement should give kappa=1.0."""
        predictions = np.array([2, 2, 0, 0, 2, 0])
        targets = np.array([2, 2, 0, 0, 2, 0])
        m = compute_classification_metrics(predictions, targets)
        assert m.cohen_kappa == 1.0

    def test_random_agreement(self) -> None:
        """Random agreement should give kappa near 0.0."""
        predictions = np.array([2, 2, 2, 2, 0, 0, 0, 0])
        targets = np.array([2, 2, 0, 0, 2, 2, 0, 0])
        m = compute_classification_metrics(predictions, targets)
        assert abs(m.cohen_kappa) < 0.05

    def test_single_class_returns_zero(self) -> None:
        """Single-class targets return kappa=0.0 (torchmetrics nan→0.0)."""
        predictions = np.array([0, 0, 0, 0])
        targets = np.array([0, 0, 0, 0])
        m = compute_classification_metrics(predictions, targets)
        assert m.cohen_kappa == 0.0

    def test_zero_total(self) -> None:
        """Zero total should return zeroed metrics."""
        predictions = np.array([], dtype=np.int64)
        targets = np.array([], dtype=np.int64)
        m = compute_classification_metrics(predictions, targets)
        assert m.cohen_kappa == 0.0


class TestComputeMetricsFromExamples:
    """Test compute_metrics_from_examples from validation result dicts."""

    def test_all_correct(self) -> None:
        """All correct predictions should give accuracy=1.0."""
        results = [
            {
                "text": "great",
                "expected": "positive",
                "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
                "correct": True,
            },
            {
                "text": "terrible",
                "expected": "negative",
                "scores": {"negative": 0.9, "neutral": 0.05, "positive": 0.05},
                "correct": True,
            },
            {
                "text": "amazing",
                "expected": "positive",
                "scores": {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
                "correct": True,
            },
            {
                "text": "awful",
                "expected": "negative",
                "scores": {"negative": 0.95, "neutral": 0.03, "positive": 0.02},
                "correct": True,
            },
        ]
        m = compute_metrics_from_examples(results)
        assert m.accuracy == 1.0
        assert m.total == 4

    def test_mixed_results(self) -> None:
        """Mixed correct/incorrect should compute correctly."""
        results = [
            {
                "text": "great",
                "expected": "positive",
                "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
                "correct": True,
            },
            {
                "text": "terrible",
                "expected": "negative",
                "scores": {"negative": 0.1, "neutral": 0.1, "positive": 0.8},
                "correct": False,
            },
            {
                "text": "amazing",
                "expected": "positive",
                "scores": {"negative": 0.5, "neutral": 0.2, "positive": 0.3},
                "correct": False,
            },
            {
                "text": "awful",
                "expected": "negative",
                "scores": {"negative": 0.9, "neutral": 0.05, "positive": 0.05},
                "correct": True,
            },
        ]
        m = compute_metrics_from_examples(results)
        assert m.accuracy == 0.5  # 2/4 correct
        assert m.total == 4

    def test_empty_results(self) -> None:
        """Empty results should return zeroed metrics."""
        m = compute_metrics_from_examples([])
        assert m.total == 0
        assert m.accuracy == 0.0

    def test_auc_roc_computed(self) -> None:
        """Per-class AUC-ROC should be computed from score values."""
        results = [
            {
                "text": "great",
                "expected": "positive",
                "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
                "correct": True,
            },
            {
                "text": "terrible",
                "expected": "negative",
                "scores": {"negative": 0.9, "neutral": 0.05, "positive": 0.05},
                "correct": True,
            },
        ]
        m = compute_metrics_from_examples(results)
        assert m.positive_auc_roc is not None
        assert m.negative_auc_roc is not None


class TestNaNHandling:
    """Test that NaN values in probabilities are handled gracefully."""

    def test_nan_in_probabilities_compute_classification_metrics(self) -> None:
        """NaN probabilities should be replaced without crashing."""
        predictions = np.array([2, 0, 2, 0, 2, 0])
        targets = np.array([2, 0, 2, 0, 0, 2])
        probabilities = np.array(
            [
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [np.nan, np.nan, np.nan],
                [0.9, 0.05, 0.05],
                [np.nan, np.nan, np.nan],
                [0.05, 0.05, 0.9],
            ]
        )

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.positive_auc_roc is not None
        assert 0.0 <= m.positive_auc_roc <= 1.0

    def test_all_nan_probabilities(self) -> None:
        """All-NaN probabilities should still produce a valid result."""
        predictions = np.array([2, 0, 2, 0])
        targets = np.array([2, 0, 2, 0])
        probabilities = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.negative_auc_roc is not None
        assert 0.0 <= m.negative_auc_roc <= 1.0

    def test_nan_predictions_unchanged(self) -> None:
        """Predictions should not be affected by NaN in probabilities."""
        predictions = np.array([2, 0, 2])
        targets = np.array([2, 0, 2])
        probabilities = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.accuracy == 1.0


class TestHelperFunctions:
    """Test internal helper functions."""

    def test_to_long_tensor_numpy(self) -> None:
        """_to_long_tensor should convert numpy arrays."""
        arr = np.array([1, 0, 1])
        result = _to_long_tensor(arr)
        assert result.dtype == torch.long
        assert result.tolist() == [1, 0, 1]

    def test_to_long_tensor_torch(self) -> None:
        """_to_long_tensor should convert torch tensors."""
        t = torch.tensor([1.0, 0.0, 1.0])
        result = _to_long_tensor(t)
        assert result.dtype == torch.long
        assert result.tolist() == [1, 0, 1]

    def test_to_float_tensor_numpy(self) -> None:
        """_to_float_tensor should convert numpy arrays to float32."""
        arr = np.array([0.9, 0.1])
        result = _to_float_tensor(arr)
        assert result.dtype == torch.float32
        assert abs(result[0].item() - 0.9) < 1e-6

    def test_replace_nan_probs_no_nan(self) -> None:
        """_replace_nan_probs should return unchanged tensor when no NaN."""
        t = torch.tensor([0.9, 0.1, 0.5])
        result, count = _replace_nan_probs(t)
        assert count == 0
        assert torch.equal(result, t)

    def test_replace_nan_probs_with_nan(self) -> None:
        """_replace_nan_probs should replace NaN with 1/num_classes for 1D, re-normalize for 2D."""
        t = torch.tensor([0.9, float("nan"), 0.5])
        result, count = _replace_nan_probs(t)
        assert count == 1
        # 1D: NaN replaced with 1/NUM_CLASSES = 1/3
        assert abs(result[1].item() - 1.0 / 3) < 1e-6

    def test_safe_item_normal(self) -> None:
        """_safe_item should return float for normal values."""
        assert _safe_item(torch.tensor(0.5)) == 0.5
        assert _safe_item(0.5) == 0.5

    def test_safe_item_nan(self) -> None:
        """_safe_item should convert NaN to 0.0."""
        assert _safe_item(float("nan")) == 0.0
        assert _safe_item(torch.tensor(float("nan"))) == 0.0

    def test_safe_item_list_normal(self) -> None:
        """_safe_item_list should convert a tensor to a list of floats."""
        result = _safe_item_list(torch.tensor([0.5, 0.3, 0.2]))
        assert result == pytest.approx([0.5, 0.3, 0.2], abs=1e-6)

    def test_safe_item_list_nan(self) -> None:
        """_safe_item_list should convert NaN values to 0.0."""
        result = _safe_item_list(torch.tensor([0.5, float("nan"), 0.2]))
        assert result == pytest.approx([0.5, 0.0, 0.2], abs=1e-6)

    def test_safe_item_list_empty(self) -> None:
        """_safe_item_list should return empty list for empty tensor."""
        result = _safe_item_list(torch.tensor([]))
        assert result == []
