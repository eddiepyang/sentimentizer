"""Tests for the sentimentizer.metrics module.

Tests cover:
- ClassificationMetrics dataclass defaults and to_dict()
- compute_classification_metrics from arrays
- compute_metrics_from_examples from validation results
- torchmetrics-based precision/recall/F1/Cohen's kappa/AUC-ROC
- Edge cases (empty, single class, NaN probabilities)
"""

import numpy as np
import torch

from sentimentizer.metrics import (
    ClassificationMetrics,
    _replace_nan_probs,
    _safe_item,
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
        assert m.positive_accuracy == 0.0
        assert m.negative_accuracy == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.cohen_kappa == 0.0
        assert m.mcc == 0.0
        assert m.npv == 0.0
        assert m.macro_f1 == 0.0
        assert m.auc_roc is None
        assert m.avg_precision is None
        assert m.tp == 0
        assert m.tn == 0
        assert m.fp == 0
        assert m.fn == 0
        assert m.total == 0

    def test_to_dict(self) -> None:
        """to_dict should produce a flat dictionary with rounded values."""
        m = ClassificationMetrics(
            accuracy=0.875,
            positive_accuracy=0.9,
            negative_accuracy=0.85,
            precision=0.8888,
            recall=0.9,
            f1=0.8947,
            cohen_kappa=0.75,
            mcc=0.6,
            npv=0.85,
            macro_f1=0.78,
            auc_roc=0.92,
            avg_precision=0.88,
            tp=9,
            tn=17,
            fp=2,
            fn=1,
            total=29,
        )
        d = m.to_dict()
        assert d["accuracy"] == 0.875
        assert d["positive_accuracy"] == 0.9
        assert d["negative_accuracy"] == 0.85
        assert d["precision"] == 0.8888  # rounded to 4 decimal places
        assert d["recall"] == 0.9
        assert d["f1"] == 0.8947
        assert d["cohen_kappa"] == 0.75
        assert d["mcc"] == 0.6
        assert d["npv"] == 0.85
        assert d["macro_f1"] == 0.78
        assert d["auc_roc"] == 0.92
        assert d["avg_precision"] == 0.88
        assert d["confusion_matrix"]["tp"] == 9
        assert d["confusion_matrix"]["tn"] == 17
        assert d["confusion_matrix"]["fp"] == 2
        assert d["confusion_matrix"]["fn"] == 1
        assert d["total"] == 29

    def test_to_dict_none_auc(self) -> None:
        """to_dict should handle None auc_roc and avg_precision."""
        m = ClassificationMetrics(accuracy=0.8, auc_roc=None, avg_precision=None)
        d = m.to_dict()
        assert d["auc_roc"] is None
        assert d["avg_precision"] is None


class TestComputeClassificationMetrics:
    """Test compute_classification_metrics from numpy/torch arrays."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should give accuracy=1.0, precision=1.0, etc."""
        predictions = np.array([1, 1, 0, 0, 1, 0])
        targets = np.array([1, 1, 0, 0, 1, 0])
        probabilities = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.05])

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.accuracy == 1.0
        assert m.positive_accuracy == 1.0
        assert m.negative_accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.cohen_kappa == 1.0
        assert m.mcc == 1.0
        assert m.npv == 1.0
        assert m.macro_f1 == 1.0
        assert m.auc_roc == 1.0
        assert m.avg_precision == 1.0
        assert m.tp == 3
        assert m.tn == 3
        assert m.fp == 0
        assert m.fn == 0
        assert m.total == 6

    def test_all_wrong_predictions(self) -> None:
        """All wrong predictions should give accuracy=0.0."""
        predictions = np.array([0, 0, 1, 1])
        targets = np.array([1, 1, 0, 0])

        m = compute_classification_metrics(predictions, targets)
        assert m.accuracy == 0.0
        assert m.tp == 0
        assert m.tn == 0
        assert m.fp == 2
        assert m.fn == 2

    def test_mixed_predictions(self) -> None:
        """Mixed predictions should compute correct metrics."""
        # 4 positive, 4 negative
        # Predictions: 1,1,0,0,1,0,0,1 → actual: 1,1,0,0,1,1,0,0
        predictions = np.array([1, 1, 0, 0, 1, 0, 0, 1])
        targets = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        m = compute_classification_metrics(predictions, targets)
        # idx 0: pred=1, actual=1 → TP
        # idx 1: pred=1, actual=1 → TP
        # idx 2: pred=0, actual=0 → TN
        # idx 3: pred=0, actual=0 → TN
        # idx 4: pred=1, actual=1 → TP
        # idx 5: pred=0, actual=1 → FN
        # idx 6: pred=0, actual=0 → TN
        # idx 7: pred=1, actual=0 → FP
        assert m.tp == 3
        assert m.tn == 3
        assert m.fp == 1
        assert m.fn == 1
        assert m.total == 8
        assert m.accuracy == 6 / 8  # (3+3)/8
        assert m.positive_accuracy == 3 / 4  # TP/(TP+FN) = 3/4
        assert m.negative_accuracy == 3 / 4  # TN/(TN+FP) = 3/4
        assert m.precision == 3 / 4  # TP/(TP+FP) = 3/4
        assert m.recall == 3 / 4  # TP/(TP+FN) = 3/4

    def test_torch_tensors(self) -> None:
        """Should work with torch tensors."""
        predictions = torch.tensor([1, 0, 1, 0])
        targets = torch.tensor([1, 0, 1, 0])

        m = compute_classification_metrics(predictions, targets)
        assert m.accuracy == 1.0
        assert m.tp == 2
        assert m.tn == 2

    def test_empty_arrays(self) -> None:
        """Empty arrays should return zeroed metrics."""
        predictions = np.array([], dtype=np.int64)
        targets = np.array([], dtype=np.int64)

        m = compute_classification_metrics(predictions, targets)
        assert m.total == 0
        assert m.accuracy == 0.0

    def test_auc_roc_with_probabilities(self) -> None:
        """AUC-ROC should be computed when probabilities are provided."""
        predictions = np.array([1, 1, 0, 0])
        targets = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.8, 0.1, 0.2])

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.auc_roc is not None
        assert m.auc_roc == 1.0  # Perfect separation

    def test_auc_roc_none_without_probabilities(self) -> None:
        """AUC-ROC should be None when probabilities are not provided."""
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 1, 0])

        m = compute_classification_metrics(predictions, targets)
        assert m.auc_roc is None


class TestCohenKappa:
    """Test Cohen's kappa calculation via torchmetrics."""

    def test_perfect_agreement(self) -> None:
        """Perfect agreement should give kappa=1.0."""
        predictions = np.array([1, 1, 0, 0, 1, 0])
        targets = np.array([1, 1, 0, 0, 1, 0])
        m = compute_classification_metrics(predictions, targets)
        assert m.cohen_kappa == 1.0

    def test_random_agreement(self) -> None:
        """Random agreement (50/50) should give kappa≈0.0."""
        # 50% positive, 50% negative, predictions match 50% by chance
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        targets = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        m = compute_classification_metrics(predictions, targets)
        assert abs(m.cohen_kappa) < 0.05  # Should be near 0

    def test_single_class_returns_zero(self) -> None:
        """Single-class targets should return kappa=0.0 (torchmetrics returns nan, coerced to 0.0).

        This differs from the previous custom implementation which returned 1.0 for
        perfect single-class agreement. The torchmetrics convention (nan→0.0) is more
        conservative and avoids Prometheus gauge issues with NaN values.
        """
        predictions = np.array([0, 0, 0, 0])
        targets = np.array([0, 0, 0, 0])
        m = compute_classification_metrics(predictions, targets)
        assert m.cohen_kappa == 0.0  # nan→0.0 via _safe_item

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
            {"text": "great", "expected": "positive", "score": 0.9, "correct": True},
            {"text": "terrible", "expected": "negative", "score": 0.1, "correct": True},
            {"text": "amazing", "expected": "positive", "score": 0.95, "correct": True},
            {"text": "awful", "expected": "negative", "score": 0.05, "correct": True},
        ]
        m = compute_metrics_from_examples(results)
        assert m.accuracy == 1.0
        assert m.positive_accuracy == 1.0
        assert m.negative_accuracy == 1.0
        assert m.tp == 2
        assert m.tn == 2
        assert m.fp == 0
        assert m.fn == 0

    def test_mixed_results(self) -> None:
        """Mixed correct/incorrect should compute correctly."""
        results = [
            {"text": "great", "expected": "positive", "score": 0.9, "correct": True},
            {"text": "terrible", "expected": "negative", "score": 0.8, "correct": False},
            {"text": "amazing", "expected": "positive", "score": 0.3, "correct": False},
            {"text": "awful", "expected": "negative", "score": 0.1, "correct": True},
        ]
        m = compute_metrics_from_examples(results)
        assert m.accuracy == 0.5  # 2/4 correct
        assert m.tp == 1  # "great" correct positive
        assert m.tn == 1  # "awful" correct negative
        assert m.fp == 1  # "terrible" predicted positive but actual negative
        assert m.fn == 1  # "amazing" predicted negative but actual positive

    def test_empty_results(self) -> None:
        """Empty results should return zeroed metrics."""
        m = compute_metrics_from_examples([])
        assert m.total == 0
        assert m.accuracy == 0.0

    def test_auc_roc_computed(self) -> None:
        """AUC-ROC should be computed from score values."""
        results = [
            {"text": "great", "expected": "positive", "score": 0.9, "correct": True},
            {"text": "terrible", "expected": "negative", "score": 0.1, "correct": True},
        ]
        m = compute_metrics_from_examples(results)
        assert m.auc_roc is not None
        assert m.auc_roc == 1.0  # Perfect separation


class TestNaNHandling:
    """Test that NaN values in probabilities are handled gracefully."""

    def test_nan_in_probabilities_compute_classification_metrics(self) -> None:
        """NaN probabilities should be replaced with 0.5 without crashing."""
        predictions = np.array([1, 0, 1, 0, 1, 0])
        targets = np.array([1, 0, 1, 0, 0, 1])
        probabilities = np.array([0.9, 0.1, np.nan, 0.2, np.nan, 0.3])

        # Should not raise ValueError
        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.auc_roc is not None
        # NaNs replaced with 0.5 should still allow computation
        assert 0.0 <= m.auc_roc <= 1.0

    def test_all_nan_probabilities(self) -> None:
        """All-NaN probabilities should still produce a valid result (auc_roc≈0.5)."""
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 1, 0])
        probabilities = np.array([np.nan, np.nan, np.nan, np.nan])

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.auc_roc is not None
        # All 0.5 probabilities = random guess = AUC ~0.5
        assert 0.0 <= m.auc_roc <= 1.0

    def test_nan_predictions_unchanged(self) -> None:
        """Predictions should not be affected by NaN in probabilities."""
        predictions = np.array([1, 0, 1])
        targets = np.array([1, 0, 1])
        probabilities = np.array([np.nan, np.nan, np.nan])

        m = compute_classification_metrics(predictions, targets, probabilities)
        # With all NaN probs replaced with 0.5, predictions based on >=0.5 should match
        assert m.accuracy == 1.0  # all correct since targets match predictions


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
        """_replace_nan_probs should replace NaN with 0.5."""
        t = torch.tensor([0.9, float("nan"), 0.5])
        result, count = _replace_nan_probs(t)
        assert count == 1
        assert result[1].item() == 0.5

    def test_safe_item_normal(self) -> None:
        """_safe_item should return float for normal values."""
        assert _safe_item(torch.tensor(0.5)) == 0.5
        assert _safe_item(0.5) == 0.5

    def test_safe_item_nan(self) -> None:
        """_safe_item should convert NaN to 0.0."""
        assert _safe_item(float("nan")) == 0.0
        assert _safe_item(torch.tensor(float("nan"))) == 0.0
