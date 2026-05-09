"""Tests for the sentimentizer.metrics module.

Tests cover:
- ClassificationMetrics dataclass defaults and to_dict()
- compute_classification_metrics from arrays
- compute_metrics_from_examples from validation results
- Cohen's kappa calculation
- AUC-ROC calculation
- Edge cases (empty, single class)
"""

import numpy as np
import torch

from sentimentizer.metrics import (
    ClassificationMetrics,
    _cohen_kappa,
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
        assert m.auc_roc is None
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
            auc_roc=0.92,
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
        assert d["auc_roc"] == 0.92
        assert d["confusion_matrix"]["tp"] == 9
        assert d["confusion_matrix"]["tn"] == 17
        assert d["confusion_matrix"]["fp"] == 2
        assert d["confusion_matrix"]["fn"] == 1
        assert d["total"] == 29

    def test_to_dict_none_auc(self) -> None:
        """to_dict should handle None auc_roc."""
        m = ClassificationMetrics(accuracy=0.8, auc_roc=None)
        d = m.to_dict()
        assert d["auc_roc"] is None


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
        assert m.auc_roc == 1.0
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
        # TP: pred=1, actual=1 → 3 (indices 0,1,4)
        # TN: pred=0, actual=0 → 2 (indices 2,3)
        # FP: pred=1, actual=0 → 1 (index 7)
        # FN: pred=0, actual=1 → 1 (index 5)
        # Wait, let me recount:
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
    """Test Cohen's kappa calculation."""

    def test_perfect_agreement(self) -> None:
        """Perfect agreement should give kappa=1.0."""
        assert _cohen_kappa(tp=10, tn=10, fp=0, fn=0, total=20) == 1.0

    def test_random_agreement(self) -> None:
        """Random agreement (50/50) should give kappa≈0.0."""
        # 50% positive, 50% negative, predictions match 50% by chance
        kappa = _cohen_kappa(tp=25, tn=25, fp=25, fn=25, total=100)
        assert abs(kappa) < 0.05  # Should be near 0

    def test_all_negative(self) -> None:
        """All negative predictions should give kappa based on agreement."""
        # All predictions negative, all actual negative → perfect agreement
        kappa = _cohen_kappa(tp=0, tn=20, fp=0, fn=0, total=20)
        assert kappa == 1.0

    def test_zero_total(self) -> None:
        """Zero total should return 0.0."""
        assert _cohen_kappa(tp=0, tn=0, fp=0, fn=0, total=0) == 0.0


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
        """All-NaN probabilities should still produce a valid result (auc_roc=0.5)."""
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 1, 0])
        probabilities = np.array([np.nan, np.nan, np.nan, np.nan])

        m = compute_classification_metrics(predictions, targets, probabilities)
        assert m.auc_roc is not None
        # All 0.5 probabilities = random guess = AUC ~0.5
        assert 0.0 <= m.auc_roc <= 1.0

    def test_nan_in_probabilities_auc_roc_direct(self) -> None:
        """_auc_roc should handle NaN probabilities gracefully."""
        from sentimentizer.metrics import _auc_roc

        probabilities = np.array([0.9, 0.1, np.nan, 0.2])
        targets = np.array([1, 0, 1, 0])

        result = _auc_roc(probabilities, targets)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_nan_predictions_unchanged(self) -> None:
        """Predictions should not be affected by NaN in probabilities."""
        predictions = np.array([1, 0, 1])
        targets = np.array([1, 0, 1])
        probabilities = np.array([np.nan, np.nan, np.nan])

        m = compute_classification_metrics(predictions, targets, probabilities)
        # With all NaN probs replaced with 0.5, predictions based on >=0.5 should match
        assert m.accuracy == 1.0  # all correct since targets match predictions
