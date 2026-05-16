"""Characterization tests for DRY/SOLID refactoring.

These tests capture the CURRENT behavior of code patterns that will be
refactored. They serve as regression safeguards: if a refactoring changes
behavior, these tests will fail.

Covers:
- Metric publishing: all gauge keys, JSON fields, value consistency
- Model predict(): output shape, range, device handling
- Config defaults: OptimizationParams, SchedulerParams per model type
- Validation loop: metric fields, NaN handling
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from sentimentizer.config import (
    DecoderOptimizationParams,
    DecoderSchedulerParams,
    EncoderOptimizationParams,
    EncoderSchedulerParams,
    OptimizationParams,
    SchedulerParams,
    default_dataloader_workers,
    default_epochs,
)
from sentimentizer.metrics import ClassificationMetrics, compute_classification_metrics
from sentimentizer.metrics_publisher import write_epoch_metrics_to_file
from sentimentizer.trainer import (
    Trainer,
    TrainerConfig,
    _get_opt_params,
    _get_sched_params,
)

# ─── Helpers ─────────────────────────────────────────────────────────


class TinyModel(nn.Module):
    """Minimal model for fast training tests."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x))).squeeze(-1)


class FloatDataset(torch.utils.data.Dataset):
    """Simple dataset yielding (float_data, float_target) pairs."""

    def __init__(self, n: int = 32, input_dim: int = 10) -> None:
        self.data = torch.randn(n, input_dim)
        self.targets = torch.randint(0, 2, (n,)).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


# ─── Metric Publishing Characterization ────────────────────────────


class TestMetricPublishingCharacterization:
    """Verify that metric publishing sets ALL expected fields consistently.

    These tests characterize the current behavior of Trainer.evaluate()
    so that refactoring the gauge-setting code into a shared function
    does not accidentally drop any metric.
    """

    EXPECTED_METRIC_FIELDS = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cohen_kappa",
        "mcc",
        "npv",
        "macro_f1",
        "auc_roc",
        "avg_precision",
        "positive_accuracy",
        "negative_accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
        "total",
    ]

    EXPECTED_JSON_KEYS = [
        "train_loss",
        "val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cohen_kappa",
        "mcc",
        "npv",
        "macro_f1",
        "auc_roc",
        "avg_precision",
        "positive_accuracy",
        "negative_accuracy",
        "epoch",
        "lr",
        "_written_by",
        "_written_at",
    ]

    EXPECTED_RAY_GAUGE_KEYS = [
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_cohen_kappa",
        "val_mcc",
        "val_npv",
        "val_macro_f1",
        "val_auc_roc",
        "val_avg_precision",
        "val_positive_accuracy",
        "val_negative_accuracy",
        "epoch",
        "lr",
    ]

    def test_write_epoch_metrics_json_contains_all_fields(self) -> None:
        """_write_epoch_metrics_to_file must include ALL metric fields."""
        metrics = ClassificationMetrics(
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1=0.72,
            cohen_kappa=0.5,
            mcc=0.4,
            npv=0.6,
            macro_f1=0.65,
            auc_roc=0.85,
            avg_precision=0.78,
            positive_accuracy=0.8,
            negative_accuracy=0.7,
            tp=4,
            tn=3,
            fp=1,
            fn=1,
            total=9,
        )

        write_epoch_metrics_to_file(
            model_type="test_char_metrics",
            epoch=3,
            train_loss=0.42,
            val_loss=0.55,
            metrics=metrics,
            lr=0.001,
        )

        path = Path("/tmp/sentimentizer_metrics") / "test_char_metrics_metrics.json"
        assert path.exists()
        data = json.loads(path.read_text())

        # Verify ALL expected JSON keys are present
        for key in self.EXPECTED_JSON_KEYS:
            assert key in data, f"Missing key '{key}' in metrics JSON"

        # Verify values match input
        assert data["accuracy"] == 0.8
        assert data["mcc"] == 0.4
        assert data["npv"] == 0.6
        assert data["macro_f1"] == 0.65
        assert data["auc_roc"] == 0.85
        assert data["avg_precision"] == 0.78
        assert data["positive_accuracy"] == 0.8
        assert data["negative_accuracy"] == 0.7
        assert data["epoch"] == 3
        assert data["lr"] == 0.001
        assert data["_written_by"] == "test_char_metrics"

        # Clean up
        path.unlink(missing_ok=True)

    def test_json_handles_none_auc_roc_and_avg_precision(self) -> None:
        """_write_epoch_metrics_to_file must handle None auc_roc/avg_precision."""
        metrics = ClassificationMetrics(
            accuracy=0.75,
            precision=0.7,
            recall=0.65,
            f1=0.67,
            cohen_kappa=0.4,
            mcc=0.3,
            npv=0.55,
            macro_f1=0.6,
            auc_roc=None,
            avg_precision=None,
            positive_accuracy=0.7,
            negative_accuracy=0.6,
            tp=3,
            tn=2,
            fp=1,
            fn=2,
            total=8,
        )

        write_epoch_metrics_to_file(
            model_type="test_char_none",
            epoch=1,
            train_loss=0.5,
            val_loss=0.6,
            metrics=metrics,
            lr=0.002,
        )

        path = Path("/tmp/sentimentizer_metrics") / "test_char_none_metrics.json"
        assert path.exists()
        data = json.loads(path.read_text())

        assert data["auc_roc"] is None
        assert data["avg_precision"] is None

        path.unlink(missing_ok=True)

    def test_evaluate_sets_all_ray_gauge_keys(self) -> None:
        """Trainer.evaluate() must call .set() on ALL 16 Ray gauge keys.

        This is a regression safeguard: if a metric gauge is accidentally
        dropped during refactoring, this test will fail.
        """
        mock_model = MagicMock()
        mock_loader = MagicMock()
        mock_loader.dataset = [1, 2]
        mock_loader.__iter__.return_value = iter([(torch.zeros(4, 10), torch.ones(4, 1))])

        cfg = TrainerConfig(device="cpu")
        trainer = Trainer(
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            cfg=cfg,
            model_type="test_ray_keys",
        )

        # Create mock gauges for ALL expected keys
        mock_gauges = {key: MagicMock() for key in self.EXPECTED_RAY_GAUGE_KEYS}

        with (
            patch("sentimentizer.trainer._get_ray_gauges", return_value=mock_gauges),
            patch("sentimentizer.trainer._write_epoch_metrics_to_file"),
            patch("sentimentizer.exporter.TRAINING_TRAIN_LOSS", create=True),
            patch("sentimentizer.exporter.TRAINING_EPOCH", create=True),
            patch("sentimentizer.exporter.TRAINING_LR", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_LOSS", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_ACCURACY", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_PRECISION", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_RECALL", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_F1", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_COHEN_KAPPA", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_MCC", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_NPV", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_MACRO_F1", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_AUC_ROC", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_AVG_PRECISION", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_POSITIVE_ACCURACY", create=True),
            patch("sentimentizer.exporter.TRAINING_VAL_NEGATIVE_ACCURACY", create=True),
        ):
            mock_model.return_value = torch.randn(4, 1)
            trainer.evaluate(mock_model, mock_loader, epoch=1)

            # EVERY Ray gauge key must have been .set() at least once
            for key in self.EXPECTED_RAY_GAUGE_KEYS:
                mock_gauges[key].set.assert_called_once(), (
                    f"Ray gauge '{key}' was not set during evaluate()"
                )

    def test_classification_metrics_has_all_expected_fields(self) -> None:
        """ClassificationMetrics dataclass must have all fields that gauges reference.

        If a field is removed or renamed, the gauge-setting code will break.
        """
        metrics = ClassificationMetrics(
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1=0.72,
            cohen_kappa=0.5,
            mcc=0.4,
            npv=0.6,
            macro_f1=0.65,
            auc_roc=0.85,
            avg_precision=0.78,
            positive_accuracy=0.8,
            negative_accuracy=0.7,
            tp=4,
            tn=3,
            fp=1,
            fn=1,
            total=9,
        )

        for field in self.EXPECTED_METRIC_FIELDS:
            assert hasattr(metrics, field), f"ClassificationMetrics missing field '{field}'"

    def test_evaluate_populates_latest_metrics(self) -> None:
        """Trainer.evaluate() must set latest_metrics with all ClassificationMetrics fields."""
        model = TinyModel(input_dim=10, hidden_dim=16)
        train_data = FloatDataset(n=64, input_dim=10)
        val_data = FloatDataset(n=32, input_dim=10)

        cfg = TrainerConfig(
            batch_size=16,
            epochs=1,
            device="cpu",
            dataloader_workers=0,
            early_stopping_patience=0,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        loss_fn = nn.BCEWithLogitsLoss()

        trainer = Trainer(
            loss_function=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            model_type="test_metrics_populate",
        )

        trainer.fit(model, train_data, val_data)  # type: ignore[arg-type]

        assert trainer.latest_metrics is not None, "latest_metrics should be set after fit()"

        # Verify all fields are populated (not None where not expected)
        for field in self.EXPECTED_METRIC_FIELDS:
            value = getattr(trainer.latest_metrics, field)
            assert value is not None, f"latest_metrics.{field} should not be None after training"

        # Verify auc_roc and avg_precision are computed when probabilities exist
        assert trainer.latest_metrics.auc_roc is not None
        assert trainer.latest_metrics.avg_precision is not None


# ─── Config Defaults Characterization ──────────────────────────────


class TestConfigDefaultsCharacterization:
    """Snapshot current config defaults so refactoring doesn't change them."""

    def test_optimization_params_defaults(self) -> None:
        """OptimizationParams defaults per model type must be preserved."""
        # RNN defaults
        rnn_opt = _get_opt_params("rnn")
        assert isinstance(rnn_opt, OptimizationParams)
        assert rnn_opt.lr == 0.001
        assert rnn_opt.betas == (0.9, 0.999)
        assert rnn_opt.weight_decay == 1e-4

        # Encoder defaults
        enc_opt = _get_opt_params("encoder")
        assert isinstance(enc_opt, EncoderOptimizationParams)
        assert enc_opt.lr == 0.0005
        assert enc_opt.betas == (0.9, 0.999)
        assert enc_opt.weight_decay == 0.01

        # Decoder defaults
        dec_opt = _get_opt_params("decoder")
        assert isinstance(dec_opt, DecoderOptimizationParams)
        assert dec_opt.lr == 0.0003
        assert dec_opt.betas == (0.9, 0.999)
        assert dec_opt.weight_decay == 0.02

    def test_scheduler_params_defaults(self) -> None:
        """SchedulerParams defaults per model type must be preserved."""
        # RNN defaults
        rnn_sched = _get_sched_params("rnn")
        assert isinstance(rnn_sched, SchedulerParams)
        assert rnn_sched.T_max == 4
        assert rnn_sched.eta_min == 1e-6
        assert rnn_sched.last_epoch == -1

        # Encoder defaults
        enc_sched = _get_sched_params("encoder")
        assert isinstance(enc_sched, EncoderSchedulerParams)
        assert enc_sched.T_max == 8
        assert enc_sched.eta_min == 1e-6
        assert enc_sched.last_epoch == -1
        assert enc_sched.warmup_epochs == 1

        # Decoder defaults
        dec_sched = _get_sched_params("decoder")
        assert isinstance(dec_sched, DecoderSchedulerParams)
        assert dec_sched.T_max == 8
        assert dec_sched.eta_min == 1e-5
        assert dec_sched.last_epoch == -1
        assert dec_sched.warmup_epochs == 2

    def test_default_epochs_per_model(self) -> None:
        """default_epochs() must return correct values per model type."""
        assert default_epochs("rnn") == 4
        assert default_epochs("encoder") == 8
        assert default_epochs("decoder") == 8
        # Unknown model type should default to 4
        assert default_epochs("unknown") == 4

    def test_default_dataloader_workers_per_device(self) -> None:
        """default_dataloader_workers() must return 0 for MPS."""
        assert default_dataloader_workers("mps") == 0
        # For CPU/CUDA it depends on cpu_count, just check it's positive
        cpu_workers = default_dataloader_workers("cpu")
        assert cpu_workers >= 1

    def test_opt_params_invalid_model_type(self) -> None:
        """_get_opt_params must raise ValueError for invalid model type."""
        with pytest.raises(ValueError, match="no matching model"):
            _get_opt_params("invalid")

    def test_sched_params_invalid_model_type(self) -> None:
        """_get_sched_params must raise ValueError for invalid model type."""
        with pytest.raises(ValueError, match="no matching model"):
            _get_sched_params("invalid")

    def test_optimization_params_dataclass_fields(self) -> None:
        """All OptimizationParams variants must have the same field names."""
        base_fields = {f.name for f in OptimizationParams.__dataclass_fields__.values()}
        enc_fields = {f.name for f in EncoderOptimizationParams.__dataclass_fields__.values()}
        dec_fields = {f.name for f in DecoderOptimizationParams.__dataclass_fields__.values()}
        assert (
            base_fields == enc_fields == dec_fields
        ), "All OptimizationParams variants must share the same field names"

    def test_scheduler_params_dataclass_fields(self) -> None:
        """SchedulerParams base must have fields that encoder/decoder extend."""
        base_fields = {f.name for f in SchedulerParams.__dataclass_fields__.values()}
        enc_fields = {f.name for f in EncoderSchedulerParams.__dataclass_fields__.values()}
        dec_fields = {f.name for f in DecoderSchedulerParams.__dataclass_fields__.values()}
        # Encoder and Decoder add warmup_epochs
        assert base_fields.issubset(enc_fields)
        assert base_fields.issubset(dec_fields)
        assert "warmup_epochs" in enc_fields
        assert "warmup_epochs" in dec_fields


# ─── Validation Loop Characterization ──────────────────────────────


class TestValidationLoopCharacterization:
    """Verify that the validation loop produces correct metrics and handles NaN."""

    def test_evaluate_nan_handling_in_probabilities(self) -> None:
        """NaN probabilities must be replaced with 0.5 during evaluation.

        This tests the NaN-replacement path in Trainer.evaluate() that
        will be extracted into _evaluate_model().
        """
        # Test the underlying compute_classification_metrics with NaN
        predictions = np.array([1, 0, 1, 0, 1])
        targets = np.array([1, 0, 1, 0, 1])
        probabilities = np.array([0.9, 0.1, float("nan"), 0.2, 0.8])

        metrics = compute_classification_metrics(predictions, targets, probabilities)
        # NaN should be handled gracefully
        assert metrics.auc_roc is not None
        assert 0.0 <= metrics.auc_roc <= 1.0

    def test_compute_classification_metrics_all_fields_populated(self) -> None:
        """compute_classification_metrics must return all ClassificationMetrics fields."""
        predictions = np.array([1, 1, 0, 0, 1])
        targets = np.array([1, 0, 0, 0, 1])
        probabilities = np.array([0.9, 0.2, 0.6, 0.3, 0.8])

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
        )

        # All scalar fields must be non-None
        assert metrics.accuracy is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1 is not None
        assert metrics.cohen_kappa is not None
        assert metrics.mcc is not None
        assert metrics.npv is not None
        assert metrics.macro_f1 is not None
        assert metrics.positive_accuracy is not None
        assert metrics.negative_accuracy is not None
        assert metrics.auc_roc is not None
        assert metrics.avg_precision is not None

        # Confusion matrix counts must be consistent
        assert metrics.tp + metrics.tn + metrics.fp + metrics.fn == metrics.total
        assert metrics.total == len(predictions)

    def test_compute_metrics_without_probabilities(self) -> None:
        """Metrics computed without probabilities must have None auc_roc/avg_precision."""
        predictions = np.array([1, 0, 1, 0])
        targets = np.array([1, 0, 1, 0])

        metrics = compute_classification_metrics(predictions=predictions, targets=targets)

        assert metrics.auc_roc is None
        assert metrics.avg_precision is None
        # Other metrics should still be computed
        assert metrics.accuracy == 1.0
        assert metrics.f1 == 1.0


# ─── Model predict() Characterization ──────────────────────────────


class TestModelPredictCharacterization:
    """Verify that predict() behavior is consistent across all models.

    These tests characterize the predict() method that will be extracted
    into a BaseSentimentModel base class.
    """

    def test_rnn_predict_returns_valid_output(self) -> None:
        """RNN.predict() must return a tensor in [0, 1] with correct shape."""
        from sentimentizer.models.rnn import RNN

        # Create a small RNN model
        emb_weights = torch.randn(100, 50)  # small vocab, small embedding
        model = RNN(emb_weights=emb_weights, hidden_size=32, num_layers=1, dropout=0.0)
        model.eval()

        # Use batch_size=2 because torch.squeeze collapses batch=1 to scalar
        converted_text = np.zeros((2, 10), dtype=np.int64)
        converted_text[:, :5] = np.random.randint(1, 100, size=(2, 5))

        output = model.predict(converted_text)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2, f"Expected batch dim 2, got {output.shape}"
        assert (output >= 0.0).all() and (output <= 1.0).all(), "Output not in [0, 1]"

    def test_encoder_predict_returns_valid_output(self) -> None:
        """Encoder.predict() must return a tensor in [0, 1] with correct shape."""
        from sentimentizer.models.encoder import Encoder

        emb_weights = torch.randn(100, 50)
        model = Encoder(emb_weights=emb_weights, d_model=32, n_heads=2, n_layers=1, dropout=0.0)
        model.eval()

        converted_text = np.zeros((2, 10), dtype=np.int64)
        converted_text[:, :5] = np.random.randint(1, 100, size=(2, 5))

        output = model.predict(converted_text)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2, f"Expected batch dim 2, got {output.shape}"
        assert (output >= 0.0).all() and (output <= 1.0).all(), "Output not in [0, 1]"

    def test_decoder_predict_returns_valid_output(self) -> None:
        """Decoder.predict() must return a tensor in [0, 1] with correct shape."""
        from sentimentizer.models.decoder import Decoder

        emb_weights = torch.randn(100, 50)
        model = Decoder(
            emb_weights=emb_weights,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dropout=0.0,
        )
        model.eval()

        converted_text = np.zeros((2, 10), dtype=np.int64)
        converted_text[:, :5] = np.random.randint(1, 100, size=(2, 5))

        output = model.predict(converted_text)
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 2, f"Expected batch dim 2, got {output.shape}"
        assert (output >= 0.0).all() and (output <= 1.0).all(), "Output not in [0, 1]"

    def test_predict_output_is_sigmoid_of_forward(self) -> None:
        """predict() must return sigmoid(forward(x)), not raw logits."""
        from sentimentizer.models.rnn import RNN

        emb_weights = torch.randn(100, 50)
        model = RNN(emb_weights=emb_weights, hidden_size=32, num_layers=1, dropout=0.0)
        model.eval()

        converted_text = np.zeros((1, 10), dtype=np.int64)
        converted_text[0, :5] = np.random.randint(1, 100, size=5)

        predict_output = model.predict(converted_text)

        with torch.no_grad():
            input_tensor = torch.from_numpy(converted_text)
            logits = model(input_tensor)
            expected = torch.sigmoid(logits)

        assert torch.allclose(
            predict_output, expected, atol=1e-6
        ), "predict() output must match sigmoid(forward())"

    def test_predict_does_not_compute_gradients(self) -> None:
        """predict() must not compute gradients (torch.no_grad context)."""
        from sentimentizer.models.rnn import RNN

        emb_weights = torch.randn(100, 50)
        model = RNN(emb_weights=emb_weights, hidden_size=32, num_layers=1, dropout=0.0)
        model.eval()

        converted_text = np.zeros((1, 10), dtype=np.int64)
        converted_text[0, :5] = np.random.randint(1, 100, size=5)

        # Reset gradients
        for p in model.parameters():
            p.grad = None

        model.predict(converted_text)

        # No gradients should be computed
        for p in model.parameters():
            assert p.grad is None, "predict() should not compute gradients"


# ─── Metric Value Consistency Characterization ──────────────────────


class TestMetricValueConsistency:
    """Verify that metrics written to JSON, Ray gauges, and Prometheus gauges
    all receive the same numeric values from a single evaluate() call.

    After refactoring, the single publish_epoch_metrics() function must
    produce identical output to the current 4-block duplication.
    """

    def test_json_and_compute_metrics_values_match(self) -> None:
        """JSON output values must exactly match ClassificationMetrics fields."""
        metrics = ClassificationMetrics(
            accuracy=0.875,
            precision=0.889,
            recall=0.889,
            f1=0.889,
            cohen_kappa=0.75,
            mcc=0.5,
            npv=0.833,
            macro_f1=0.875,
            auc_roc=0.944,
            avg_precision=0.917,
            positive_accuracy=0.889,
            negative_accuracy=0.857,
            tp=8,
            tn=6,
            fp=1,
            fn=1,
            total=16,
        )

        write_epoch_metrics_to_file(
            model_type="test_consistency",
            epoch=5,
            train_loss=0.123,
            val_loss=0.456,
            metrics=metrics,
            lr=0.0005,
        )

        path = Path("/tmp/sentimentizer_metrics") / "test_consistency_metrics.json"
        assert path.exists()
        data = json.loads(path.read_text())

        # Each JSON value must match the corresponding ClassificationMetrics field
        assert data["accuracy"] == metrics.accuracy
        assert data["precision"] == metrics.precision
        assert data["recall"] == metrics.recall
        assert data["f1"] == metrics.f1
        assert data["cohen_kappa"] == metrics.cohen_kappa
        assert data["mcc"] == metrics.mcc
        assert data["npv"] == metrics.npv
        assert data["macro_f1"] == metrics.macro_f1
        assert data["auc_roc"] == metrics.auc_roc
        assert data["avg_precision"] == metrics.avg_precision
        assert data["positive_accuracy"] == metrics.positive_accuracy
        assert data["negative_accuracy"] == metrics.negative_accuracy
        assert data["train_loss"] == 0.123
        assert data["val_loss"] == 0.456
        assert data["epoch"] == 5

        path.unlink(missing_ok=True)
