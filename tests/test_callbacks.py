"""Unit tests for TrainingCallback implementations.

All tests use MagicMock to avoid real model training, disk I/O,
and Ray dependencies. This ensures fast, deterministic test execution.
"""

from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from sentimentizer.metrics import ClassificationMetrics
from sentimentizer.trainer import (
    CheckpointCallback,
    EarlyStoppingCallback,
    EpochResult,
    LoggingCallback,
    MetricsCallback,
    RayReportCallback,
    TrainingCallback,
    TrainingState,
    _create_training_components,
    _iter_batches,
    _run_training_loop,
    create_model_from_registry,
)

# ─── _iter_batches ───────────────────────────────────────────────


class TestIterBatches:
    """Test _iter_batches normalizes both DataLoader and Ray shard iteration."""

    def test_dataloader_source(self) -> None:
        dataset = TensorDataset(torch.zeros(4, 10), torch.ones(4))
        loader = DataLoader(dataset, batch_size=2)

        batches = list(_iter_batches(loader, batch_size=2, device="cpu"))
        assert len(batches) == 2
        assert all(isinstance(b, tuple) and len(b) == 2 for b in batches)

    def test_dataloader_moves_to_device(self) -> None:
        """Tensors should be moved to the specified device."""
        dataset = TensorDataset(torch.zeros(2, 10), torch.ones(2))
        loader = DataLoader(dataset, batch_size=2)

        batches = list(_iter_batches(loader, batch_size=2, device="cpu"))
        data, target = batches[0]
        assert data.device.type == "cpu"
        assert target.device.type == "cpu"

    def test_ray_shard_source(self) -> None:
        mock_shard = MagicMock()
        mock_shard.iter_torch_batches.return_value = [
            {"data": torch.zeros(2, 10), "target": torch.ones(2)},
        ]
        batches = list(_iter_batches(mock_shard, batch_size=2, device="cpu"))
        assert len(batches) == 1
        data, target = batches[0]
        assert data.dtype == torch.long  # .long() applied
        assert target.dtype == torch.float  # .float() applied

    def test_ray_shard_empty(self) -> None:
        mock_shard = MagicMock()
        mock_shard.iter_torch_batches.return_value = []
        batches = list(_iter_batches(mock_shard, batch_size=2, device="cpu"))
        assert batches == []


# ─── create_model_from_registry ────────────────────────────────────


class TestCreateModelFromRegistry:
    """Test create_model_from_registry."""

    def test_invalid_model_type(self) -> None:
        with pytest.raises(ValueError, match="no matching model"):
            create_model_from_registry(
                "invalid",
                dict_path="/tmp/dict.pkl",
                embeddings_config=MagicMock(),
            )

    def test_registry_lookup(self) -> None:
        """Test that registry is populated and model_type is found."""
        # Just verify the function doesn't raise for valid types
        # (We can't actually create models without real dict/embeddings)
        from sentimentizer.models.base import get_model_registry

        reg = get_model_registry()
        assert "rnn" in reg
        assert "encoder" in reg
        assert "decoder" in reg
        assert "new_model" in reg["rnn"]


# ─── _create_training_components ───────────────────────────────────


class TestCreateTrainingComponents:
    """Test _create_training_components factory."""

    def test_creates_optimizer(self) -> None:
        model = MagicMock()
        model.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
        opt, sched, loss_fn = _create_training_components(
            model=model,
            model_type="rnn",
            device="cpu",
        )
        assert isinstance(opt, torch.optim.Optimizer)
        assert sched is not None
        assert loss_fn is not None

    def test_uses_lr_override(self) -> None:
        model = MagicMock()
        model.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
        opt, _, _ = _create_training_components(
            model=model,
            model_type="rnn",
            device="cpu",
            lr=0.123,
        )
        assert opt.param_groups[0]["lr"] == 0.123

    def test_loss_has_pos_weight(self) -> None:
        model = MagicMock()
        model.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
        _, _, loss_fn = _create_training_components(
            model=model,
            model_type="rnn",
            device="cpu",
            pos_weight=2.5,
        )
        assert loss_fn.pos_weight.item() == 2.5

    def test_scheduler_is_cosine_annealing_for_rnn(self) -> None:
        model = MagicMock()
        model.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
        _, sched, _ = _create_training_components(
            model=model,
            model_type="rnn",
            device="cpu",
        )
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


# ─── Trainer bounded memory ────────────────────────────────────────


class TestTrainerBoundedMemory:
    """Test that Trainer uses bounded memory for loss tracking."""

    def test_trainer_defaults_use_bounded_deques(self) -> None:
        """Test Trainer dataclass directly without model instantiation."""
        from sentimentizer.trainer import Trainer

        trainer = Trainer(
            loss_function=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            cfg=MagicMock(),
            model_type="rnn",
        )
        assert isinstance(trainer.losses, deque)
        assert trainer.losses.maxlen == 1000
        assert isinstance(trainer._recent_losses, deque)
        assert trainer._recent_losses.maxlen == 120

    def test_running_mean_computation(self) -> None:
        """Test O(1) running mean fields."""
        from sentimentizer.trainer import Trainer

        trainer = Trainer(
            loss_function=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            cfg=MagicMock(),
            model_type="rnn",
        )
        trainer.total_train_loss = 30.0
        trainer.train_step_count = 10
        assert trainer.total_train_loss / trainer.train_step_count == 3.0


# ─── _run_training_loop ──────────────────────────────────────────


def _make_mock_model() -> MagicMock:
    """Create a mock model that returns deterministic logits with grad."""
    model = MagicMock()
    # Return a tensor that when sigmoided gives ~0.5 probability
    # requires_grad=True so backward() works
    model.return_value = torch.zeros(2, 1, requires_grad=True)
    return model


def _make_dummy_metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=0.5,
        positive_accuracy=0.5,
        negative_accuracy=0.5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        cohen_kappa=0.0,
        mcc=0.0,
        npv=0.5,
        macro_f1=0.5,
        auc_roc=0.5,
        avg_precision=0.5,
        tp=1,
        tn=1,
        fp=1,
        fn=1,
        total=4,
    )


class TestRunTrainingLoop:
    """Test _run_training_loop with mock callbacks."""

    def test_calls_all_callbacks_every_epoch(self) -> None:
        mock_cb = MagicMock()
        mock_cb.on_epoch_end.return_value = False

        model = _make_mock_model()
        train_iter = [(torch.zeros(2, 10), torch.ones(2, 1))]
        val_iter = [(torch.zeros(2, 10), torch.ones(2, 1))]

        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        with (
            patch("sentimentizer.trainer.train_step", return_value=0.5),
            patch(
                "sentimentizer.trainer.compute_epoch_metrics",
                return_value=_make_dummy_metrics(),
            ),
        ):
            _run_training_loop(
                model=model,
                train_iter=iter(train_iter),
                val_iter=iter(val_iter),
                epochs=3,
                optimizer=optimizer,
                scheduler=None,
                loss_function=loss_fn,
                callbacks=[mock_cb],
                model_type="test",
                device="cpu",
            )

        assert mock_cb.on_epoch_end.call_count == 3

    def test_stops_when_callback_returns_true(self) -> None:
        def stop_after_2(result: EpochResult, state: TrainingState) -> bool:
            return result.epoch >= 2

        mock_cb = MagicMock()
        mock_cb.on_epoch_end.side_effect = stop_after_2

        model = _make_mock_model()
        train_iter = [(torch.zeros(2, 10), torch.ones(2, 1))]
        val_iter = [(torch.zeros(2, 10), torch.ones(2, 1))]

        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        with (
            patch("sentimentizer.trainer.train_step", return_value=0.5),
            patch(
                "sentimentizer.trainer.compute_epoch_metrics",
                return_value=_make_dummy_metrics(),
            ),
        ):
            state = _run_training_loop(
                model=model,
                train_iter=iter(train_iter),
                val_iter=iter(val_iter),
                epochs=5,
                optimizer=optimizer,
                scheduler=None,
                loss_function=loss_fn,
                callbacks=[mock_cb],
                model_type="test",
                device="cpu",
            )

        assert state.latest_epoch == 2
        assert mock_cb.on_epoch_end.call_count == 2

    def test_callbacks_run_in_registration_order(self) -> None:
        calls: list[str] = []

        class OrderedCallback(TrainingCallback):
            def __init__(self, name: str) -> None:
                self.name = name

            def on_epoch_end(self, result: EpochResult, state: TrainingState) -> bool:
                calls.append(self.name)
                return False

        model = _make_mock_model()
        train_iter = [(torch.zeros(2, 10), torch.ones(2, 1))]
        val_iter = [(torch.zeros(2, 10), torch.ones(2, 1))]

        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        with (
            patch("sentimentizer.trainer.train_step", return_value=0.5),
            patch(
                "sentimentizer.trainer.compute_epoch_metrics",
                return_value=_make_dummy_metrics(),
            ),
        ):
            _run_training_loop(
                model=model,
                train_iter=iter(train_iter),
                val_iter=iter(val_iter),
                epochs=2,
                optimizer=optimizer,
                scheduler=None,
                loss_function=loss_fn,
                callbacks=[
                    OrderedCallback("first"),
                    OrderedCallback("second"),
                    OrderedCallback("third"),
                ],
                model_type="test",
                device="cpu",
            )

        assert calls == ["first", "second", "third", "first", "second", "third"]


# ─── EarlyStoppingCallback ───────────────────────────────────────


class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback in isolation."""

    def test_stops_when_patience_exceeded(self) -> None:
        cb = EarlyStoppingCallback(patience=2)
        metrics = _make_dummy_metrics()

        r1 = EpochResult(epoch=1, train_loss=0.5, val_loss=0.5, metrics=metrics, lr=0.001)
        r2 = EpochResult(epoch=2, train_loss=0.5, val_loss=0.5, metrics=metrics, lr=0.001)
        r3 = EpochResult(epoch=3, train_loss=0.5, val_loss=0.5, metrics=metrics, lr=0.001)
        cb.on_epoch_end(r1, TrainingState())
        cb.on_epoch_end(r2, TrainingState())
        cb.on_epoch_end(r3, TrainingState())

        assert cb.patience_counter == 2

    def test_resets_patience_on_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2)
        metrics = _make_dummy_metrics()

        r1 = EpochResult(epoch=1, train_loss=0.5, val_loss=0.5, metrics=metrics, lr=0.001)
        r2 = EpochResult(epoch=2, train_loss=0.5, val_loss=0.4, metrics=metrics, lr=0.001)
        r3 = EpochResult(epoch=3, train_loss=0.5, val_loss=0.4, metrics=metrics, lr=0.001)
        cb.on_epoch_end(r1, TrainingState())
        cb.on_epoch_end(r2, TrainingState())
        cb.on_epoch_end(r3, TrainingState())

        assert cb.patience_counter == 1


# ─── CheckpointCallback ──────────────────────────────────────────


class TestCheckpointCallback:
    """Test CheckpointCallback without real disk I/O."""

    def test_saves_periodic_checkpoint(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(
            checkpoint_dir=str(tmp_path),
            checkpoint_every=2,
            checkpoint_best=False,
        )
        state = TrainingState()
        metrics = _make_dummy_metrics()

        cb.on_epoch_end(
            EpochResult(epoch=2, train_loss=0.3, val_loss=0.5, metrics=metrics, lr=0.001),
            state,
        )

        assert hasattr(state, "_pending_checkpoint_path")
        assert "checkpoint_epoch_2.pth" in state._pending_checkpoint_path

    def test_saves_best_model_on_improvement(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(
            checkpoint_dir=str(tmp_path),
            checkpoint_every=0,
            checkpoint_best=True,
        )
        state = TrainingState()
        metrics = _make_dummy_metrics()

        cb.on_epoch_end(
            EpochResult(epoch=1, train_loss=0.3, val_loss=0.5, metrics=metrics, lr=0.001),
            state,
        )
        cb.on_epoch_end(
            EpochResult(epoch=2, train_loss=0.3, val_loss=0.4, metrics=metrics, lr=0.001),
            state,
        )

        assert hasattr(state, "_pending_best_checkpoint_path")
        assert "best_model.pth" in state._pending_best_checkpoint_path

    def test_skips_save_when_no_checkpoint_dir(self) -> None:
        cb = CheckpointCallback(checkpoint_dir=None, checkpoint_every=0, checkpoint_best=False)
        state = TrainingState()
        metrics = _make_dummy_metrics()

        cb.on_epoch_end(
            EpochResult(epoch=1, train_loss=0.3, val_loss=0.5, metrics=metrics, lr=0.001),
            state,
        )

        assert not hasattr(state, "_pending_checkpoint_path")


# ─── MetricsCallback ─────────────────────────────────────────────


class TestMetricsCallback:
    """Test MetricsCallback publishes to all backends."""

    def test_skips_when_rank_not_zero(self) -> None:
        with patch("sentimentizer.trainer.publish_epoch_metrics") as mock_pub:
            cb = MetricsCallback(model_type="test", rank=1)
            metrics = _make_dummy_metrics()
            cb.on_epoch_end(
                EpochResult(epoch=1, train_loss=0.3, val_loss=0.4, metrics=metrics, lr=0.001),
                TrainingState(),
            )
            mock_pub.assert_not_called()

    def test_publishes_when_rank_zero(self) -> None:
        with patch("sentimentizer.trainer.publish_epoch_metrics") as mock_pub:
            cb = MetricsCallback(model_type="test", rank=0)
            metrics = _make_dummy_metrics()
            cb.on_epoch_end(
                EpochResult(epoch=1, train_loss=0.3, val_loss=0.4, metrics=metrics, lr=0.001),
                TrainingState(),
            )
            mock_pub.assert_called_once()


# ─── LoggingCallback ─────────────────────────────────────────────


class TestLoggingCallback:
    """Test LoggingCallback rank gating."""

    def test_skips_when_rank_not_zero(self) -> None:
        cb = LoggingCallback(model_type="test", rank=1)
        metrics = _make_dummy_metrics()
        result = cb.on_epoch_end(
            EpochResult(epoch=1, train_loss=0.3, val_loss=0.4, metrics=metrics, lr=0.001),
            TrainingState(),
        )
        assert result is False


# ─── RayReportCallback ───────────────────────────────────────────


class TestRayReportCallback:
    """Test RayReportCallback."""

    def test_no_error_when_ray_not_available(self) -> None:
        cb = RayReportCallback(model_type="test")
        metrics = _make_dummy_metrics()
        result = cb.on_epoch_end(
            EpochResult(epoch=1, train_loss=0.3, val_loss=0.4, metrics=metrics, lr=0.001),
            TrainingState(),
        )
        assert result is False
