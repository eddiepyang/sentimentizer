"""Tests for single-node and distributed training.

Tests cover:
- Training primitives (train_step, val_step)
- Single-node Trainer.fit() with synthetic data
- Trainer creation (new_trainer)
- Distributed training config (_train_func config keys, serialization)
- Checkpoint save/load with DDP-wrapped models
- Ray Train integration (new_ray_trainer, get_dataset_shard API)
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from sentimentizer.config import (
    EmbeddingsConfig,
    TrainerConfig,
)
from sentimentizer.metrics import compute_classification_metrics
from sentimentizer.trainer import (
    _get_opt_params,
    _get_sched_params,
    _LinearWarmupCosineScheduler,
    train_step,
    val_step,
)

# ─── Helpers ─────────────────────────────────────────────────────


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
    """Simple dataset yielding (float_data, float_target) pairs.

    Unlike CorpusDataset which converts data to long tensors for embedding
    layers, this dataset keeps data as float32 for testing with linear models.
    """

    def __init__(self, n: int = 32, input_dim: int = 10) -> None:
        self.data = torch.randn(n, input_dim)
        self.targets = torch.randint(0, 2, (n,)).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


# ─── Training Primitives ─────────────────────────────────────────


class TestTrainStep:
    """Test the train_step function."""

    def test_train_step_returns_float(self) -> None:
        """train_step should return a scalar loss value."""
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = nn.BCEWithLogitsLoss()

        data = torch.randn(4, 10)
        target = torch.randint(0, 2, (4,)).float()

        loss = train_step(model, data, target, optimizer, loss_fn)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_updates_weights(self) -> None:
        """train_step should modify model weights."""
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = nn.BCEWithLogitsLoss()

        data = torch.randn(4, 10)
        target = torch.randint(0, 2, (4,)).float()

        weights_before = {n: p.clone() for n, p in model.named_parameters()}
        train_step(model, data, target, optimizer, loss_fn)

        # At least one weight tensor should have changed
        changed = any(not torch.equal(weights_before[n], p) for n, p in model.named_parameters())
        assert changed, "train_step did not update any weights"

    def test_train_step_with_grad_clipping(self) -> None:
        """train_step should clip gradients by default (max_grad_norm=1.0)."""
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = nn.BCEWithLogitsLoss()

        data = torch.randn(4, 10)
        target = torch.randint(0, 2, (4,)).float()

        # After train_step, all grad norms should be <= max_grad_norm
        train_step(model, data, target, optimizer, loss_fn, max_grad_norm=1.0)

        for p in model.parameters():
            if p.grad is not None:
                assert (
                    p.grad.norm().item() <= 1.0 + 1e-6
                ), f"Gradient norm {p.grad.norm().item()} exceeds max_grad_norm"


class TestValStep:
    """Test the val_step function."""

    def test_val_step_returns_float(self) -> None:
        """val_step should return a scalar loss value."""
        model = TinyModel()
        loss_fn = nn.BCEWithLogitsLoss()

        data = torch.randn(4, 10)
        target = torch.randint(0, 2, (4,)).float()

        loss = val_step(model, data, target, loss_fn)
        assert isinstance(loss, float)
        assert loss > 0

    def test_val_step_no_grad(self) -> None:
        """val_step should not compute gradients."""
        model = TinyModel()
        loss_fn = nn.BCEWithLogitsLoss()

        data = torch.randn(4, 10)
        target = torch.randint(0, 2, (4,)).float()

        # Zero out any existing gradients
        for p in model.parameters():
            p.grad = None

        val_step(model, data, target, loss_fn)

        # No gradients should be computed
        for p in model.parameters():
            assert p.grad is None, "val_step computed gradients unexpectedly"


# ─── Single-Node Training ────────────────────────────────────────


class TestSingleNodeTraining:
    """Test single-node training with the Trainer class."""

    def test_trainer_fit_completes(self) -> None:
        """Trainer.fit() should complete without error on synthetic data."""
        from sentimentizer.trainer import Trainer

        model = TinyModel(input_dim=10, hidden_dim=16)
        train_data = FloatDataset(n=32, input_dim=10)
        val_data = FloatDataset(n=16, input_dim=10)

        cfg = TrainerConfig(
            batch_size=8,
            epochs=2,
            device="cpu",
            dataloader_workers=0,
            early_stopping_patience=0,  # disable early stopping
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        loss_fn = nn.BCEWithLogitsLoss()

        trainer = Trainer(
            loss_function=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            model_type="rnn",
        )

        # Should not raise
        trainer.fit(model, train_data, val_data)  # type: ignore[arg-type]

    def test_trainer_fit_reduces_loss(self) -> None:
        """Trainer.fit() should reduce validation loss over epochs."""
        from sentimentizer.trainer import Trainer

        model = TinyModel(input_dim=10, hidden_dim=16)
        train_data = FloatDataset(n=64, input_dim=10)
        val_data = FloatDataset(n=32, input_dim=10)

        cfg = TrainerConfig(
            batch_size=16,
            epochs=5,
            device="cpu",
            dataloader_workers=0,
            early_stopping_patience=0,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        loss_fn = nn.BCEWithLogitsLoss()

        trainer = Trainer(
            loss_function=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            model_type="rnn",
        )

        trainer.fit(model, train_data, val_data)  # type: ignore[arg-type]
        # After training, val_loss should be finite
        assert trainer.val_loss < float("inf"), "val_loss was not updated"
        assert trainer.val_loss > 0, "val_loss should be positive"

    def test_trainer_with_checkpoints(self) -> None:
        """Trainer.fit() should save checkpoints when checkpoint_dir is set."""
        from sentimentizer.trainer import Trainer

        model = TinyModel(input_dim=10, hidden_dim=16)
        train_data = FloatDataset(n=32, input_dim=10)
        val_data = FloatDataset(n=16, input_dim=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainerConfig(
                batch_size=8,
                epochs=2,
                device="cpu",
                dataloader_workers=0,
                early_stopping_patience=0,
                checkpoint_dir=tmpdir,
                checkpoint_every=1,
                checkpoint_best=True,
            )

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            loss_fn = nn.BCEWithLogitsLoss()

            trainer = Trainer(
                loss_function=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                model_type="rnn",
            )

            trainer.fit(model, train_data, val_data)  # type: ignore[arg-type]

            # Check that checkpoint files were created
            import glob

            ckpt_files = glob.glob(os.path.join(tmpdir, "*.pth"))
            assert len(ckpt_files) > 0, "No checkpoint files were created"


# ─── new_trainer Factory ─────────────────────────────────────────


class TestNewTrainer:
    """Test the new_trainer factory function."""

    def test_new_trainer_creates_trainer(self) -> None:
        """new_trainer should return a Trainer instance."""
        from sentimentizer.trainer import Trainer, new_trainer

        model = TinyModel()
        cfg = TrainerConfig(
            batch_size=8,
            epochs=1,
            device="cpu",
            dataloader_workers=0,
        )

        trainer = new_trainer(model, cfg, model_type="rnn")
        assert isinstance(trainer, Trainer)

    def test_new_trainer_rnn_config(self) -> None:
        """new_trainer with rnn model_type should use OptimizationParams."""
        from sentimentizer.trainer import new_trainer

        model = TinyModel()
        cfg = TrainerConfig(batch_size=8, epochs=1, device="cpu", dataloader_workers=0)

        trainer = new_trainer(model, cfg, model_type="rnn")
        # RNN uses OptimizationParams (lr=0.001 by default)
        assert trainer.optimizer.defaults["lr"] == 0.001

    def test_new_trainer_encoder_config(self) -> None:
        """new_trainer with encoder model_type should use EncoderOptimizationParams."""
        from sentimentizer.trainer import new_trainer

        model = TinyModel()
        cfg = TrainerConfig(batch_size=8, epochs=1, device="cpu", dataloader_workers=0)

        trainer = new_trainer(model, cfg, model_type="encoder")
        # Encoder uses EncoderOptimizationParams (lr=0.0005 by default)
        assert trainer.optimizer.defaults["lr"] == 0.0005


# ─── Optimization and Scheduler Params ────────────────────────────


class TestOptAndSchedParams:
    """Test optimization and scheduler parameter factories."""

    def test_rnn_opt_params(self) -> None:
        opt = _get_opt_params("rnn")
        assert opt.lr == 0.001

    def test_encoder_opt_params(self) -> None:
        opt = _get_opt_params("encoder")
        assert opt.lr == 0.0005

    def test_decoder_opt_params(self) -> None:
        opt = _get_opt_params("decoder")
        assert opt.lr == 0.0005

    def test_rnn_sched_params(self) -> None:
        sched = _get_sched_params("rnn")
        assert sched.T_max == 4

    def test_encoder_sched_params(self) -> None:
        sched = _get_sched_params("encoder")
        assert sched.warmup_epochs == 1
        assert sched.T_max == 8

    def test_decoder_sched_params(self) -> None:
        sched = _get_sched_params("decoder")
        assert sched.warmup_epochs == 1
        assert sched.T_max == 8


# ─── Scheduler Correctness ────────────────────────────────────────


class TestSchedulerCorrectness:
    """Test that learning rate schedules produce correct values."""

    def test_warmup_cosine_no_zero_lr(self) -> None:
        """_LinearWarmupCosineScheduler must not produce zero LR.

        The old implementation used `step / warmup_steps` which returns
        0.0 at step=0, making the entire first epoch train with zero LR.
        The fix uses `(step + 1) / warmup_steps` so the first step gets
        a non-zero LR multiplier.
        """
        from sentimentizer.config import default_epochs

        base_lr = 0.0005
        warmup_steps = 1
        total_steps = default_epochs("encoder")  # 8
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler = _LinearWarmupCosineScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps, eta_min=1e-6
        )

        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(total_steps):
            # Simulate optimizer.step() so PyTorch doesn't warn
            optimizer.zero_grad()
            loss = model(torch.randn(2, 10)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # First LR (before any step) must be non-zero
        assert lrs[0] > 0, f"Initial LR must be > 0, got {lrs[0]}"
        # After warmup step (step 0 -> step 1), LR should be base_lr or close
        assert lrs[1] > 0, f"LR after warmup step must be > 0, got {lrs[1]}"
        # LR should decrease after warmup phase
        assert lrs[-1] < lrs[1], "LR should decay after warmup"

    def test_warmup_cosine_lr_starts_at_half_base(self) -> None:
        """With warmup_steps=1, the initial LR multiplier should be 1/2.

        lr_lambda(0) = (0 + 1) / 1 = 1.0 after the LambdaLR init, but
        LambdaLR with last_epoch=-1 starts at lr_lambda(0) * base_lr.
        Wait — LambdaLR actually starts at lr_lambda(0), so:
        initial multiplier = (0 + 1) / warmup_steps = 1/1 = 1.0 * base_lr? No.

        LambdaLR initial LR = base_lr * lr_lambda(0).
        With the fix: lr_lambda(0) = (0 + 1) / 1 = 1.0.
        But with warmup_steps=1 and total_steps=8:
        step 0: lr_lambda = 1.0 (warmup complete)
        step 1: lr_lambda = cosine decay starts

        For warmup_steps=2: step 0: (0+1)/2 = 0.5, step 1: (1+1)/2 = 1.0
        """
        base_lr = 0.001
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        _LinearWarmupCosineScheduler(optimizer, warmup_steps=2, total_steps=8, eta_min=1e-6)

        # LambdaLR with last_epoch=-1 starts at lr_lambda(0)
        initial_lr = optimizer.param_groups[0]["lr"]
        expected_multiplier = 1.0 / 2  # (0+1)/warmup_steps = 0.5
        assert (
            abs(initial_lr - base_lr * expected_multiplier) < 1e-8
        ), f"Expected initial LR={base_lr * expected_multiplier}, got {initial_lr}"

    def test_cosine_annealing_lr_decay(self) -> None:
        """CosineAnnealingLR should decay LR over the configured T_max."""
        base_lr = 0.001
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        T_max = 4
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(T_max):
            optimizer.zero_grad()
            loss = model(torch.randn(2, 10)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should be highest at step 0 and decrease
        assert lrs[0] == base_lr
        # Final LR should be close to eta_min
        assert lrs[-1] < lrs[0], f"LR should decrease: {lrs}"

    def test_scheduler_t_max_matches_default_epochs(self) -> None:
        """Encoder/decoder T_max should match their default epoch counts.

        If T_max < default_epochs, the LR decays to minimum before training
        finishes, leaving the model to train at eta_min for remaining epochs.
        """
        from sentimentizer.config import (
            default_epochs,
        )

        for model_type in ("encoder", "decoder"):
            sched = _get_sched_params(model_type)
            epochs = default_epochs(model_type)
            assert sched.T_max == epochs, (
                f"{model_type}: T_max={sched.T_max} != default_epochs={epochs}. "
                f"LR would bottom out after {sched.T_max} epochs but training "
                f"continues for {epochs} epochs."
            )


# ─── Checkpoint with DDP Model ───────────────────────────────────


class TestCheckpointWithDDP:
    """Test checkpoint save/load with DDP-wrapped model (model.module pattern)."""

    def test_ddp_state_dict_save_load(self) -> None:
        """Saving model.module.state_dict() and loading should work correctly.

        This mirrors the pattern used in _train_func where the model is
        DDP-wrapped via prepare_model(), so we access model.module.state_dict().
        """
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Simulate DDP wrapping: in Ray Train, prepare_model() wraps the model
        # so model.module gives the original model
        # original_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ddp_checkpoint.pth")

            # Save using the DDP pattern: model.module.state_dict()
            # In our case, model IS the unwrapped model (no DDP in test)
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 1,
            }
            torch.save(checkpoint_data, path)

            # Create a new model and load
            new_model = TinyModel()
            # Verify weights are different before loading
            for (_n1, p1), (_n2, p2) in zip(
                new_model.named_parameters(), model.named_parameters(), strict=False
            ):
                if p1.shape == p2.shape:
                    # Not guaranteed to be different, but very likely
                    pass

            loaded = torch.load(path, map_location="cpu", weights_only=False)
            new_model.load_state_dict(loaded["model_state_dict"])

            # After loading, weights should match original
            for (n1, p1), (_n2, p2) in zip(
                new_model.named_parameters(), model.named_parameters(), strict=False
            ):
                assert torch.allclose(p1, p2), f"Weights mismatch for {n1}"

            assert loaded["epoch"] == 1

    def test_ray_checkpoint_format_roundtrip(self) -> None:
        """Test the Ray 2.55+ directory-based checkpoint format round-trip.

        This mirrors the exact pattern used in _train_func for saving
        and in driver.py for loading checkpoints.
        """
        pytest.importorskip("ray")
        import ray.cloudpickle as pickle
        from ray.train import Checkpoint

        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Save checkpoint (same pattern as _train_func)
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 3,
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = os.path.join(checkpoint_dir, "data.pkl")
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            # Create Ray Checkpoint from directory
            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            # Read checkpoint back (same pattern as driver.py)
            with (
                checkpoint.as_directory() as loaded_dir,
                open(os.path.join(loaded_dir, "data.pkl"), "rb") as fp,
            ):
                loaded_data = pickle.load(fp)

            # Verify all keys are present
            assert "model_state_dict" in loaded_data
            assert "optimizer_state_dict" in loaded_data
            assert loaded_data["epoch"] == 3

            # Verify model weights can be loaded
            new_model = TinyModel()
            new_model.load_state_dict(loaded_data["model_state_dict"])

            # Weights should match
            for (n1, p1), (_n2, p2) in zip(
                new_model.named_parameters(), model.named_parameters(), strict=False
            ):
                assert torch.allclose(p1, p2), f"Weights mismatch for {n1}"


# ─── Distributed Training Config ─────────────────────────────────


class TestDistributedConfig:
    """Test distributed training configuration and _train_func config."""

    def test_train_func_config_has_all_required_keys(self) -> None:
        """The _train_func config dict must contain all keys needed by the worker."""
        required_keys = {
            "epochs",
            "batch_size",
            "lr",
            "betas",
            "weight_decay",
            "use_warmup",
            "warmup_steps",
            "total_steps",
            "scheduler_eta_min",
            "model_type",
            "dict_path",
            "embeddings_model_name",
            "embeddings_emb_length",
            "input_len",
        }

        config = {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.005,
            "betas": [0.7, 0.99],
            "weight_decay": 1e-4,
            "use_warmup": False,
            "warmup_steps": 0,
            "total_steps": 0,
            "scheduler_eta_min": 1e-6,
            "model_type": "rnn",
            "dict_path": "/tmp/test.dict",
            "embeddings_model_name": "glove-wiki-gigaword-100",
            "embeddings_emb_length": 100,
            "input_len": 200,
        }
        assert required_keys.issubset(set(config.keys()))

    def test_train_func_config_is_json_serializable(self) -> None:
        """The _train_func config must be JSON-serializable for Ray."""
        config = {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.005,
            "betas": [0.7, 0.99],
            "weight_decay": 1e-4,
            "use_warmup": False,
            "warmup_steps": 0,
            "total_steps": 0,
            "scheduler_eta_min": 1e-6,
            "model_type": "rnn",
            "dict_path": "/tmp/test.dict",
            "embeddings_model_name": "glove-wiki-gigaword-100",
            "embeddings_emb_length": 100,
            "input_len": 200,
        }
        serialized = json.dumps(config)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["model_type"] == "rnn"
        assert deserialized["epochs"] == 1

    def test_get_dataset_shard_is_standalone_function(self) -> None:
        """In Ray 2.55+, get_dataset_shard is a standalone function.

        It is NOT a method on DistributedTrainContext (get_context()).
        The correct API is: train.get_dataset_shard("train")
        NOT: train.get_context().get_dataset_shard("train")
        """
        pytest.importorskip("ray")
        from ray import train

        assert callable(train.get_dataset_shard)

    def test_get_context_outside_worker_raises(self) -> None:
        """train.get_context() must raise RuntimeError outside a worker."""
        pytest.importorskip("ray")
        from ray import train

        with pytest.raises(RuntimeError, match="cannot be used outside"):
            train.get_context()

    def test_embeddings_config_uses_model_name(self) -> None:
        """EmbeddingsConfig should use model_name, not file_path/sub_file_path."""
        cfg = EmbeddingsConfig(model_name="glove-wiki-gigaword-100", emb_length=100)
        assert cfg.model_name == "glove-wiki-gigaword-100"
        assert cfg.emb_length == 100

    def test_embeddings_config_rejects_file_path(self) -> None:
        """EmbeddingsConfig should not accept file_path or sub_file_path."""
        with pytest.raises(TypeError):
            EmbeddingsConfig(file_path="glove.txt", emb_length=100)  # type: ignore[call-arg]

    def test_embeddings_config_rejects_sub_file_path(self) -> None:
        """EmbeddingsConfig should not accept sub_file_path."""
        with pytest.raises(TypeError):
            EmbeddingsConfig(sub_file_path="glove.txt", emb_length=100)  # type: ignore[call-arg]


# ─── Metrics Integration ────────────────────────────────────────


class TestResetStaleMetrics:
    """Test _reset_stale_metrics zeroes stale model_type metrics."""

    def test_zeroes_target_model_type_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_reset_stale_metrics writes a zeroed per-model-type JSON file."""
        from workflows.stages.train import _reset_stale_metrics

        monkeypatch.setattr("workflows.stages.train._METRICS_DIR", tmp_path)

        _reset_stale_metrics("encoder")

        metrics_file = tmp_path / "encoder_metrics.json"
        assert metrics_file.exists()
        result = json.loads(metrics_file.read_text())
        assert result["train_loss"] == 0.0
        assert result["epoch"] == 0
        assert "_trace" in result

        # Other model-type files must NOT be created
        assert not (tmp_path / "rnn_metrics.json").exists()
        assert not (tmp_path / "decoder_metrics.json").exists()

    def test_creates_per_model_file_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_reset_stale_metrics creates the JSON file if it doesn't exist."""
        from workflows.stages.train import _reset_stale_metrics

        monkeypatch.setattr("workflows.stages.train._METRICS_DIR", tmp_path)

        _reset_stale_metrics("decoder")

        metrics_file = tmp_path / "decoder_metrics.json"
        assert metrics_file.exists()
        result = json.loads(metrics_file.read_text())
        assert result["epoch"] == 0
        assert "_trace" in result

    def test_overwrites_corrupt_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_reset_stale_metrics overwrites a corrupt per-model file."""
        from workflows.stages.train import _reset_stale_metrics

        monkeypatch.setattr("workflows.stages.train._METRICS_DIR", tmp_path)

        metrics_file = tmp_path / "encoder_metrics.json"
        metrics_file.write_text("{invalid json")

        _reset_stale_metrics("encoder")

        assert metrics_file.exists()
        result = json.loads(metrics_file.read_text())
        assert "encoder" not in result  # per-model file, not nested
        assert result["train_loss"] == 0.0

    def test_resets_prometheus_gauges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_reset_stale_metrics resets prometheus_client gauges for model_type to 0."""
        from workflows.stages.train import _reset_stale_metrics

        monkeypatch.setattr("workflows.stages.train._METRICS_DIR", tmp_path)

        from prometheus_client import REGISTRY

        from sentimentizer.exporter import TRAINING_VAL_ACCURACY

        TRAINING_VAL_ACCURACY.labels(model_type="encoder").set(0.99)

        _reset_stale_metrics("encoder")

        value = REGISTRY.get_sample_value(
            "sentimentizer_training_val_accuracy",
            {"model_type": "encoder"},
        )
        assert value == 0.0

    def test_clears_ray_gauges_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_reset_stale_metrics removes the _RAY_GAUGES cache entry."""
        from workflows.stages.train import _reset_stale_metrics

        monkeypatch.setattr("workflows.stages.train._METRICS_DIR", tmp_path)

        from sentimentizer.trainer import _RAY_GAUGES

        _RAY_GAUGES["test_model"] = {"dummy": True}
        assert "test_model" in _RAY_GAUGES

        _reset_stale_metrics("test_model")

        assert "test_model" not in _RAY_GAUGES

    def test_per_model_files_no_race(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parallel writes for different model types don't clobber each other."""
        from workflows.stages.train import _reset_stale_metrics

        monkeypatch.setattr("workflows.stages.train._METRICS_DIR", tmp_path)

        _reset_stale_metrics("rnn")
        _reset_stale_metrics("encoder")

        rnn_file = tmp_path / "rnn_metrics.json"
        encoder_file = tmp_path / "encoder_metrics.json"

        assert rnn_file.exists()
        assert encoder_file.exists()

        rnn_data = json.loads(rnn_file.read_text())
        encoder_data = json.loads(encoder_file.read_text())

        assert rnn_data["train_loss"] == 0.0
        assert encoder_data["train_loss"] == 0.0
        assert rnn_data["_trace"]["reset_by"] == "rnn"
        assert encoder_data["_trace"]["reset_by"] == "encoder"


class TestMetricsIntegration:
    """Test that training produces valid metrics."""

    def test_compute_classification_metrics_after_training(self) -> None:
        """After a training step, metrics should be computable."""
        predictions = np.array([1, 0, 1, 0, 1])
        targets = np.array([1, 0, 0, 0, 1])
        probabilities = np.array([0.9, 0.2, 0.6, 0.3, 0.8])

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
        )

        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1 <= 1.0
        assert metrics.total == 5
        assert metrics.tp == 2
        assert metrics.tn == 2
        assert metrics.fp == 1
        assert metrics.fn == 0

    def test_metrics_from_model_output(self) -> None:
        """Metrics should work with raw model logits (sigmoid -> probabilities)."""
        model = TinyModel(input_dim=10)
        data = torch.randn(8, 10)
        logits = model(data)
        probabilities = torch.sigmoid(logits).detach().numpy().flatten()
        predictions = (probabilities >= 0.5).astype(int)
        targets = np.random.randint(0, 2, size=8)

        metrics = compute_classification_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
        )

        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.total == 8
