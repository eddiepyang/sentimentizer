"""Tests for model checkpointing (save/load/resume).

Tests cover:
- save_checkpoint / load_checkpoint round-trip
- list_checkpoints / latest_checkpoint utilities
- TrainerConfig checkpoint fields
- Checkpoint directory creation during training
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sentimentizer.config import TrainerConfig
from sentimentizer.trainer import (
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)

# ─── Fixtures ────────────────────────────────────────────────────


class SimpleModel(nn.Module):
    """A tiny model for testing checkpointing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def model() -> SimpleModel:
    return SimpleModel()


@pytest.fixture
def optimizer(model: SimpleModel) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def scheduler(optimizer: torch.optim.Adam) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


@pytest.fixture
def checkpoint_dir() -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ─── Save / Load Tests ───────────────────────────────────────────


class TestSaveLoadCheckpoint:
    """Test checkpoint save and load round-trip."""

    def test_save_and_load_basic(
        self, model: SimpleModel, optimizer: torch.optim.Adam, checkpoint_dir: str
    ) -> None:
        """Saving and loading should preserve model weights."""
        path = os.path.join(checkpoint_dir, "test_ckpt.pth")
        save_checkpoint(model, optimizer, epoch=3, path=path)

        # Create a new model with different weights
        new_model = SimpleModel()
        original_weight = model.linear.weight.data.clone()
        new_model.linear.weight.data.fill_(0)

        checkpoint = load_checkpoint(path, new_model, device="cpu")

        # After loading, weights should match original
        assert torch.allclose(new_model.linear.weight.data, original_weight)
        assert checkpoint["epoch"] == 3

    def test_save_and_load_with_scheduler(
        self,
        model: SimpleModel,
        optimizer: torch.optim.Adam,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_dir: str,
    ) -> None:
        """Saving with scheduler should restore scheduler state."""
        path = os.path.join(checkpoint_dir, "test_ckpt_sched.pth")
        save_checkpoint(model, optimizer, epoch=5, path=path, scheduler=scheduler)

        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

        # Step the scheduler to change state
        scheduler.step()
        scheduler.step()

        # Save and reload
        save_checkpoint(model, optimizer, epoch=2, path=path, scheduler=scheduler)
        checkpoint = load_checkpoint(path, new_model, new_optimizer, new_scheduler, device="cpu")

        assert "scheduler_state_dict" in checkpoint
        assert checkpoint["epoch"] == 2

    def test_save_with_val_loss(
        self, model: SimpleModel, optimizer: torch.optim.Adam, checkpoint_dir: str
    ) -> None:
        """Saving with val_loss should include it in the checkpoint."""
        path = os.path.join(checkpoint_dir, "test_ckpt_loss.pth")
        save_checkpoint(model, optimizer, epoch=1, path=path, val_loss=0.456)

        new_model = SimpleModel()
        checkpoint = load_checkpoint(path, new_model, device="cpu")

        assert checkpoint["val_loss"] == pytest.approx(0.456)

    def test_save_creates_directory(
        self, model: SimpleModel, optimizer: torch.optim.Adam, checkpoint_dir: str
    ) -> None:
        """save_checkpoint should create parent directories."""
        nested_path = os.path.join(checkpoint_dir, "nested", "dir", "ckpt.pth")
        save_checkpoint(model, optimizer, epoch=1, path=nested_path)
        assert Path(nested_path).exists()

    def test_load_nonexistent_raises(self, model: SimpleModel, optimizer: torch.optim.Adam) -> None:
        """Loading from a nonexistent path should raise an error."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_checkpoint("/nonexistent/path/ckpt.pth", model, optimizer, device="cpu")


# ─── List / Latest Checkpoint Tests ────────────────────────────────


class TestListCheckpoints:
    """Test checkpoint listing utilities."""

    def test_list_checkpoints_empty_dir(self, checkpoint_dir: str) -> None:
        """list_checkpoints should return empty list for empty directory."""
        assert list_checkpoints(checkpoint_dir) == []

    def test_list_checkpoints_nonexistent_dir(self) -> None:
        """list_checkpoints should return empty list for nonexistent directory."""
        assert list_checkpoints("/nonexistent/dir") == []

    def test_list_checkpoints_finds_checkpoints(
        self, model: SimpleModel, optimizer: torch.optim.Adam, checkpoint_dir: str
    ) -> None:
        """list_checkpoints should find checkpoint files sorted by epoch."""
        for epoch in [1, 3, 5]:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch=epoch, path=path)

        checkpoints = list_checkpoints(checkpoint_dir)
        assert len(checkpoints) == 3
        assert checkpoints[0].name == "checkpoint_epoch_1.pth"
        assert checkpoints[2].name == "checkpoint_epoch_5.pth"

    def test_latest_checkpoint_finds_latest(
        self, model: SimpleModel, optimizer: torch.optim.Adam, checkpoint_dir: str
    ) -> None:
        """latest_checkpoint should return the highest-epoch checkpoint."""
        for epoch in [1, 2, 7]:
            path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch=epoch, path=path)

        latest = latest_checkpoint(checkpoint_dir)
        assert latest is not None
        assert latest.name == "checkpoint_epoch_7.pth"

    def test_latest_checkpoint_empty_dir(self, checkpoint_dir: str) -> None:
        """latest_checkpoint should return None for empty directory."""
        assert latest_checkpoint(checkpoint_dir) is None


# ─── TrainerConfig Checkpoint Fields ──────────────────────────────


class TestTrainerConfigCheckpoints:
    """Test TrainerConfig checkpoint fields."""

    def test_default_checkpoint_config(self) -> None:
        """Default config should have checkpointing disabled."""
        cfg = TrainerConfig()
        assert cfg.checkpoint_dir == ""
        assert cfg.checkpoint_every == 1
        assert cfg.checkpoint_best is True

    def test_custom_checkpoint_config(self) -> None:
        """Custom checkpoint dir should be settable."""
        cfg = TrainerConfig(checkpoint_dir="/tmp/ckpts", checkpoint_every=2)
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.checkpoint_every == 2

    def test_checkpoint_every_zero_disables_periodic(self) -> None:
        """Setting checkpoint_every=0 should disable periodic checkpoints."""
        cfg = TrainerConfig(checkpoint_dir="/tmp/ckpts", checkpoint_every=0)
        assert cfg.checkpoint_every == 0
