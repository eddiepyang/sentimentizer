import pytest

# ruff: noqa: E402
ray = pytest.importorskip("ray")

from sentimentizer.config import TrainerConfig
from sentimentizer.loader import load_train_val_ray_datasets
from sentimentizer.trainer import new_ray_trainer


class TestModernBERTRay:
    """Ray-specific training integration checks for ModernBERT."""

    def test_new_ray_trainer_creation(self, relative_root) -> None:
        """Verify new_ray_trainer can successfully construct a TorchTrainer with modernbert."""
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            trainer = new_ray_trainer(
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=cfg,
                model_type="modernbert",
            )
            from ray.train.torch import TorchTrainer

            assert isinstance(trainer, TorchTrainer)
            assert trainer.train_loop_config["model_type"] == "modernbert"
            assert trainer.train_loop_config["lr"] == 2e-5  # ModernBERT default LR
        except Exception:
            pytest.skip("test parquet data not available")
