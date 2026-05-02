import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import ray
import torch
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_model
from torch import optim
from torch.utils.data import DataLoader

from sentimentizer import new_logger
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    DriverConfig,
    EncoderOptimizationParams,
    EncoderSchedulerParams,
    OptimizationParams,
    SchedulerParams,
    TrainerConfig,
    default_dataloader_workers,
    default_epochs,
)
from sentimentizer.loader import CorpusDataset

logger = new_logger(DEFAULT_LOG_LEVEL)


def _new_loaders(
    train_data: CorpusDataset, val_data: CorpusDataset, cfg: TrainerConfig
) -> tuple[DataLoader, DataLoader]:
    # Resolve auto-detect dataloader_workers (-1 means auto)
    workers = cfg.dataloader_workers
    if workers == -1:
        workers = default_dataloader_workers(cfg.device)

    pin_mem = cfg.memory if cfg.device != "mps" else False

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=pin_mem,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader


@dataclass
class Trainer:
    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""

    loss_function: Callable
    optimizer: optim.Adam
    scheduler: optim.lr_scheduler.LRScheduler
    cfg: TrainerConfig
    losses: list[float] = field(default_factory=lambda: list())
    _mode: str = field(default="training")

    def _train_epoch(self, model: torch.nn.Module, train_loader: DataLoader) -> None:
        i = 0
        n = len(train_loader.dataset)  # type: ignore[arg-type]
        model.train()

        for j, (sent, target) in enumerate(train_loader):
            self.optimizer.zero_grad()

            log_probs = model(sent.to(self.cfg.device))
            loss = self.loss_function(log_probs, target.to(self.cfg.device))

            # gets gradient
            loss.backward()

            # clips high gradients
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)

            # updates with new gradient
            self.optimizer.step()

            i += len(target)
            self.losses.append(loss.item())
            if i % (self.cfg.batch_size * 100) == 0:
                logger.info(
                    f"{i/n:.2f} of rows completed in "
                    f"{j + 1} cycles, current loss at {np.mean(self.losses[-60:]):.6f}"
                )
                logger.info(f"current learning rate at {self.optimizer.param_groups[0]['lr']:.6f}")

    def fit(
        self, model: torch.nn.Module, train_data: CorpusDataset, val_data: CorpusDataset
    ) -> None:
        train_loader, val_loader = _new_loaders(train_data, val_data, self.cfg)
        model.to(self.cfg.device)
        start = time.time()

        # Resolve default epochs if auto (-1)
        epochs = self.cfg.epochs
        if epochs == -1:
            epochs = default_epochs("rnn")  # fallback; driver resolves this

        logger.info("fitting model...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self._train_epoch(model, train_loader)
            self.eval(model, val_loader)

            if self.scheduler:
                self.scheduler.step()

            # Early stopping based on validation loss
            if self.cfg.early_stopping_patience > 0:
                if self.val_loss < best_val_loss:
                    best_val_loss = self.val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.early_stopping_patience:
                        logger.info(
                            f"early stopping at epoch {epoch}, "
                            f"val_loss hasn't improved for {patience_counter} epochs"
                        )
                        break

            logger.info(f"epoch {epoch} completed")
        logger.info(f"model fitting completed, {time.time()-start:.0f} seconds passed")

    def eval(self, model: torch.nn.Module, val_loader: DataLoader) -> None:
        logger.info("evaluating predictions...")
        losses = []
        i = 0
        n = len(val_loader.dataset)  # type: ignore[arg-type]
        model.to(self.cfg.device)

        with torch.no_grad():
            model.eval()
            for j, (sent, target) in enumerate(val_loader):
                preds = model(sent.to(self.cfg.device))
                losses.append(self.loss_function(preds, target.to(self.cfg.device)).item())
                i += len(target)
                if i % (self.cfg.batch_size * 100) == 0:
                    logger.info(
                        f"{i/n:.2f} of rows completed in "
                        f"{j + 1} cycles, validation loss at {np.mean(losses[-60:]):.6f}"
                    )
            self.val_loss = np.mean(losses)
            logger.info(f"validation loss at: {self.val_loss: .6f}")


def _get_opt_params(model_type: str) -> OptimizationParams | EncoderOptimizationParams:
    """Return optimization params appropriate for the model type."""
    if model_type in ("encoder", "decoder"):
        return EncoderOptimizationParams()
    return OptimizationParams()


def _get_sched_params(model_type: str) -> SchedulerParams | EncoderSchedulerParams:
    """Return scheduler params appropriate for the model type."""
    if model_type in ("encoder", "decoder"):
        return EncoderSchedulerParams()
    return SchedulerParams()


class _LinearWarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay.

    During warmup, LR increases linearly from 0 to base_lr.
    After warmup, follows CosineAnnealing decay to eta_min.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-6,
    ) -> None:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return eta_min + (1.0 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


def new_trainer(
    model: torch.nn.Module,
    cfg: TrainerConfig,
    model_type: str = "rnn",
) -> Trainer:
    opt = _get_opt_params(model_type)
    sched = _get_sched_params(model_type)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        betas=opt.betas,
        weight_decay=opt.weight_decay,
    )

    # Use warmup+cosine for transformer models, simple cosine for RNN
    if isinstance(sched, EncoderSchedulerParams) and sched.warmup_epochs > 0:
        warmup_steps = sched.warmup_epochs * cfg.epochs  # approximate
        total_steps = sched.T_max * cfg.epochs
        scheduler = _LinearWarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=sched.eta_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched.T_max,
            eta_min=sched.eta_min,
            last_epoch=sched.last_epoch,
        )

    trainer = Trainer(
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )

    return trainer


# ──────────────────────────────────────────────
# Ray Train distributed training
# ──────────────────────────────────────────────


def _train_func(config: dict) -> None:
    """Training function executed on each Ray Train worker.

    Creates the model on the worker, wraps it with DDP via prepare_model(),
    and runs the training loop with dataset shards from Ray Data.
    Reports metrics and checkpoints back to Ray Train.
    """

    # Unpack training config
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    betas = tuple(config["betas"])
    weight_decay = config["weight_decay"]
    use_warmup = config.get("use_warmup", False)
    warmup_steps = config.get("warmup_steps", 0)
    total_steps = config.get("total_steps", 0)
    scheduler_eta_min = config.get("scheduler_eta_min", 1e-6)

    # Unpack model config
    model_type = config["model_type"]
    dict_path = config["dict_path"]
    embeddings_file_path = config["embeddings_file_path"]
    embeddings_sub_file_path = config["embeddings_sub_file_path"]
    embeddings_emb_length = config["embeddings_emb_length"]
    input_len = config["input_len"]

    # Create model on this worker
    from sentimentizer.config import EmbeddingsConfig

    embeddings_config = EmbeddingsConfig(
        file_path=embeddings_file_path,
        sub_file_path=embeddings_sub_file_path,
        emb_length=embeddings_emb_length,
    )

    if model_type == "rnn":
        from sentimentizer.models.rnn import new_model
    elif model_type == "encoder":
        from sentimentizer.models.encoder import new_model
    elif model_type == "decoder":
        from sentimentizer.models.decoder import new_model
    else:
        raise ValueError(f"no matching model for {model_type}")

    model = new_model(
        dict_path=dict_path,
        embeddings_config=embeddings_config,
        batch_size=batch_size,
        input_len=input_len,
    )

    # Prepare model for distributed training (DDP) and move to correct device
    model = prepare_model(model)

    # Determine device from model (set by prepare_model)
    device = next(model.parameters()).device

    # Set up optimizer (AdamW for all models)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    # Set up scheduler (warmup+cosine for transformers, cosine for RNN)
    if use_warmup and warmup_steps > 0:
        scheduler = _LinearWarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=scheduler_eta_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_eta_min,
        )
    loss_function = torch.nn.BCEWithLogitsLoss()

    # Get dataset shards for this worker
    train_shard = train.get_context().get_dataset_shard("train")
    val_shard = train.get_context().get_dataset_shard("val")

    start = time.time()
    logger.info("fitting model (distributed)...")

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in train_shard.iter_torch_batches(batch_size=batch_size):
            data = batch["data"].long().to(device)
            target = batch["target"].float().to(device)

            optimizer.zero_grad()
            log_probs = model(data)
            loss = loss_function(log_probs, target)
            loss.backward()

            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            epoch_losses.append(loss.item())

        if scheduler:
            scheduler.step()

        # Validation
        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_shard.iter_torch_batches(batch_size=batch_size):
                data = batch["data"].long().to(device)
                target = batch["target"].float().to(device)
                preds = model(data)
                val_losses.append(loss_function(preds, target).item())

        val_loss = np.mean(val_losses) if val_losses else 0.0
        train_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        logger.info(
            f"epoch {epoch} completed, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        # Report metrics and checkpoint to Ray Train
        train.report(
            {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
            checkpoint=Checkpoint.from_dict(
                {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                }
            ),
        )

    logger.info(f"model fitting completed, {time.time()-start:.0f} seconds passed")


def new_ray_trainer(
    train_ds: ray.data.Dataset,
    val_ds: ray.data.Dataset,
    cfg: TrainerConfig,
    model_type: str = "rnn",
    driver_config: type[DriverConfig] = DriverConfig,
) -> TorchTrainer:
    """Factory function to create a Ray Train TorchTrainer for distributed training.

    Args:
        train_ds: Ray Dataset for training
        val_ds: Ray Dataset for validation
        cfg: TrainerConfig with training hyperparameters
        model_type: Model type ("rnn", "encoder", or "decoder")
        driver_config: DriverConfig class with file paths and model config

    Returns:
        Configured TorchTrainer ready to call .fit()
    """

    # Get model-specific optimization and scheduler params
    opt = _get_opt_params(model_type)
    sched = _get_sched_params(model_type)

    use_warmup = isinstance(sched, EncoderSchedulerParams) and sched.warmup_epochs > 0
    if isinstance(sched, EncoderSchedulerParams):
        warmup_steps = sched.warmup_epochs * cfg.epochs
        total_steps = sched.T_max * cfg.epochs
    else:
        warmup_steps = 0
        total_steps = 0

    # Build train_loop_config with everything the worker needs to
    # create the model and run training (models cannot be passed across workers)
    train_loop_config = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": opt.lr,
        "betas": list(opt.betas),
        "weight_decay": opt.weight_decay,
        "use_warmup": use_warmup,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "scheduler_eta_min": sched.eta_min,
        "model_type": model_type,
        "dict_path": driver_config.files.dictionary_file_path,
        "embeddings_file_path": driver_config.embeddings.file_path,
        "embeddings_sub_file_path": driver_config.embeddings.sub_file_path,
        "embeddings_emb_length": driver_config.embeddings.emb_length,
        "input_len": driver_config.tokenizer.max_len,
    }

    use_gpu = cfg.device in ("cuda",)

    trainer = TorchTrainer(
        train_loop_per_worker=_train_func,
        train_loop_config=train_loop_config,
        scaling_config=ScalingConfig(
            num_workers=cfg.ray_workers,
            use_gpu=use_gpu,
        ),
        datasets={"train": train_ds, "val": val_ds},
    )

    return trainer
