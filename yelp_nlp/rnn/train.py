import logging
from typing import Callable, List
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import torch
from torch import optim
from torch.utils.data import DataLoader

from yelp_nlp.logging_utils import new_logger
from yelp_nlp.rnn.loader import CorpusDataset
from yelp_nlp.rnn.model import RNN
from yelp_nlp.rnn.config import TrainerConfig, OptimizationParams, SchedulerParams


logger = new_logger(logging.INFO)


def _new_loaders(
    train_data: CorpusDataset, val_data: CorpusDataset, cfg: TrainerConfig
):

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=cfg.memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=cfg.memory,
    )

    return train_loader, val_loader


@dataclass
class Trainer:

    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""

    loss_function: Callable
    optimizer: optim.Adam
    scheduler: optim.lr_scheduler._LRScheduler
    train_data: CorpusDataset
    val_data: CorpusDataset
    cfg: TrainerConfig
    losses: List[float] = field(default_factory=lambda: list())
    _mode: str = field(default="training")

    def _train_epoch(self, model: RNN, train_loader: DataLoader):

        self.train_data
        i = 0
        n = len(self.train_data)
        model.train()

        for j, (sent, target) in enumerate(train_loader):

            self.optimizer.zero_grad()

            # noqa: E501
            log_probs = model(sent.to(self.cfg.device))
            loss = self.loss_function(
                log_probs, target.to(self.cfg.device)
            )  # noqa: E501

            # gets graident
            loss.backward()

            # clips high gradients
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), max_norm=0.3, norm_type=2
            )

            # updates with new gradient
            self.optimizer.step()

            i += len(target)
            self.losses.append(loss.item())
            if i % (self.cfg.batch_size * 100) == 0:
                if self.scheduler:
                    self.scheduler.step()
                logger.info(
                    f"{i/n:.2f} of rows completed in {j + 1} cycles, current loss at {np.mean(self.losses[-60:]):.6f}"
                )  # noqa: E501
                logger.info(
                    f"current learning rate at {self.optimizer.param_groups[0]['lr']:.6f}"
                )  # noqa: E501

    def fit(self, model: RNN):
        train_loader, val_loader = _new_loaders(
            self.train_data, self.val_data, self.cfg
        )
        model.to(self.cfg.device)
        start = time.time()
        epoch_count = 0
        logger.info("fitting model...")

        for epoch in range(self.cfg.epochs):
            self._train_epoch(model, train_loader)
            self.eval(model, val_loader)
            epoch_count += 1
            if self.scheduler:
                self.scheduler.step()
            logger.info(f"epoch {epoch_count} completed")
        logger.info(
            f"model fitting completed, {time.time()-start:.0f} seconds passed"
        )  # noqa: E501

    def eval(self, model: RNN, val_loader: DataLoader):

        logger.info("evaluating predictions...")
        losses = []
        i = 0
        n = len(self.val_data)
        model.to(self.cfg.device)

        with torch.no_grad():
            model.eval()
            for j, (sent, target) in enumerate(val_loader):
                preds = model(sent.to(self.cfg.device))
                losses.append(
                    np.mean(
                        self.loss_function(preds, target.to(self.cfg.device)).item()
                    )
                )
                i += len(target)
                if i % (self.cfg.batch_size * 100) == 0:
                    logger.info(
                        f"{i/n:.2f} of rows completed in {j + 1} cycles, current loss at {np.mean(losses[-60:]):.6f}"
                    )  # noqa: E501
            self.val_loss = np.mean(losses)
            logger.info(f"validation loss at: {self.val_loss: .6f}")


def new_trainer(
    model: RNN,
    train_dataset: CorpusDataset,
    val_dataset: CorpusDataset,
    cfg: TrainerConfig,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OptimizationParams.lr,
        betas=OptimizationParams.betas,
        weight_decay=OptimizationParams.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SchedulerParams.T_max,
        eta_min=SchedulerParams.eta_min,
        last_epoch=SchedulerParams.last_epoch,
    )

    trainer = Trainer(
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=train_dataset,
        val_data=val_dataset,
        cfg=cfg,
    )

    return trainer
