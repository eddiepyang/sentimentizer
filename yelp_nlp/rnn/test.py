import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass

import time
import numpy as np
from data import id_to_glove, CorpusData
from model import RNN


@dataclass
class Trainer:
    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""

    optimizer: list
    dataclass: list
    batch_size: int
    epochs: int
    workers: int
    device: str

    def train_epoch(self, model):

        i = 0
        n = len(self.dataclass)

        for j, (sent, target) in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            sent, labels = sent.long().to(self.device), target.float().to(self.device)
            log_probs = model(sent)
            loss = self.loss_function(log_probs, labels.to(device))

            # gets graident
            loss.backward()

            # clips high gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.3, norm_type=2
            )

            # updates with new gradient
            self.optimizer.step()

            i += len(labels)
            self.losses.append(loss.item())
            if i % (batch_size * 100) == 0:
                print(
                    f"""{i/n:.2f} of rows completed in {j+1} cycles, current loss at {np.mean(self.losses[-30:]):.4f}"""
                )  # noqa: E501

    def fit(self, model):

        model.to(self.device)
        start = time.time()
        model.train()

        epoch_count = 0
        self.losses = []

        print("fitting model...")

        for epoch in range(self.epochs):

            self.train_epoch(model)

            epoch_count += 1
            print(f"epoch {epoch_count} completed")
        print(f"model fitting completed, {time.time()-start:.0f} seconds passed")
        return self
