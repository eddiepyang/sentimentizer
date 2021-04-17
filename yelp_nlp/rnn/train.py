import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import os
from dataclasses import dataclass
import time
import numpy as np

from data import id_to_glove, CorpusData
from model import RNN, RNN2


@dataclass
class Trainer:
    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""
    loss_function: object
    optimizer: optim.Adam
    scheduler: optim.lr_scheduler
    dataclass: CorpusData
    batch_size: int
    epochs: int
    workers: int
    device: str

    def train_epoch(self, model):

        i = 0
        n = len(self.dataclass)

        for j, (sent, target) in enumerate(self.train_loader):

            self.optimizer.zero_grad()
            sent, labels = sent.long().to(self.device), target.float().to(self.device)  # noqa: E501
            log_probs = model(sent)
            loss = self.loss_function(log_probs, labels.to(self.device))

            # gets graident
            loss.backward()

            # clips high gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.3,
                norm_type=2
            )

            # updates with new gradient
            self.optimizer.step()

            i += len(labels)
            self.losses.append(loss.item())
            if i % (self.batch_size*100) == 0:
                print(f"{i/n:.2f} of rows completed in {j+1} cycles, current loss at {np.mean(self.losses[-30:]):.4f}")  # noqa: E501
                self.scheduler.step()
                print(f"current learning rate at {self.optimizer.param_groups[0]['lr']:.6f}")  # noqa: E501

    def fit(self, model):

        self.train_loader: DataLoader = DataLoader(
            dataset=self.dataclass,
            batch_size=self.batch_size,
            num_workers=self.workers
        )
        model.to(self.device)
        start = time.time()
        model.train()

        epoch_count = 0
        self.losses = []

        print('fitting model...')

        for epoch in range(self.epochs):

            self.train_epoch(model)

            epoch_count += 1
            print(f'epoch {epoch_count} completed')
        print(f'model fitting completed, {time.time()-start:.0f} seconds passed')  # noqa: E501


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--archive_name', default='archive.zip')
    parser.add_argument('--fname', default='yelp_academic_dataset_review.json')
    parser.add_argument('--abs_path', default='projects/yelp_nlp/data/')
    parser.add_argument('--state_path', default='model_weight.pt')
    parser.add_argument('--device', default='cpu', help='run model on cuda or cpu')  # noqa: E501
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--input_len', type=int, default=200)
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=2
    )
    parser.add_argument(
        '--stop',
        type=int,
        default=10000,
        help='how many lines to load'
    )

    args = parser.parse_args()

    dataset = CorpusData(
        fpath=os.path.join(args.abs_path, args.archive_name),
        fname=args.fname,
        stop=args.stop
    )

    embedding_matrix = id_to_glove(dataset.dict_yelp, args.abs_path)
    emb_t = torch.from_numpy(embedding_matrix)

    model = RNN(
        emb_weights=emb_t,
        batch_size=args.batch_size,
        input_len=args.input_len)

    model.load_weights()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        # betas=(0.7, 0.99),
        weight_decay=1e-5
    )

    loss_function = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        100,
        eta_min=0,
        last_epoch=-1
    )

    trainer = Trainer(
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        dataclass=dataset,
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        workers=4,
        device=args.device
    )

    trainer.fit(model)
    weight_path = os.path.join(
        os.path.expanduser('~'),
        args.abs_path,
        args.state_path
    )
    torch.save(model.state_dict(), weight_path)
    print(f'model weights saved to: {args.state_path}')


if __name__ == '__main__':
    main()
