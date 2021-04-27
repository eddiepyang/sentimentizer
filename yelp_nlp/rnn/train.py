import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import os
import time
import numpy as np

from data import id_to_glove, CorpusData
from model import RNN


class Trainer:

    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""

    def __init__(
        self,
        loss_function: object,
        optimizer: optim.Adam,
        scheduler: optim.lr_scheduler,
        dataclass: CorpusData,
        batch_size: int,
        epochs: int,
        workers: int,
        device: str,
        memory: bool = True,
        mode: str = 'training'
    ):

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataclass = dataclass
        self.batch_size = batch_size
        self.epochs = epochs
        self.workers = workers
        self.device = device
        self.memory = memory
        if mode not in ['fitting', 'training']:
            raise ValueError('not an available mode')
        else:
            self.mode = mode
        self.create_loaders()

    def create_loaders(self):

        self.dataclass.set_mode(self.mode)
        self.train_loader: DataLoader = DataLoader(
            dataset=self.dataclass,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.memory
        )

        self.dataclass.set_mode('eval')
        self.val_loader = DataLoader(
            self.dataclass,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.memory
        )

    def train_epoch(self, model):

        self.dataclass.set_mode(self.mode)
        i = 0
        n = len(self.dataclass)

        for j, (sent, target) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            # noqa: E501
            log_probs = model(sent.long().to(self.device))
            loss = self.loss_function(log_probs, target.float().to(self.device))  # noqa: E501

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

            i += len(target)
            self.losses.append(loss.item())
            if i % (self.batch_size*100) == 0:
                print(f"{i/n:.2f} of rows completed in {j + 1} cycles, current loss at {np.mean(self.losses[-60:]):.6f}")  # noqa: E501
                self.scheduler.step()
                print(f"current learning rate at {self.optimizer.param_groups[0]['lr']:.6f}")  # noqa: E501

    def fit(self, model: object, mode: str = 'fitting'):

        model.to(self.device)
        start = time.time()
        model.train()

        epoch_count = 0
        self.losses = []

        print('fitting model...')

        for epoch in range(self.epochs):

            self.train_epoch(model)
            self.eval(model)
            epoch_count += 1
            print(f'epoch {epoch_count} completed')
        print(f'model fitting completed, {time.time()-start:.0f} seconds passed')  # noqa: E501

    def eval(self, model: object):

        print('evaluating predictions...')
        self.dataclass.set_mode('eval')
        losses = []
        i = 0
        n = len(self.dataclass)
        model.to(self.device)

        with torch.no_grad():

            model.eval()

            for j, (sent, target) in enumerate(self.val_loader):

                preds = model(sent.to(self.device))

                losses.append(
                    np.mean(
                        self.loss_function(
                            preds,
                            target.float().to(self.device)
                        ).item()
                    )
                )

                i += len(target)
                if i % (self.batch_size*100) == 0:
                    print(f"{i/n:.2f} of rows completed in {j + 1} cycles, current loss at {np.mean(losses[-60:]):.6f}")  # noqa: E501
            self.val_loss = np.mean(losses)
            print(f'validation loss at: {self.val_loss: .6f}')


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--archive_name', default='archive.zip')
    parser.add_argument('--fname', default='yelp_academic_dataset_review.json')
    parser.add_argument('--abs_path', default='projects/yelp_nlp/data/')
    parser.add_argument('--state_path', default='model_weight.pt')
    parser.add_argument('--device', default='cpu', help='run model on cuda or cpu')  # noqa: E501
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--input_len', type=int, default=100, help='width of lstm layer')  # noqa: E501
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
        stop=args.stop,
        max_len=args.input_len
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
        lr=0.001,
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
        device=args.device,
        mode='training'
    )

    trainer.fit(model)
    
    weight_path = os.path.join(
        os.path.expanduser('~'),
        args.abs_path,
        args.state_path
    )
    torch.save(model.state_dict(), weight_path)
    print(f'model weights saved to: {args.abs_path}{args.state_path}')


if __name__ == '__main__':
    main()
