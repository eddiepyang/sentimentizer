import torch
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import numpy as np
import pandas as pd

from yelp_nlp.rnn.data import id_to_glove, CorpusDataset
from model import RNN


class Trainer:

    """Trainer class helps with creating the data loader,
    tracking the torch optimizer and model fitting"""

    def __init__(
        self,
        loss_function: object,
        optimizer: optim.Adam,
        dataclass: CorpusDataset,
        batch_size: int,
        epochs: int,
        workers: int,
        device: str,
        scheduler: optim.lr_scheduler = None,  # type: ignore
        memory: bool = True,
        mode: str = "training",
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
        if mode not in ["fitting", "training"]:
            raise ValueError("not an available mode")
        else:
            self.mode = mode
        self.create_loaders()

    def create_loaders(self):

        self.train_loader: DataLoader = DataLoader(
            dataset=self.dataclass,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.memory,
        )

        self.dataclass.set_mode("eval")
        self.val_loader = DataLoader(
            self.dataclass,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=self.memory,
        )

    def train_epoch(self, model):

        self.dataclass.set_mode(self.mode)
        i = 0
        n = len(self.dataclass)
        model.train()

        for j, (sent, target) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            # noqa: E501
            log_probs = model(sent.to(self.device))
            loss = self.loss_function(log_probs, target.to(self.device))  # noqa: E501

            # gets graident
            loss.backward()

            # clips high gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.3, norm_type=2
            )

            # updates with new gradient
            self.optimizer.step()

            i += len(target)
            self.losses.append(loss.item())
            if i % (self.batch_size * 100) == 0:
                if self.scheduler:
                    self.scheduler.step()
                print(
                    f"{i/n:.2f} of rows completed in {j + 1} cycles, current loss at {np.mean(self.losses[-60:]):.6f}"
                )  # noqa: E501
                print(
                    f"current learning rate at {self.optimizer.param_groups[0]['lr']:.6f}"
                )  # noqa: E501

    def fit(self, model: object):

        model.to(self.device)
        start = time.time()
        epoch_count = 0
        self.losses = []

        print("fitting model...")

        for epoch in range(self.epochs):

            self.train_epoch(model)
            self.eval(model)
            epoch_count += 1

            if self.scheduler:
                self.scheduler.step()

            print(f"epoch {epoch_count} completed")
        print(
            f"model fitting completed, {time.time()-start:.0f} seconds passed"
        )  # noqa: E501

    def eval(self, model: object):

        print("evaluating predictions...")
        self.dataclass.set_mode("eval")
        losses = []
        i = 0
        n = len(self.dataclass)
        model.to(self.device)

        with torch.no_grad():

            model.eval()

            for j, (sent, target) in enumerate(self.val_loader):

                preds = model(sent.to(self.device))

                losses.append(
                    np.mean(self.loss_function(preds, target.to(self.device)).item())
                )

                i += len(target)
                if i % (self.batch_size * 100) == 0:
                    print(
                        f"{i/n:.2f} of rows completed in {j + 1} cycles, current loss at {np.mean(losses[-60:]):.6f}"
                    )  # noqa: E501
            self.val_loss = np.mean(losses)
            print(f"validation loss at: {self.val_loss: .6f}")


def main(
    df_path=None,
    dictionary_path=None,
):

    import argparse
    from yelp_nlp.rnn.config import OptParams, SchedulerParams, loss_function

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--archive_name",
        default="archive.zip",
        help="file where yelp data is saved, expects an archive of json files",
    )
    parser.add_argument("--fname", default="yelp_academic_dataset_review.json")
    parser.add_argument(
        "--abs_path",
        default="projects/yelp_nlp/data/",
        help="folder where data is stored, path after /home/{user}/",
    )
    parser.add_argument(
        "--state_path",
        default="model_weight.pt",
        help="file name for saved pytorch model weights",
    )
    parser.add_argument(
        "--device", default="cpu", help="run model on cuda or cpu"
    )  # noqa: E501
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument(
        "--input_len", type=int, default=200, help="width of lstm layer"
    )  # noqa: E501
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument(
        "--stop", type=int, default=10000, help="how many lines to load"
    )

    args = parser.parse_args()

    if df_path:

        df = pd.read_parquet(df_path)  # noqa: E501

        dataset = CorpusDataset(
            fpath=os.path.join(args.abs_path, args.archive_name),
            fname=args.fname,
            df=df,
            stop=args.stop,
            max_len=args.input_len,
        )
    else:

        dataset = CorpusDataset(
            fpath=os.path.join(args.abs_path, args.archive_name),
            fname=args.fname,
            stop=args.stop,
            max_len=args.input_len,
        )

    embedding_matrix = id_to_glove(dataset.dict_yelp, args.abs_path)
    emb_t = torch.from_numpy(embedding_matrix)

    model = RNN(emb_weights=emb_t, batch_size=args.batch_size, input_len=args.input_len)

    model.load_weights()

    params = OptParams()

    optimizer = optim.Adam(
        model.parameters(),
        lr=params.lr,
        betas=params.betas,
        weight_decay=params.weight_decay,
    )

    sp = SchedulerParams()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sp.T_max, eta_min=sp.eta_min, last_epoch=sp.last_epoch
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
        mode="training",
    )

    trainer.fit(model)

    weight_path = os.path.join(os.path.expanduser("~"), args.abs_path, args.state_path)
    torch.save(model.state_dict(), weight_path)
    print(f"model weights saved to: {args.abs_path}{args.state_path}")


if __name__ == "__main__":

    main()
