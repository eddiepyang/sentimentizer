from typing import Tuple
import pandas as pd
import os
import argparse

import torch
from gensim import corpora

from yelp_nlp.rnn.data import new_train_val_datasets
from yelp_nlp.rnn.train import Trainer, new_trainer
from yelp_nlp.rnn.model import RNN, new_model
from yelp_nlp.logging_utils import new_logger, time_decorator

from yelp_nlp.rnn.config import OptimizationParams, SchedulerParams, TrainerConfig

logger = new_logger(20)


@time_decorator
def main(
    df_path=None,
    dictionary_path=None,
):

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

    train_dataset, val_dataset = new_train_val_datasets(args.abs_path)

    model = new_model(
        dict_path=args.abs_path,
        embedding_path=args.embedding_path,
        batch_size=args.batch_size,
        input_len=args.input_len,
    )

    trainer = new_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=TrainerConfig(),
    )

    trainer.fit(model)

    weight_path = os.path.join(os.path.expanduser("~"), args.abs_path, args.state_path)
    torch.save(model.state_dict(), weight_path)
    logger.info(f"model weights saved to: {args.abs_path}{args.state_path}")


if __name__ == "__main__":

    main()
