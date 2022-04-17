import pandas as pd
import os
import argparse

import torch
from gensim import corpora

from yelp_nlp.rnn.data import CorpusDataset, id_to_glove
from yelp_nlp.rnn.train import Trainer
from yelp_nlp.rnn.model import RNN
from yelp_nlp.logging_utils import new_logger, time_decorator

from yelp_nlp.rnn.config import OptParams, SchedulerParams, TrainerConfig, loss_function

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

    df = pd.read_parquet(df_path)  # noqa: E501
    dataset = CorpusDataset(data=df)

    dict_yelp = corpora.Dictionary.load(args.abs_path)
    embedding_matrix = id_to_glove(dict_yelp, args.abs_path)
    emb_t = torch.from_numpy(embedding_matrix)

    model = RNN(emb_weights=emb_t, batch_size=args.batch_size, input_len=args.input_len)

    model.load_weights()

    params = OptParams()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
        betas=params.betas,
        weight_decay=params.weight_decay,
    )

    sp = SchedulerParams()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sp.T_max, eta_min=sp.eta_min, last_epoch=sp.last_epoch
    )

    trainer = Trainer(
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data=dataset,
        val_data=dataset,
        cfg=TrainerConfig,
    )

    trainer.fit(model)

    weight_path = os.path.join(os.path.expanduser("~"), args.abs_path, args.state_path)
    torch.save(model.state_dict(), weight_path)
    logger.info(f"model weights saved to: {args.abs_path}{args.state_path}")


if __name__ == "__main__":

    main()
