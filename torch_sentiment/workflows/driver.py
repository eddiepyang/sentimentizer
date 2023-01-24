import argparse
import torch
import pandas as pd
from torch_sentiment.extractor import extract_data, write_arrow
from torch_sentiment.trainer import new_trainer

from torch_sentiment.rnn.loader import load_train_val_corpus_datasets
from torch_sentiment.rnn.model import get_trained_model, new_model
from torch_sentiment.logging_utils import new_logger, time_decorator

from torch_sentiment.rnn.config import DriverConfig, DEFAULT_LOG_LEVEL
from torch_sentiment.rnn.tokenizer import Tokenizer

logger = new_logger(DEFAULT_LOG_LEVEL)


def new_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda", help="run model on cuda or cpu"
    )  # noqa: E501
    parser.add_argument(
        "--type", default="new", help="type of run, must be new or update"
    )  # noqa: E501
    parser.add_argument(
        "--stop", type=int, default=10000, help="how many lines to load"
    )
    parser.add_argument("--save", type=bool, default=False, help="save data and model")
    args = parser.parse_args()

    if args.type not in ("new", "update"):
        raise ValueError("type must be new or update")

    logger.info(
        "running with args",
        device=args.device,
        early_stop=args.stop,
    )
    return args


@time_decorator
def main():

    args = new_parser()

    model = None
    if args.type == "new":
        gen = extract_data(
            DriverConfig.files.archive_file_path,
            DriverConfig.files.raw_file_path,
            stop=args.stop,
        )
        write_arrow(gen, args.stop, DriverConfig.files.raw_reviews_file_path)

        reviews_data = pd.read_feather(DriverConfig.files.raw_reviews_file_path)
        transformer = Tokenizer.from_data(reviews_data)
        transformer.transform_dataframe(reviews_data)
        transformer.save(reviews_data)

        model = new_model(
            dict_path=DriverConfig.files.dictionary_file_path,
            embeddings_config=DriverConfig.embeddings(),
            batch_size=DriverConfig.trainer.batch_size,
            input_len=DriverConfig.tokenizer.max_len,
        )

    if args.type == "update":
        model = get_trained_model(DriverConfig.trainer.batch_size, args.device)

    if model is None:
        raise TypeError("no model loaded, check the input arguments")

    train_dataset, val_dataset = load_train_val_corpus_datasets(
        DriverConfig.files.processed_reviews_file_path
    )
    trainer = new_trainer(
        model=model,
        cfg=DriverConfig.trainer(device=args.device),
    )

    trainer.fit(model, train_data=train_dataset, val_data=val_dataset)

    if args.save:
        torch.save(model.state_dict(), DriverConfig.files.weights_file_path)
        logger.info(f"model weights saved to: {DriverConfig.files.weights_file_path}")


if __name__ == "__main__":

    main()
