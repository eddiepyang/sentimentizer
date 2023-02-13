import argparse
import torch
import pandas as pd
from gensim import corpora

from torch_sentiment.extractor import extract_data, write_arrow
from torch_sentiment.trainer import new_trainer

from torch_sentiment.loader import load_train_val_corpus_datasets
from torch_sentiment import new_logger, time_decorator

from torch_sentiment.config import DriverConfig, DEFAULT_LOG_LEVEL
from torch_sentiment.tokenizer import Tokenizer

logger = new_logger(DEFAULT_LOG_LEVEL)


class RunTypeError(Exception):
    def __init__(self):
        super().__init__("incorrect run type found")


def new_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda", help="run model on cuda or cpu"
    )  # noqa: E501
    parser.add_argument(
        "--model",
        default="transformer",
        help="model loaded, must be rnn or transformer",
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
        raise RunTypeError

    logger.info(
        "running with args",
        device=args.device,
        early_stop=args.stop,
    )
    return args


def _load_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "rnn":
        from torch_sentiment.rnn.model import new_model, get_trained_model
    elif args.model == "transformer":
        from torch_sentiment.transformer.model import new_model, get_trained_model
    else:
        raise ValueError(f"no model {args.model}")

    if args.type == "new":
        model = new_model(
            dict_path=DriverConfig.files.dictionary_file_path,
            embeddings_config=DriverConfig.embeddings(),
            batch_size=DriverConfig.trainer.batch_size,
            input_len=DriverConfig.tokenizer.max_len,
        )
    elif args.type == "update":
        model = get_trained_model(DriverConfig.trainer.batch_size, args.device)
    else:
        raise RunTypeError

    return model


def run_extract(args: argparse.Namespace) -> None:

    gen = extract_data(
        DriverConfig.files.archive_file_path,
        DriverConfig.files.raw_file_path,
        stop=args.stop,
    )
    write_arrow(gen, args.stop, DriverConfig.files.raw_reviews_file_path)


def run_tokenize(args: argparse.Namespace) -> None:
    reviews_data = pd.read_feather(DriverConfig.files.raw_reviews_file_path)
    if args.type == "new":
        tokenizer = Tokenizer.from_data(reviews_data)
    elif args.type == "update":
        dictionary = corpora.Dictionary.load(DriverConfig.files.dictionary_file_path)
        tokenizer = Tokenizer(dictionary=dictionary)
    else:
        raise RunTypeError
    tokenizer.transform_dataframe(reviews_data)
    tokenizer.save(reviews_data)


def run_fit(args: argparse.Namespace) -> None:

    train_dataset, val_dataset = load_train_val_corpus_datasets(
        DriverConfig.files.processed_reviews_file_path
    )

    model = _load_model(args)

    trainer = new_trainer(
        model=model,
        cfg=DriverConfig.trainer(device=args.device),
    )
    trainer.fit(model, train_data=train_dataset, val_data=val_dataset)

    if args.save:
        torch.save(model.state_dict(), DriverConfig.files.weights_file_path)
        logger.info(f"model weights saved to: {DriverConfig.files.weights_file_path}")


@time_decorator
def main():

    args = new_parser()
    run_extract(args)
    run_tokenize(args)
    run_fit(args)


if __name__ == "__main__":

    main()
