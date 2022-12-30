import os
import argparse
import torch
from torch_sentiment import root
from torch_sentiment.rnn.extractor import extract_data

from torch_sentiment.rnn.loader import load_train_val_corpus_datasets
from torch_sentiment.rnn.trainer import new_trainer
from torch_sentiment.rnn.model import new_model
from torch_sentiment.logging_utils import new_logger, time_decorator

from torch_sentiment.rnn.config import DriverConfig
from torch_sentiment.rnn.config import LogLevels
from torch_sentiment.rnn.tokenizer import Tokenizer

logger = new_logger(LogLevels.debug.value)


def new_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cpu", help="run model on cuda or cpu"
    )  # noqa: E501
    parser.add_argument(
        "--stop", type=int, default=10000, help="how many lines to load"
    )
    args = parser.parse_args()

    logger.info(
        "running with args",
        device=args.device,
        early_stop=args.stop,
    )
    return args


@time_decorator
def main():

    args = new_parser()

    reviews_data = extract_data(
        DriverConfig.files.archive_file_path,
        DriverConfig.files.raw_file_path,
        stop=args.stop,
    )

    transformer = Tokenizer(reviews_data, DriverConfig.tokenizer)
    transformer.transform_dataframe(reviews_data).save(reviews_data)

    model = new_model(
        dict_path=DriverConfig.files.dictionary_file_path,
        embeddings_config=DriverConfig.embeddings,
        batch_size=DriverConfig.trainer.batch_size,
        input_len=DriverConfig.tokenizer.max_len,
    )

    train_dataset, val_dataset = load_train_val_corpus_datasets(
        DriverConfig.files.reviews_file_path
    )

    trainer = new_trainer(
        model=model,
        cfg=DriverConfig.trainer(device=args.device),
    )
    trainer.fit(model, train_data=train_dataset, val_data=val_dataset)

    torch.save(model.state_dict(), DriverConfig.files.weights_file_path)
    logger.info(f"model weights saved to: {DriverConfig.files.weights_file_path}")


if __name__ == "__main__":

    main()
