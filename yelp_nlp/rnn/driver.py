import os
import argparse
import torch
from yelp_nlp import root
from yelp_nlp.rnn.extractor import extract_data

from yelp_nlp.rnn.loader import load_train_val_corpus_datasets
from yelp_nlp.rnn.trainer import new_trainer
from yelp_nlp.rnn.model import new_model
from yelp_nlp.logging_utils import new_logger, time_decorator

from yelp_nlp.rnn.config import DriverConfig
from yelp_nlp.rnn.config import LogLevels
from yelp_nlp.rnn.transformer import DataTransformer

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

    transformer = DataTransformer(reviews_data, DriverConfig.transformer())
    transformer.transform_sentences().save()

    model = new_model(
        dict_path=DriverConfig.files.dictionary_file_path,
        embedding_path=DriverConfig.embeddings.file_path,
        batch_size=DriverConfig.trainer.batch_size,
        input_len=DriverConfig.transformer.max_len,
    )

    train_dataset, val_dataset = load_train_val_corpus_datasets(
        DriverConfig.files.reviews_file_path
    )

    trainer = new_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=DriverConfig.trainer(device=args.device),
    )
    trainer.fit(model)

    torch.save(model.state_dict(), DriverConfig.files.weights_file_path)
    logger.info(f"model weights saved to: {DriverConfig.files.weights_file_path}")


if __name__ == "__main__":

    main()
