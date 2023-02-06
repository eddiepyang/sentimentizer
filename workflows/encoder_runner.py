import argparse
import torch
from torch_sentiment.extractor import extract_data
from torch_sentiment.trainer import new_trainer

from torch_sentiment.rnn.loader import load_train_val_corpus_datasets
from torch_sentiment.transformer.model import new_model
from torch_sentiment import new_logger, time_decorator

from torch_sentiment.rnn.config import DriverConfig, DEFAULT_LOG_LEVEL
from torch_sentiment.rnn.tokenizer import Tokenizer

logger = new_logger(DEFAULT_LOG_LEVEL)


def new_parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda", help="run model on cuda or cpu"
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

    transformer = Tokenizer.from_data(reviews_data)
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
