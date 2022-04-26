import os
import argparse
import torch
from yelp_nlp import root
from yelp_nlp.rnn.extractor import extract_data

from yelp_nlp.rnn.loader import load_train_val_corpus_datasets
from yelp_nlp.rnn.trainer import new_trainer
from yelp_nlp.rnn.model import new_model
from yelp_nlp.logging_utils import new_logger, time_decorator

from yelp_nlp.rnn.config import (
    EmbeddingsConfig,
    ParserConfig,
    RunnerConfig,
    TrainerConfig,
    FileConfig,
)
from yelp_nlp.rnn.config import LogLevels
from yelp_nlp.rnn.transformer import DataParser

logger = new_logger(LogLevels.debug.value)


@time_decorator
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state_path",
        default="model_weight.pt",
        help="file name for saved pytorch model weights",
    )
    parser.add_argument(
        "--device", default="cpu", help="run model on cuda or cpu"
    )  # noqa: E501
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument(
        "--stop", type=int, default=10000, help="how many lines to load"
    )
    args = parser.parse_args()

    train_dataset, val_dataset = load_train_val_corpus_datasets(RunnerConfig.data_path)
    reviews_data = extract_data(
        FileConfig.archive_path, FileConfig.review_filename, stop=args.stop
    )
    parser = DataParser(reviews_data, ParserConfig())
    parser.convert_sentences().save()

    model = new_model(
        dict_path=RunnerConfig.dictionary_path,
        embedding_path=EmbeddingsConfig.emb_path,
        batch_size=TrainerConfig.batch_size,
        input_len=ParserConfig.max_len,
    )

    trainer = new_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=TrainerConfig(),
    )
    trainer.fit(model)

    weight_path = os.path.join(root, args.state_path)
    torch.save(model.state_dict(), weight_path)
    logger.info(f"model weights saved to: {root}{args.state_path}")


if __name__ == "__main__":

    main()
