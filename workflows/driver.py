import argparse
import shutil

import ray
import torch
from gensim import corpora

from sentimentizer import new_logger, time_decorator
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    DecoderConfig,
    DriverConfig,
    EncoderConfig,
    RNNConfig,
    auto_detect_device,
    default_dataloader_workers,
)
from sentimentizer.extractor import extract_data
from sentimentizer.loader import load_train_val_corpus_datasets, load_train_val_ray_datasets
from sentimentizer.tokenizer import Tokenizer
from sentimentizer.trainer import new_ray_trainer, new_trainer

logger = new_logger(DEFAULT_LOG_LEVEL)


class RunTypeError(Exception):
    def __init__(self) -> None:
        super().__init__("incorrect run type found")


def new_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="auto", help="device to use: auto (detect), cuda, mps, or cpu"
    )  # noqa: E501
    parser.add_argument(
        "--model",
        default="rnn",
        help="model loaded, must be rnn or transformer",
    )  # noqa: E501
    parser.add_argument(
        "--type", default="new", help="type of run, must be new or update"
    )  # noqa: E501
    parser.add_argument("--stop", type=int, default=10000, help="how many lines to load")
    parser.add_argument("--save", type=bool, default=False, help="save data and model")
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="use Ray Train for distributed training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="number of Ray Train workers for distributed training (--distributed only)",
    )
    args = parser.parse_args()

    if args.type not in ("new", "update"):
        raise RunTypeError

    # Resolve auto-detect device
    if args.device == "auto":
        args.device = auto_detect_device()

    logger.info(  # type: ignore[call-arg]
        "running with args",
        device=args.device,
        early_stop=args.stop,
        distributed=args.distributed,
        ray_workers=args.num_workers,
    )
    return args


def _get_model_config(
    model_type: str,
) -> RNNConfig | EncoderConfig | DecoderConfig:
    """Get the model config class for the given model type."""
    if model_type == "rnn":
        return DriverConfig.rnn()
    elif model_type == "encoder":
        return DriverConfig.encoder()
    elif model_type == "decoder":
        return DriverConfig.decoder()
    else:
        raise ValueError(f"no matching model config for {model_type}")


def _load_model(args: argparse.Namespace) -> torch.nn.Module:
    model_config = _get_model_config(args.model)

    if args.model == "rnn":
        from sentimentizer.models.rnn import get_trained_model, new_model
    elif args.model == "encoder":
        from sentimentizer.models.encoder import get_trained_model, new_model
    elif args.model == "decoder":
        from sentimentizer.models.decoder import get_trained_model, new_model
    else:
        raise ValueError(f"no matching model for {args.model}")

    if args.type == "new":
        model = new_model(
            dict_path=DriverConfig.files.dictionary_file_path,
            embeddings_config=DriverConfig.embeddings(),
            batch_size=DriverConfig.trainer.batch_size,
            input_len=DriverConfig.tokenizer.max_len,
            model_config=model_config,
        )
    elif args.type == "update":
        model = get_trained_model(
            DriverConfig.trainer.batch_size,
            args.device,
            model_config=model_config,
        )
    else:
        raise RunTypeError

    return model


def run_extract(args: argparse.Namespace) -> None:
    ds = extract_data(
        DriverConfig.files.archive_file_path,
        DriverConfig.files.raw_file_path,
        stop=args.stop,
    )
    # Remove existing directory if it exists to clean up
    shutil.rmtree(DriverConfig.files.raw_reviews_file_path, ignore_errors=True)
    ds.write_parquet(DriverConfig.files.raw_reviews_file_path)


def run_tokenize(args: argparse.Namespace) -> None:
    reviews_data = ray.data.read_parquet(DriverConfig.files.raw_reviews_file_path)
    if args.type == "new":
        tokenizer = Tokenizer.from_dataset(reviews_data)
    elif args.type == "update":
        dictionary = corpora.Dictionary.load(DriverConfig.files.dictionary_file_path)
        tokenizer = Tokenizer(dictionary=dictionary)
    else:
        raise RunTypeError

    processed_ds = tokenizer.transform_dataset(reviews_data)
    shutil.rmtree(DriverConfig.files.processed_reviews_file_path, ignore_errors=True)
    processed_ds.write_parquet(DriverConfig.files.processed_reviews_file_path)


def run_fit(args: argparse.Namespace) -> None:
    if args.distributed:
        _run_fit_distributed(args)
    else:
        _run_fit_single(args)


def _run_fit_single(args: argparse.Namespace) -> None:
    """Single-node training using the existing Trainer class."""
    train_dataset, val_dataset = load_train_val_corpus_datasets(
        DriverConfig.files.processed_reviews_file_path
    )

    model = _load_model(args)

    cfg = DriverConfig.trainer(
        device=args.device,
        dataloader_workers=default_dataloader_workers(args.device),
    )
    trainer = new_trainer(
        model=model,
        cfg=cfg,
        model_type=args.model,
    )
    trainer.fit(model, train_data=train_dataset, val_data=val_dataset)

    if args.save:
        torch.save(model.state_dict(), DriverConfig.files.weights_file_path)
        logger.info(f"model weights saved to: {DriverConfig.files.weights_file_path}")


def _run_fit_distributed(args: argparse.Namespace) -> None:
    """Distributed training using Ray Train TorchTrainer."""
    train_ds, val_ds = load_train_val_ray_datasets(DriverConfig.files.processed_reviews_file_path)

    cfg = DriverConfig.trainer(device=args.device, ray_workers=args.num_workers)

    ray_trainer = new_ray_trainer(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        model_type=args.model,
    )

    result = ray_trainer.fit()

    logger.info(  # type: ignore[call-arg]
        "distributed training completed",
        best_checkpoint=result.checkpoint,
        metrics=result.metrics,
    )

    if args.save:
        # Load best checkpoint and save model weights
        with result.checkpoint.as_directory():
            checkpoint_data = result.checkpoint.to_dict()
            torch.save(
                checkpoint_data["model_state_dict"],
                DriverConfig.files.weights_file_path,
            )
        logger.info(f"model weights saved to: {DriverConfig.files.weights_file_path}")


@time_decorator
def main() -> None:
    args = new_parser()
    run_extract(args)
    run_tokenize(args)
    run_fit(args)


if __name__ == "__main__":
    main()
