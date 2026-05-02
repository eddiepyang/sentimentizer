import argparse
import asyncio
import os
import shutil

# Disable Ray's automatic uv runtime environment to prevent VIRTUAL_ENV warnings
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

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
    default_epochs,
)
from sentimentizer.extractor import extract_data
from sentimentizer.loader import load_train_val_corpus_datasets, load_train_val_ray_datasets
from sentimentizer.tokenizer import Tokenizer
from sentimentizer.trainer import latest_checkpoint, load_checkpoint, new_ray_trainer, new_trainer

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
    parser.add_argument(
        "--agent-tune",
        action="store_true",
        default=False,
        help="use Pydantic AI + LangGraph agent for hyperparameter tuning (GLM 5.1 via Ollama)",
    )
    parser.add_argument(
        "--agent-config",
        type=str,
        default=None,
        help="path to agent config YAML file (default: sentimentizer/agent/config.yaml)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="directory to save training checkpoints (empty = no checkpointing)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="save checkpoint every N epochs (0 = disabled, default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training from the latest checkpoint in --checkpoint-dir",
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

    epochs = default_epochs(args.model)
    cfg = DriverConfig.trainer(
        device=args.device,
        epochs=epochs,
        dataloader_workers=default_dataloader_workers(args.device),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    trainer = new_trainer(
        model=model,
        cfg=cfg,
        model_type=args.model,
    )

    # Resume from checkpoint if requested
    if args.resume:
        ckpt_path = latest_checkpoint(args.checkpoint_dir)
        if ckpt_path is None:
            logger.info("no checkpoint found, starting from scratch")
        else:
            checkpoint = load_checkpoint(
                ckpt_path, model, trainer.optimizer, trainer.scheduler, device=args.device
            )
            logger.info(
                f"resumed from checkpoint: {ckpt_path}, epoch={checkpoint.get('epoch', '?')}"
            )

    trainer.fit(model, train_data=train_dataset, val_data=val_dataset)

    if args.save:
        torch.save(model.state_dict(), DriverConfig.files.weights_file_path)
        logger.info(f"model weights saved to: {DriverConfig.files.weights_file_path}")


def _run_fit_distributed(args: argparse.Namespace) -> None:
    """Distributed training using Ray Train TorchTrainer."""
    train_ds, val_ds = load_train_val_ray_datasets(DriverConfig.files.processed_reviews_file_path)

    cfg = DriverConfig.trainer(
        device=args.device,
        ray_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

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


def run_agent_tune(args: argparse.Namespace) -> None:
    """Run the Pydantic AI + LangGraph tuning agent.

    Uses GLM 5.1 (via Ollama) for LLM reasoning about hyperparameter
    tuning, Ray Tune + Optuna for the actual search, and LangGraph
    for workflow orchestration with checkpointing.
    """
    from sentimentizer.agent import run_agent_tuning

    logger.info(  # type: ignore[call-arg]
        "starting_agent_tuning",
        model=args.model,
        agent_config=args.agent_config,
    )

    result = asyncio.run(
        run_agent_tuning(
            model_type=args.model,
            config_path=args.agent_config,
        )
    )

    logger.info(  # type: ignore[call-arg]
        "agent_tuning_complete",
        best_accuracy=result.best_accuracy,
        best_loss=result.best_loss,
        iterations=result.iterations_completed,
        converged=result.converged,
        best_config=result.best_config,
    )

    if args.save:
        import json
        from pathlib import Path

        output_path = Path("best_config.json")
        with open(output_path, "w") as f:
            json.dump(
                {
                    "best_config": result.best_config,
                    "best_accuracy": result.best_accuracy,
                    "best_loss": result.best_loss,
                    "iterations": result.iterations_completed,
                    "converged": result.converged,
                },
                f,
                indent=2,
            )
        logger.info(f"best config saved to: {output_path}")


@time_decorator
def main() -> None:
    args = new_parser()

    if args.agent_tune:
        # Agent tuning mode: uses LLM-guided hyperparameter search
        run_agent_tune(args)
    else:
        # Standard training mode
        run_extract(args)
        run_tokenize(args)
        run_fit(args)


if __name__ == "__main__":
    main()
