import argparse
import asyncio
import atexit
import os
import shutil
import signal
from pathlib import Path

from dotenv import load_dotenv

# Load static environment variables (e.g. RAY_*) from .env file.
# .env is optional — defaults below are applied when the var is not set
# in the environment or .env file.
load_dotenv()

# Ray runtime: disable uv runtime env to prevent VIRTUAL_ENV warnings
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", "0.75")

# Ray Dashboard: where to find Grafana and Prometheus for the Metrics tab.
# These must be set BEFORE ray.init() so the dashboard can embed iframes.
os.environ.setdefault("RAY_GRAFANA_HOST", "http://localhost:3000")
os.environ.setdefault("RAY_PROMETHEUS_HOST", "http://localhost:9090")

# Ensure NVIDIA CUDA libraries bundled in the venv (nvidia/cu13/lib, etc.)
# are on LD_LIBRARY_PATH so that the driver process can load torch's CUDA dependencies.
from sentimentizer.env import ensure_nvidia_ld_library_path

nvidia_ld_library_path = ensure_nvidia_ld_library_path()

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
    weights_path_for,
)
from sentimentizer.extractor import extract_data
from sentimentizer.loader import (
    compute_pos_weight,
    load_train_val_corpus_datasets,
    load_train_val_ray_datasets,
)
from sentimentizer.tokenizer import Tokenizer
from sentimentizer.trainer import latest_checkpoint, load_checkpoint, new_ray_trainer, new_trainer

logger = new_logger(DEFAULT_LOG_LEVEL)


# ──────────────────────────────────────────────
# CUDA cleanup: graceful GPU release on exit
# ──────────────────────────────────────────────


def _cuda_cleanup() -> None:
    """Release cached CUDA memory and synchronize pending GPU ops.

    Safe to call even when CUDA is not available.  Called by atexit
    and signal handlers to prevent orphaned GPU contexts after Ctrl-C.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            # CUDA context may already be torn down during shutdown
            pass


# Register cleanup for normal process exit (including unhandled exceptions)
atexit.register(_cuda_cleanup)


def _sigint_handler(signum: int, frame: object) -> None:
    """Handle Ctrl-C by cleaning up CUDA before re-raising KeyboardInterrupt.

    Without this, a Ctrl-C during CUDA training can leave the GPU context
    in a bad state, causing error 804 on subsequent runs.
    """
    logger.info("Received SIGINT, cleaning up CUDA...")
    _cuda_cleanup()
    raise KeyboardInterrupt


# Install the handler for interactive Ctrl-C
signal.signal(signal.SIGINT, _sigint_handler)


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
    parser.add_argument("--save", action="store_true", default=False, help="save data and model")
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="use Ray Train for distributed training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
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
        "--tune",
        action="store_true",
        default=False,
        help="use TuningRun skill to tune hyperparameters and validate model predictions",
    )
    parser.add_argument(
        "--tune-mode",
        default="agent",
        choices=["agent", "standalone"],
        help="tuning mode: 'agent' (LLM-guided loop) or 'standalone' (single Ray Tune run)",
    )
    parser.add_argument(
        "--tune-samples",
        type=int,
        default=20,
        help="number of Ray Tune trials per tuning iteration (default: 20)",
    )
    parser.add_argument(
        "--tune-max-iterations",
        type=int,
        default=5,
        help="maximum agent tuning iterations (default: 5)",
    )
    parser.add_argument(
        "--tune-output-dir",
        default="tuning_results",
        help="directory to save tuning results and model weights (default: tuning_results)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="skip model prediction validation after tuning",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.75,
        help="minimum fraction of correct predictions to pass validation (default: 0.75)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="maximum re-tuning attempts if validation fails (default: 2)",
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
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        default=False,
        help="enable class balancing (undersampling majority class in training data)",  # noqa: E501
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="random seed for class balancing undersampling (default: 42)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=0.0,
        help="loss weight for the positive class (0 = auto-calculate from data, default: 0)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="push model weights to Hugging Face Hub after training",
    )
    parser.add_argument(
        "--pull-from-hub",
        action="store_true",
        default=False,
        help="pull model weights from Hugging Face Hub before running",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        default=False,
        help=(
            "run diagnostic checks on the training pipeline (dictionary alignment, "
            "embedding matrix, class balance, data integrity)"
        ),
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DriverConfig.hf.repo_id,
        help=f"Hugging Face repository ID (default: {DriverConfig.hf.repo_id})",
    )
    args = parser.parse_args()

    if args.type not in ("new", "update"):
        raise RunTypeError

    # Resolve auto-detect device
    if args.device == "auto":
        args.device = auto_detect_device()

    logger.info(  # type: ignore[call-arg]
        "running with args",
        early_stop=args.stop,
        distributed=args.distributed,
        ray_workers=args.num_workers,
    )
    return args


def _get_model_config(
    model_type: str,
) -> RNNConfig | EncoderConfig | DecoderConfig:
    """Get the model config for the given model type."""
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

    if not Path(DriverConfig.files.dictionary_file_path).exists():
        raise FileNotFoundError(
            f"Dictionary file not found at {DriverConfig.files.dictionary_file_path}. "
            "Ensure the tokenization step completed successfully before loading the model."
        )

    if args.type == "new":
        model = new_model(
            dict_path=DriverConfig.files.dictionary_file_path,
            embeddings_config=DriverConfig.embeddings(),
            input_len=DriverConfig.tokenizer.max_len,
            model_config=model_config,
        )
    elif args.type == "update":
        model = get_trained_model(
            device=args.device,
            model_config=model_config,
        )
    else:
        raise RunTypeError

    return model


def _remove_path(path: str) -> None:
    """Remove a file or directory at the given path.

    Ray Data writes parquet as a directory of part files, but a previous
    run may have left a regular file at the same location. This helper
    handles both cases so write_parquet never hits FileExistsError.
    """
    p = Path(path)
    if p.is_file() or p.is_symlink():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def _parquet_row_count(path: str) -> int:
    """Return the number of rows in a parquet file or directory from metadata.

    Returns 0 if the path does not exist.
    """
    import pyarrow.parquet as pq

    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return pq.read_metadata(str(p)).num_rows
    if p.is_dir():
        total = 0
        for f in sorted(p.glob("*.parquet")):
            total += pq.read_metadata(str(f)).num_rows
        return total
    return 0


def run_extract(args: argparse.Namespace) -> None:
    # Skip extraction if the raw parquet already has enough rows
    existing_rows = _parquet_row_count(DriverConfig.files.raw_reviews_file_path)
    if existing_rows >= args.stop:
        logger.info(f"skipping extract: {existing_rows} rows already exist (need {args.stop})")
        return

    ds = extract_data(
        DriverConfig.files.archive_file_path,
        DriverConfig.files.raw_file_path,
        stop=args.stop,
    )
    # Remove existing file or directory to avoid FileExistsError
    _remove_path(DriverConfig.files.raw_reviews_file_path)
    ds.write_parquet(DriverConfig.files.raw_reviews_file_path)


def run_tokenize(args: argparse.Namespace) -> None:
    # For 'new' runs, always (re)create the dictionary and re-tokenize the
    # data.  This guarantees the dictionary file exists on disk (needed by
    # new_model) and that token IDs in the processed data are consistent
    # with the dictionary.  Skipping is only safe for update/resume runs
    # where the dictionary already exists.
    if args.type == "new":
        reviews_data = ray.data.read_parquet(DriverConfig.files.raw_reviews_file_path)
        tokenizer = Tokenizer.from_dataset(reviews_data)

        processed_ds = tokenizer.transform_dataset(reviews_data)
        _remove_path(DriverConfig.files.processed_reviews_file_path)
        processed_ds.write_parquet(DriverConfig.files.processed_reviews_file_path)
    elif args.resume or args.type == "update":
        # Skip tokenization if the processed parquet already has enough rows
        existing_rows = _parquet_row_count(DriverConfig.files.processed_reviews_file_path)
        if existing_rows >= args.stop:
            logger.info(f"skipping tokenize: {existing_rows} rows already exist (need {args.stop})")
            return

        reviews_data = ray.data.read_parquet(DriverConfig.files.raw_reviews_file_path)
        # Load existing dictionary for checkpoint runs or update runs
        dictionary = corpora.Dictionary.load(DriverConfig.files.dictionary_file_path)
        tokenizer = Tokenizer(dictionary=dictionary)
        if args.resume:
            logger.info(
                f"resuming from checkpoint: updating dictionary from "
                f"{DriverConfig.files.dictionary_file_path}"
            )
            tokenizer.update_from_dataset(reviews_data)

        processed_ds = tokenizer.transform_dataset(reviews_data)
        _remove_path(DriverConfig.files.processed_reviews_file_path)
        processed_ds.write_parquet(DriverConfig.files.processed_reviews_file_path)
    else:
        raise RunTypeError


def run_fit(args: argparse.Namespace) -> None:
    if args.distributed:
        _run_fit_distributed(args)
    else:
        _run_fit_single(args)


def _run_fit_single(args: argparse.Namespace) -> None:
    """Single-node training using the existing Trainer class."""
    balance_classes = args.balance_classes
    train_dataset, val_dataset = load_train_val_corpus_datasets(
        data_path=DriverConfig.files.processed_reviews_file_path,
        balance_classes=balance_classes,
        random_state=args.balance_seed,
    )

    # Auto-compute pos_weight from training data if not explicitly set
    pos_weight = args.pos_weight
    if pos_weight == 0.0:
        if balance_classes:
            logger.info("using pos_weight=1.0 because class balancing (undersampling) is enabled")
            pos_weight = 1.0
        else:
            import pandas as pd

            train_df = pd.read_parquet(DriverConfig.files.processed_reviews_file_path)
            pos_weight = compute_pos_weight(train_df)
            del train_df

    model = _load_model(args)

    epochs = default_epochs(args.model)
    cfg = DriverConfig.trainer(
        epochs=epochs,
        device=args.device,
        dataloader_workers=default_dataloader_workers(args.device),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        pos_weight=pos_weight,
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

    try:
        trainer.fit(model, train_data=train_dataset, val_data=val_dataset)
    finally:
        # Ensure CUDA resources are released even on Ctrl-C or exception
        _cuda_cleanup()

    if args.save:
        weights_path = weights_path_for(args.model)
        torch.save(model.state_dict(), weights_path)
        logger.info(f"model weights saved to: {weights_path}")

        if args.push_to_hub:
            from sentimentizer.hf import push_model_to_hub

            # Use provided repo if it's not the default placeholder
            repo_id = args.hf_repo if args.hf_repo != DriverConfig.hf.repo_id else None

            push_model_to_hub(
                local_path=weights_path,
                model_type=args.model,
                repo_id=repo_id,
                dict_path=DriverConfig.files.dictionary_file_path,
            )


def _run_fit_distributed(args: argparse.Namespace) -> None:
    """Distributed training using Ray Train TorchTrainer."""
    balance_classes = args.balance_classes

    # Auto-compute pos_weight from full dataset if not explicitly set
    pos_weight = args.pos_weight
    if pos_weight == 0.0:
        if balance_classes:
            logger.info("using pos_weight=1.0 because class balancing (undersampling) is enabled")
            pos_weight = 1.0
        else:
            import pandas as pd

            full_df = pd.read_parquet(DriverConfig.files.processed_reviews_file_path)
            pos_weight = compute_pos_weight(full_df)
            del full_df

    train_ds, val_ds = load_train_val_ray_datasets(
        DriverConfig.files.processed_reviews_file_path,
        balance_classes=balance_classes,
        random_state=args.balance_seed,
    )

    epochs = default_epochs(args.model)
    cfg = DriverConfig.trainer(
        epochs=epochs,
        device=args.device,
        ray_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        pos_weight=pos_weight,
    )

    ray_trainer = new_ray_trainer(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        model_type=args.model,
    )

    try:
        result = ray_trainer.fit()
    finally:
        # Ensure CUDA resources are released even on Ctrl-C or exception
        _cuda_cleanup()

    logger.info(  # type: ignore[call-arg]
        "distributed training completed",
        best_checkpoint=result.checkpoint,
        metrics=result.metrics,
    )

    if args.save:
        # Load best checkpoint and save model weights
        # Ray 2.55+ uses directory-based checkpoints (to_dict removed)
        import os

        import ray.cloudpickle as pickle

        weights_path = weights_path_for(args.model)
        with result.checkpoint.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "data.pkl"), "rb") as fp:
                checkpoint_data = pickle.load(fp)
            torch.save(
                checkpoint_data["model_state_dict"],
                weights_path,
            )
        logger.info(f"model weights saved to: {weights_path}")

        if args.push_to_hub:
            from sentimentizer.hf import push_model_to_hub

            # Use provided repo if it's not the default placeholder
            repo_id = args.hf_repo if args.hf_repo != DriverConfig.hf.repo_id else None

            push_model_to_hub(
                local_path=weights_path,
                model_type=args.model,
                repo_id=repo_id,
                dict_path=DriverConfig.files.dictionary_file_path,
            )


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

    # Always save best config JSON
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
                "history": [
                    h.model_dump() if hasattr(h, "model_dump") else h for h in result.history
                ],
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"best config saved to: {output_path}")


def run_tune(args: argparse.Namespace) -> None:
    """Run the TuningRun skill to tune hyperparameters and validate model predictions.

    This is the main entry point for the tuning skill. It creates a
    TuningRunConfig from the CLI arguments, executes the tuning run,
    and logs the results including validation status.
    """
    from sentimentizer.agent.skill import TuningRun, TuningRunConfig

    config = TuningRunConfig(
        model_type=args.model,
        mode=args.tune_mode,
        config_path=args.agent_config,
        num_samples=args.tune_samples,
        max_iterations=args.tune_max_iterations,
        device=args.device,
        save_best_model=args.save,
        output_dir=args.tune_output_dir,
        validate_predictions=not args.no_validate,
        validation_threshold=args.validation_threshold,
        max_retries=args.max_retries,
        balance_classes=args.balance_classes,
        balance_seed=args.balance_seed,
        pos_weight=args.pos_weight,
        push_to_hub=args.push_to_hub,
    )

    run = TuningRun(config)

    logger.info(  # type: ignore[call-arg]
        "starting_tuning_skill",
        model_type=config.model_type,
        mode=config.mode,
        device=config.device,
        validate=config.validate_predictions,
        max_retries=config.max_retries,
    )

    try:
        result = run.execute()
    finally:
        # Ensure CUDA resources are released even on Ctrl-C or exception
        _cuda_cleanup()

    logger.info(  # type: ignore[call-arg]
        "tuning_skill_complete",
        best_accuracy=result.best_accuracy,
        best_loss=result.best_loss,
        best_f1=result.best_f1,
        best_cohen_kappa=result.best_cohen_kappa,
        best_positive_accuracy=result.best_positive_accuracy,
        best_negative_accuracy=result.best_negative_accuracy,
        iterations=result.iterations_completed,
        converged=result.converged,
        validation_passed=result.validation_passed,
        retry_count=result.retry_count,
        model_path=result.model_path,
        results_path=result.results_path,
        elapsed_seconds=result.elapsed_seconds,
    )

    if result.validation_passed:
        logger.info("tuning_skill_passed: model predictions validated successfully")
    else:
        logger.warning(
            "tuning_skill_failed: model predictions did not meet validation threshold "
            f"({result.validation_threshold}) after {result.retry_count} retries"
        )


def run_diagnose(args: argparse.Namespace) -> None:
    """Run diagnostic checks on the training pipeline.

    Checks for common issues that cause incorrect predictions:
    - Dictionary alignment between tokenization and embedding matrix
    - Embedding matrix row alignment with dictionary token IDs
    - Class balance in training data
    - Data integrity (token ID range validity)
    """
    import json

    from sentimentizer.agent.skill import diagnose_training_issues

    logger.info("running_diagnostics", model_type=args.model)
    result = diagnose_training_issues(model_type=args.model)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE DIAGNOSTICS")
    print("=" * 60)

    for check_name, check in result["checks"].items():
        status = (
            "PASS" if check.get("passed", False) else ("SKIP" if check.get("skipped") else "FAIL")
        )
        print(f"\n  [{status}] {check_name}")
        if "mismatch_rate" in check:
            print(
                f"    Mismatch rate: {check['mismatch_rate']:.1%} "
                f"({check.get('mismatches', '?')}/"
                f"{check.get('common_words', '?')} words)"
            )
        if "shape_matches" in check:
            print(
                f"    Shape matches: {check['shape_matches']} "
                f"(actual={check.get('actual_shape')}, "
                f"expected={check.get('expected_shape')})"
            )
        if "imbalance_ratio" in check:
            print(f"    Class imbalance ratio: {check['imbalance_ratio']}:1")
        if "invalid_token_count_in_sample" in check:
            print(f"    Invalid tokens in sample: {check['invalid_token_count_in_sample']}")
        if check.get("skipped"):
            print(f"    Skipped: {check.get('skip_reason', 'unknown')}")

    if result["critical_issues"]:
        print("\n  CRITICAL ISSUES:")
        for issue in result["critical_issues"]:
            print(f"    - {issue}")
    if result["warnings"]:
        print("\n  WARNINGS:")
        for warning in result["warnings"]:
            print(f"    - {warning}")
    if not result["critical_issues"] and not result["warnings"]:
        print("\n  All checks passed. No issues detected.")

    print("=" * 60 + "\n")

    # Also save full results as JSON
    diagnostics_path = "diagnostics_results.json"
    with open(diagnostics_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"diagnostics_saved_to_{diagnostics_path}")


@time_decorator
def main() -> None:
    args = new_parser()

    if args.pull_from_hub:
        from sentimentizer.config import DriverConfig
        from sentimentizer.hf import download_weights

        weights_path = weights_path_for(args.model)
        # Use provided repo if it's not the default placeholder
        repo_id = args.hf_repo if args.hf_repo != DriverConfig.hf.repo_id else None

        result_path = download_weights(
            model_type=args.model,
            local_path=weights_path,
            repo_id=repo_id,
            dict_path=DriverConfig.files.dictionary_file_path,
        )
        if result_path:
            logger.info(f"Pulled {args.model} weights from HF Hub. Forcing run type to 'update'.")
            args.type = "update"
        else:
            logger.error("Failed to pull weights from HF Hub. Proceeding with original run type.")

    # Diagnostics don't need Ray — run before initializing the cluster
    if args.diagnose:
        run_diagnose(args)
        return

    # Initialize Ray with a fixed metrics export port so Prometheus can scrape it.
    # The port must match metrics/prometheus.yml target (host.docker.internal:8080).
    # _metrics_export_port is a private API but is the only way to set this
    # programmatically in Ray 2.55 (public API uses `ray start --metrics-export-port`).
    if not ray.is_initialized():
        runtime_env = {}
        if nvidia_ld_library_path:
            # Propagate the LD_LIBRARY_PATH to Ray workers
            runtime_env["env_vars"] = {"LD_LIBRARY_PATH": nvidia_ld_library_path}

        ray.init(_metrics_export_port=8080, runtime_env=runtime_env)

    if args.tune:
        # Tuning skill mode: tune hyperparameters and validate model
        run_tune(args)
    elif args.agent_tune:
        # Agent tuning mode: uses LLM-guided hyperparameter search
        run_agent_tune(args)
    else:
        # Standard training mode
        run_extract(args)
        run_tokenize(args)
        run_fit(args)


if __name__ == "__main__":
    main()
