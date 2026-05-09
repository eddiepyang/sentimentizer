"""Sentimentizer ML pipeline — Click-based CLI driver.

Replaces the original argparse driver with per-stage subcommands while
preserving full behavior parity.  The ``run`` subcommand chains
extract → tokenize → train, matching the old ``--type new`` default.

Heavy imports (torch, ray, gensim) are deferred to the command bodies
that use them so that ``--help`` and lightweight commands render without
loading the ML stack.
"""

import atexit
import contextlib
import os
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 1. Environment setup — must happen before ANY ray/torch import
# ──────────────────────────────────────────────

load_dotenv()
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", "0.75")
os.environ.setdefault("RAY_GRAFANA_HOST", "http://localhost:3000")
os.environ.setdefault("RAY_PROMETHEUS_HOST", "http://localhost:9090")

# Note: do NOT call logging.basicConfig() here or in the cli callback.
# Logging is configured by structlog in sentimentizer/__init__.py at import
# time; basicConfig wouldn't reach structlog's logger factory and would
# silently shadow the existing config for any stdlib-logging consumers.

# ──────────────────────────────────────────────
# 2. Cleanup handlers — registered once at module load
# ──────────────────────────────────────────────

from logging import INFO as _INFO  # noqa: E402

from sentimentizer import new_logger  # noqa: E402

logger = new_logger(_INFO)

# Ray stores session data in /tmp/ray/session_<timestamp>_<id>/.
# Each session can consume 5+ GB. When Ray is not shut down cleanly
# (e.g., Ctrl-C, crash, or test runner killing the process), these
# directories accumulate and fill the disk.
_RAY_SESSION_DIR = Path("/tmp/ray")


def _cuda_cleanup() -> None:
    """Release cached CUDA memory.  Safe even when torch was never imported."""
    if "torch" not in sys.modules:
        return
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            # CUDA context may already be torn down during shutdown
            pass


def _ray_cleanup() -> None:
    """Shut down Ray and clean up stale session temp files.

    Safe even when ray was never imported.
    """
    if "ray" not in sys.modules:
        return
    import ray

    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    # Clean up stale Ray session directories left by previous runs
    if _RAY_SESSION_DIR.exists():
        current_session = _RAY_SESSION_DIR / "session_latest"
        for session_dir in _RAY_SESSION_DIR.iterdir():
            if session_dir == current_session:
                continue
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                try:
                    age_hours = (os.path.getmtime(session_dir) - __import__("time").time()) / 3600
                    if age_hours < -1:  # modified more than 1 hour ago
                        shutil.rmtree(session_dir, ignore_errors=True)
                        logger.debug(f"cleaned stale Ray session: {session_dir}")
                except OSError:
                    pass


def _cleanup_stale_ray_sessions() -> None:
    """Remove stale Ray session directories from /tmp/ray.

    Called before ray.init() to free disk space from previous runs.
    Only removes sessions older than 1 hour to avoid deleting active sessions.
    """
    if not _RAY_SESSION_DIR.exists():
        return

    import time

    current_time = time.time()
    removed = 0
    freed_gb = 0.0

    for session_dir in _RAY_SESSION_DIR.iterdir():
        if session_dir.name == "session_latest":
            continue
        if session_dir.is_dir() and session_dir.name.startswith("session_"):
            try:
                age_hours = (current_time - os.path.getmtime(session_dir)) / 3600
                if age_hours > 1:  # older than 1 hour
                    size = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
                    shutil.rmtree(session_dir, ignore_errors=True)
                    freed_gb += size / (1024**3)
                    removed += 1
            except OSError:
                pass

    if removed > 0:
        logger.info(  # type: ignore[call-arg]
            "cleaned_stale_ray_sessions",
            removed=removed,
            freed_gb=round(freed_gb, 1),
        )


def _sigint_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl-C by cleaning up CUDA and Ray before re-raising."""
    logger.info("Received SIGINT, cleaning up...")
    _cuda_cleanup()
    _ray_cleanup()
    raise KeyboardInterrupt


atexit.register(_cuda_cleanup)
atexit.register(_ray_cleanup)
signal.signal(signal.SIGINT, _sigint_handler)


# ──────────────────────────────────────────────
# 3. Shared Ray bootstrap — every run_* helper calls this first
# ──────────────────────────────────────────────


def _ensure_ray_initialized() -> None:
    """Initialize Ray with metrics port and runtime_env if not already running.

    Every ``run_*`` helper that touches ``ray.data`` or ``ray.train`` calls
    this as its first line.  Click command shells stay thin (one-line
    forwarders) and don't call it themselves — that way you can't
    accidentally use a helper from another helper or a test without
    bringing Ray with it.
    """
    import ray

    if ray.is_initialized():
        return
    from sentimentizer.env import ensure_nvidia_ld_library_path

    ld_path = ensure_nvidia_ld_library_path()
    runtime_env: dict[str, Any] = {}
    if ld_path:
        runtime_env["env_vars"] = {"LD_LIBRARY_PATH": ld_path}
    _cleanup_stale_ray_sessions()
    ray.init(_metrics_export_port=8080, runtime_env=runtime_env)


# ──────────────────────────────────────────────
# 4. Typed shared state (replaces raw dict ctx.obj)
# ──────────────────────────────────────────────


@dataclass
class State:
    model: str
    device: str  # raw value: "auto" | "cuda" | "mps" | "cpu" — resolve lazily
    run_type: str  # "new" | "update"


# ──────────────────────────────────────────────
# 5. Helper utilities (no click dependency)
# ──────────────────────────────────────────────


def _get_model_config(model_type: str) -> Any:
    """Get the model config for the given model type."""
    from sentimentizer.config import DriverConfig

    if model_type == "rnn":
        return DriverConfig.rnn()
    elif model_type == "encoder":
        return DriverConfig.encoder()
    elif model_type == "decoder":
        return DriverConfig.decoder()
    else:
        raise ValueError(f"no matching model config for {model_type}")


def _load_model(state: State, device: str) -> Any:
    """Load a model, either fresh (new) or from checkpoint (update)."""

    from sentimentizer.config import DriverConfig

    model_config = _get_model_config(state.model)

    if state.model == "rnn":
        from sentimentizer.models.rnn import get_trained_model, new_model
    elif state.model == "encoder":
        from sentimentizer.models.encoder import get_trained_model, new_model
    elif state.model == "decoder":
        from sentimentizer.models.decoder import get_trained_model, new_model
    else:
        raise ValueError(f"no matching model for {state.model}")

    if not Path(DriverConfig.files.dictionary_file_path).exists():
        raise FileNotFoundError(
            f"Dictionary file not found at {DriverConfig.files.dictionary_file_path}. "
            "Ensure the tokenization step completed successfully before loading the model."
        )

    if state.run_type == "new":
        model = new_model(
            dict_path=DriverConfig.files.dictionary_file_path,
            embeddings_config=DriverConfig.embeddings(),
            input_len=DriverConfig.tokenizer.max_len,
            model_config=model_config,
        )
    elif state.run_type == "update":
        model = get_trained_model(
            device=device,
            model_config=model_config,
        )
    else:
        raise ValueError(f"invalid run_type: {state.run_type}")

    return model


def _remove_path(path: str) -> None:
    """Remove a file or directory at the given path.

    Ray Data writes parquet as a directory of part files, but a previous
    run may have left a regular file at the same location.  This helper
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


# ──────────────────────────────────────────────
# 6. run_* helpers — business logic, no click decorators
# ──────────────────────────────────────────────


def run_extract(state: State, *, stop: int) -> None:
    """Extract raw reviews into parquet."""
    _ensure_ray_initialized()

    from sentimentizer.config import DriverConfig
    from sentimentizer.extractor import extract_data

    # Skip extraction if the raw parquet already has enough rows
    existing_rows = _parquet_row_count(DriverConfig.files.raw_reviews_file_path)
    if existing_rows >= stop:
        logger.info(f"skipping extract: {existing_rows} rows already exist (need {stop})")
        return

    ds = extract_data(
        DriverConfig.files.archive_file_path,
        DriverConfig.files.raw_file_path,
        stop=stop,
    )
    _remove_path(DriverConfig.files.raw_reviews_file_path)
    ds.write_parquet(DriverConfig.files.raw_reviews_file_path)


def run_tokenize(state: State, *, resume: bool) -> None:
    """Build/update dictionary and write processed parquet."""
    import ray
    from gensim import corpora

    from sentimentizer.config import DriverConfig
    from sentimentizer.tokenizer import Tokenizer

    _ensure_ray_initialized()

    # For 'new' runs, always (re)create the dictionary and re-tokenize
    if state.run_type == "new":
        reviews_data = ray.data.read_parquet(DriverConfig.files.raw_reviews_file_path)
        tokenizer = Tokenizer.from_dataset(reviews_data)

        processed_ds = tokenizer.transform_dataset(reviews_data)
        _remove_path(DriverConfig.files.processed_reviews_file_path)
        processed_ds.write_parquet(DriverConfig.files.processed_reviews_file_path)
    elif resume or state.run_type == "update":
        # Skip tokenization if the processed parquet already has enough rows
        from sentimentizer.config import TokenizerConfig

        stop = TokenizerConfig.stop
        existing_rows = _parquet_row_count(DriverConfig.files.processed_reviews_file_path)
        if existing_rows >= stop:
            logger.info(f"skipping tokenize: {existing_rows} rows already exist (need {stop})")
            return

        reviews_data = ray.data.read_parquet(DriverConfig.files.raw_reviews_file_path)
        dictionary = corpora.Dictionary.load(DriverConfig.files.dictionary_file_path)
        tokenizer = Tokenizer(dictionary=dictionary)
        if resume:
            logger.info(
                f"resuming from checkpoint: updating dictionary from "
                f"{DriverConfig.files.dictionary_file_path}"
            )
            tokenizer.update_from_dataset(reviews_data)

        processed_ds = tokenizer.transform_dataset(reviews_data)
        _remove_path(DriverConfig.files.processed_reviews_file_path)
        processed_ds.write_parquet(DriverConfig.files.processed_reviews_file_path)
    else:
        raise ValueError(f"invalid run_type: {state.run_type}")


def _publish_distributed_metrics(result: Any, model_type: str) -> None:
    """Publish final distributed training metrics to the driver-level Prometheus gauges.

    During distributed training, ``_train_func`` runs inside Ray workers --
    separate processes that set their own ``prometheus_client`` gauge objects.
    Those worker-process gauges are NOT the same as the driver's gauges, so
    the driver-level ``sentimentizer_training_*`` metrics never get updated.

    After ``ray_trainer.fit()`` returns, the driver has the final metrics in
    ``result.metrics``.  This function persists those values to a JSON file
    that the standalone exporter (port 8081) reads periodically, so training
    metrics remain visible in Grafana after the driver process exits.

    The gauge-setting code is kept as a best-effort backup: if the driver
    process happens to be scraped by Prometheus before exiting, the gauges
    will reflect the final metrics.
    """
    try:
        from sentimentizer.exporter import (
            TRAINING_EPOCH,
            TRAINING_TRAIN_LOSS,
            TRAINING_VAL_ACCURACY,
            TRAINING_VAL_AUC_ROC,
            TRAINING_VAL_COHEN_KAPPA,
            TRAINING_VAL_F1,
            TRAINING_VAL_LOSS,
            TRAINING_VAL_NEGATIVE_ACCURACY,
            TRAINING_VAL_POSITIVE_ACCURACY,
            TRAINING_VAL_PRECISION,
            TRAINING_VAL_RECALL,
        )

        metrics = result.metrics
        lbl = {"model_type": model_type}

        TRAINING_TRAIN_LOSS.labels(**lbl).set(float(metrics.get("train_loss", 0)))
        TRAINING_VAL_LOSS.labels(**lbl).set(float(metrics.get("val_loss", 0)))
        TRAINING_VAL_ACCURACY.labels(**lbl).set(float(metrics.get("accuracy", 0)))
        TRAINING_VAL_PRECISION.labels(**lbl).set(float(metrics.get("precision", 0)))
        TRAINING_VAL_RECALL.labels(**lbl).set(float(metrics.get("recall", 0)))
        TRAINING_VAL_F1.labels(**lbl).set(float(metrics.get("f1", 0)))
        TRAINING_VAL_COHEN_KAPPA.labels(**lbl).set(float(metrics.get("cohen_kappa", 0)))
        auc_roc = metrics.get("auc_roc")
        if auc_roc is not None:
            TRAINING_VAL_AUC_ROC.labels(**lbl).set(float(auc_roc))
        TRAINING_VAL_POSITIVE_ACCURACY.labels(**lbl).set(float(metrics.get("pos_acc", 0)))
        TRAINING_VAL_NEGATIVE_ACCURACY.labels(**lbl).set(float(metrics.get("neg_acc", 0)))
        TRAINING_EPOCH.labels(**lbl).set(int(metrics.get("epoch", 0)))

        logger.info(
            "distributed_metrics_published_to_prometheus",
            model_type=model_type,
            accuracy=metrics.get("accuracy"),
            val_loss=metrics.get("val_loss"),
            train_loss=metrics.get("train_loss"),
        )
    except ImportError:
        logger.warning(
            "prometheus_client_not_available",
            message="Cannot publish distributed training metrics to Prometheus gauges",
        )

    # Persist metrics to JSON so the standalone exporter can serve them
    # after the driver process exits.
    _persist_metrics_to_file(result.metrics, model_type)


def _persist_metrics_to_file(metrics: dict, model_type: str) -> None:
    """Write final training metrics to a JSON file for the standalone exporter.

    The standalone exporter on port 8081 reads this file periodically and
    updates its own Prometheus gauges.  This ensures training metrics persist
    after the driver process exits.
    """
    import json

    path = Path("/tmp/sentimentizer_training_metrics.json")

    # Read existing data (may contain metrics from other model types)
    data: dict[str, dict] = {}
    if path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            data = json.loads(path.read_text())

    auc_roc = metrics.get("auc_roc")
    data[model_type] = {
        "train_loss": float(metrics.get("train_loss", 0)),
        "val_loss": float(metrics.get("val_loss", 0)),
        "accuracy": float(metrics.get("accuracy", 0)),
        "precision": float(metrics.get("precision", 0)),
        "recall": float(metrics.get("recall", 0)),
        "f1": float(metrics.get("f1", 0)),
        "cohen_kappa": float(metrics.get("cohen_kappa", 0)),
        "auc_roc": float(auc_roc) if auc_roc is not None else None,
        "positive_accuracy": float(metrics.get("pos_acc", 0)),
        "negative_accuracy": float(metrics.get("neg_acc", 0)),
        "epoch": int(metrics.get("epoch", 0)),
    }

    path.write_text(json.dumps(data, indent=2))
    logger.info("training_metrics_persisted", path=str(path), model_type=model_type)


def run_train(
    state: State,
    *,
    distributed: bool,
    num_workers: int,
    save: bool,
    checkpoint_dir: str,
    checkpoint_every: int,
    resume: bool,
    balance_classes: bool,
    balance_seed: int,
    pos_weight: float,
    push_to_hub: bool,
    pull_from_hub: bool,
    hf_repo: str | None,
) -> None:
    """Fit the model (single-node or distributed)."""
    from sentimentizer.device import resolve_device

    device = resolve_device(state.device)

    from sentimentizer.config import (
        DriverConfig,
        default_epochs,
        weights_path_for,
    )

    # Handle --pull-from-hub before training (preserves old behavior)
    if pull_from_hub:
        from sentimentizer.hf import download_weights

        weights_path = weights_path_for(state.model)
        repo_id = hf_repo if hf_repo is not None else DriverConfig.hf.repo_id
        # Use the per-model repo if hf_repo matches the default
        if hf_repo is None:
            from sentimentizer.config import HF_WEIGHTS_REPOS

            repo_id = HF_WEIGHTS_REPOS.get(state.model)

        result_path = download_weights(
            model_type=state.model,
            local_path=weights_path,
            repo_id=repo_id,
            dict_path=DriverConfig.files.dictionary_file_path,
        )
        if result_path:
            logger.info(f"Pulled {state.model} weights from HF Hub. Forcing run type to 'update'.")
            state.run_type = "update"
        else:
            logger.error("Failed to pull weights from HF Hub. Proceeding with original run type.")

    model = _load_model(state, device)

    epochs = default_epochs(state.model)

    if distributed:
        _run_fit_distributed(
            state=state,
            device=device,
            model=model,
            epochs=epochs,
            num_workers=num_workers,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            balance_classes=balance_classes,
            balance_seed=balance_seed,
            pos_weight=pos_weight,
            save=save,
            push_to_hub=push_to_hub,
            hf_repo=hf_repo,
        )
    else:
        _run_fit_single(
            state=state,
            device=device,
            model=model,
            epochs=epochs,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            resume=resume,
            balance_classes=balance_classes,
            balance_seed=balance_seed,
            pos_weight=pos_weight,
            save=save,
            push_to_hub=push_to_hub,
            hf_repo=hf_repo,
        )


def _run_fit_single(
    state: State,
    device: str,
    model: Any,
    epochs: int,
    checkpoint_dir: str,
    checkpoint_every: int,
    resume: bool,
    balance_classes: bool,
    balance_seed: int,
    pos_weight: float,
    save: bool,
    push_to_hub: bool,
    hf_repo: str | None,
) -> None:
    """Single-node training using the existing Trainer class."""
    import torch

    from sentimentizer.config import (
        DriverConfig,
        default_dataloader_workers,
        weights_path_for,
    )
    from sentimentizer.loader import compute_pos_weight, load_train_val_corpus_datasets
    from sentimentizer.trainer import new_trainer

    from_sentinel = pos_weight == 0.0
    train_dataset, val_dataset = load_train_val_corpus_datasets(
        data_path=DriverConfig.files.processed_reviews_file_path,
        balance_classes=balance_classes,
        random_state=balance_seed,
    )

    # Auto-compute pos_weight from training data if not explicitly set
    if from_sentinel:
        if balance_classes:
            logger.info("using pos_weight=1.0 because class balancing (undersampling) is enabled")
            pos_weight = 1.0
        else:
            import pandas as pd

            train_df = pd.read_parquet(DriverConfig.files.processed_reviews_file_path)
            pos_weight = compute_pos_weight(train_df)
            del train_df

    cfg = DriverConfig.trainer(
        epochs=epochs,
        device=device,
        dataloader_workers=default_dataloader_workers(device),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        pos_weight=pos_weight,
    )

    trainer = new_trainer(
        model=model,
        cfg=cfg,
        model_type=state.model,
    )

    # Resume from checkpoint if requested
    if resume:
        from sentimentizer.trainer import latest_checkpoint, load_checkpoint

        ckpt_path = latest_checkpoint(checkpoint_dir)
        if ckpt_path is None:
            logger.info("no checkpoint found, starting from scratch")
        else:
            checkpoint = load_checkpoint(
                ckpt_path, model, trainer.optimizer, trainer.scheduler, device=device
            )
            logger.info(
                f"resumed from checkpoint: {ckpt_path}, epoch={checkpoint.get('epoch', '?')}"
            )

    try:
        trainer.fit(model, train_data=train_dataset, val_data=val_dataset)
    finally:
        _cuda_cleanup()

    if save:
        weights_path = weights_path_for(state.model)
        torch.save(model.state_dict(), weights_path)
        logger.info(f"model weights saved to: {weights_path}")

        if push_to_hub:
            from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig
            from sentimentizer.hf import push_model_to_hub

            # Use per-model repo if hf_repo is not explicitly provided
            repo_id = hf_repo if hf_repo is not None else HF_WEIGHTS_REPOS.get(state.model)

            push_model_to_hub(
                local_path=weights_path,
                model_type=state.model,
                repo_id=repo_id,
                dict_path=DriverConfig.files.dictionary_file_path,
            )


def _run_fit_distributed(
    state: State,
    device: str,
    model: Any,
    epochs: int,
    num_workers: int,
    checkpoint_dir: str,
    checkpoint_every: int,
    balance_classes: bool,
    balance_seed: int,
    pos_weight: float,
    save: bool,
    push_to_hub: bool,
    hf_repo: str | None,
) -> None:
    """Distributed training using Ray Train TorchTrainer."""
    import torch

    from sentimentizer.config import (
        DriverConfig,
        weights_path_for,
    )
    from sentimentizer.loader import compute_pos_weight, load_train_val_ray_datasets
    from sentimentizer.trainer import new_ray_trainer

    _ensure_ray_initialized()

    # Auto-compute pos_weight from full dataset if not explicitly set
    from_sentinel = pos_weight == 0.0
    if from_sentinel:
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
        random_state=balance_seed,
    )

    cfg = DriverConfig.trainer(
        epochs=epochs,
        device=device,
        ray_workers=num_workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        pos_weight=pos_weight,
    )

    ray_trainer = new_ray_trainer(
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=cfg,
        model_type=state.model,
    )

    try:
        result = ray_trainer.fit()
    finally:
        _cuda_cleanup()

    # Publish final metrics from distributed training to the driver-level
    # Prometheus gauges so they are visible on the /metrics endpoint.
    _publish_distributed_metrics(result, state.model)

    logger.info(  # type: ignore[call-arg]
        "distributed training completed",
        best_checkpoint=result.checkpoint,
        metrics=result.metrics,
    )

    if save:
        import os

        import ray.cloudpickle as pickle

        weights_path = weights_path_for(state.model)
        with result.checkpoint.as_directory() as checkpoint_dir_path:
            with open(os.path.join(checkpoint_dir_path, "data.pkl"), "rb") as fp:
                checkpoint_data = pickle.load(fp)
            torch.save(
                checkpoint_data["model_state_dict"],
                weights_path,
            )
        logger.info(f"model weights saved to: {weights_path}")

        if push_to_hub:
            from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig
            from sentimentizer.hf import push_model_to_hub

            repo_id = hf_repo if hf_repo is not None else HF_WEIGHTS_REPOS.get(state.model)

            push_model_to_hub(
                local_path=weights_path,
                model_type=state.model,
                repo_id=repo_id,
                dict_path=DriverConfig.files.dictionary_file_path,
            )


def run_tune(
    state: State,
    *,
    mode: str,
    agent_config: str | None,
    tune_samples: int,
    tune_max_iterations: int,
    tune_output_dir: str,
    no_validate: bool,
    validation_threshold: float,
    max_retries: int,
    save: bool,
    push_to_hub: bool,
    balance_classes: bool,
    balance_seed: int,
    pos_weight: float,
) -> None:
    """Hyperparameter tuning (LLM-guided agent or standalone Ray Tune run)."""
    _ensure_ray_initialized()

    from sentimentizer.agent.skill import TuningRun, TuningRunConfig
    from sentimentizer.device import resolve_device

    device = resolve_device(state.device)

    config = TuningRunConfig(
        model_type=state.model,
        mode=mode,
        config_path=agent_config,
        num_samples=tune_samples,
        max_iterations=tune_max_iterations,
        device=device,
        save_best_model=save,
        output_dir=tune_output_dir,
        validate_predictions=not no_validate,
        validation_threshold=validation_threshold,
        max_retries=max_retries,
        balance_classes=balance_classes,
        balance_seed=balance_seed,
        pos_weight=pos_weight,
        push_to_hub=push_to_hub,
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


def run_hf_push(state: State, *, repo_id: str | None) -> None:
    """Push model weights to Hugging Face Hub."""
    from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig, weights_path_for
    from sentimentizer.hf import push_model_to_hub

    weights_path = weights_path_for(state.model)
    resolved_repo = repo_id if repo_id is not None else HF_WEIGHTS_REPOS.get(state.model)

    push_model_to_hub(
        local_path=weights_path,
        model_type=state.model,
        repo_id=resolved_repo,
        dict_path=DriverConfig.files.dictionary_file_path,
    )


def run_hf_pull(state: State, *, repo_id: str | None) -> None:
    """Pull model weights from Hugging Face Hub."""
    from sentimentizer.config import HF_WEIGHTS_REPOS, DriverConfig, weights_path_for
    from sentimentizer.hf import download_weights

    weights_path = weights_path_for(state.model)
    resolved_repo = repo_id if repo_id is not None else HF_WEIGHTS_REPOS.get(state.model)

    result_path = download_weights(
        model_type=state.model,
        local_path=weights_path,
        repo_id=resolved_repo,
        dict_path=DriverConfig.files.dictionary_file_path,
    )
    if result_path:
        logger.info(f"Pulled {state.model} weights from HF Hub to {result_path}")
    else:
        logger.error(f"Failed to pull {state.model} weights from HF Hub")


def run_diagnose_env(state: State) -> None:
    """Fast environment check — no torch/ray imports."""
    import platform

    logger.info(  # type: ignore[call-arg]
        "diagnose_env",
        python_version=platform.python_version(),
        platform=platform.platform(),
        model=state.model,
    )
    print("\n" + "=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print(f"  Python:       {platform.python_version()}")
    print(f"  Platform:     {platform.platform()}")
    print(f"  Model type:   {state.model}")
    print(f"  Run type:     {state.run_type}")
    print(f"  Device:       {state.device}")

    # Check NVIDIA LD_LIBRARY_PATH
    from sentimentizer.env import get_nvidia_ld_library_path

    nvidia_paths = get_nvidia_ld_library_path()
    if nvidia_paths:
        print(f"  NVIDIA lib:   {nvidia_paths[:80]}...")
    else:
        print("  NVIDIA lib:   (not found)")

    # Check Ray-related env vars
    ray_env_vars = [
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV",
        "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION",
        "RAY_GRAFANA_HOST",
        "RAY_PROMETHEUS_HOST",
    ]
    for var in ray_env_vars:
        val = os.environ.get(var, "(not set)")
        print(f"  {var}: {val}")

    # Check if torch/ray are importable (without importing them)
    torch_available = False
    ray_available = False
    try:
        import importlib.util

        torch_available = importlib.util.find_spec("torch") is not None
        ray_available = importlib.util.find_spec("ray") is not None
    except Exception:
        pass
    print(f"  torch:        {'available' if torch_available else 'NOT available'}")
    print(f"  ray:          {'available' if ray_available else 'NOT available'}")
    print("=" * 60 + "\n")


def run_diagnose_pipeline(state: State) -> None:
    """Heavy pipeline check. Imports the ML stack."""
    import json

    from sentimentizer.agent.skill import diagnose_training_issues

    logger.info("running_diagnostics", model_type=state.model)
    result = diagnose_training_issues(model_type=state.model)

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


# ──────────────────────────────────────────────
# 7. Shared decorator for train/run flags
# ──────────────────────────────────────────────


def shared_train_options(func: click.Command) -> click.Command:
    """Shared flags for ``train`` and ``run`` — keeps their surfaces in lockstep.

    ``--resume`` is intentionally NOT in this list because ``run`` exposes
    it as split flags (--resume-tokenize / --resume-train) while ``train``
    keeps the unified ``--resume``.
    """
    options = [
        click.option("--distributed", is_flag=True, help="Use Ray Train for distributed training"),
        click.option("--num-workers", default=1, type=int, help="Number of Ray Train workers"),
        click.option("--save/--no-save", default=False, help="Save model weights after training"),
        click.option("--checkpoint-dir", default="", type=str, help="Directory for checkpoints"),
        click.option(
            "--checkpoint-every", default=1, type=int, help="Save checkpoint every N epochs"
        ),
        click.option(
            "--balance-classes",
            is_flag=True,
            help="Enable class balancing (undersampling majority)",
        ),
        click.option(
            "--balance-seed", default=42, type=int, help="Random seed for class balancing"
        ),
        click.option(
            "--pos-weight",
            default=0.0,
            type=float,
            help="Loss weight for positive class (0 = auto-calculate)",
        ),
        click.option("--push-to-hub", is_flag=True, help="Push model weights to Hugging Face Hub"),
        click.option(
            "--pull-from-hub", is_flag=True, help="Pull model weights from Hugging Face Hub"
        ),
        click.option("--hf-repo", default=None, type=str, help="Hugging Face repository ID"),
    ]
    for option in reversed(options):
        func = option(func)  # type: ignore[misc]
    return func


# ──────────────────────────────────────────────
# 8. Click CLI definition
# ──────────────────────────────────────────────


@click.group()
@click.option(
    "--model",
    default="rnn",
    type=click.Choice(["rnn", "encoder", "decoder"]),
    envvar="SENTIMENTIZER_MODEL",
    help="Model architecture: rnn, encoder, or decoder",
)
@click.option(
    "--device",
    default="auto",
    envvar="SENTIMENTIZER_DEVICE",
    help="Compute device: auto, cuda, mps, or cpu (resolved lazily)",
)
@click.option(
    "--run-type",
    default="new",
    type=click.Choice(["new", "update"]),
    help="new = build/train fresh; update = reuse existing artifacts",
)
@click.pass_context
def cli(ctx: click.Context, model: str, device: str, run_type: str) -> None:
    """Sentimentizer ML pipeline."""
    # Store raw values only. Do NOT call auto_detect_device here or import
    # sentimentizer.config — it transitively imports torch, which would
    # trigger torch loading for every <subcommand> --help.
    ctx.obj = State(model=model, device=device, run_type=run_type)


# ── extract ──────────────────────────────────


@cli.command()
@click.option("--stop", default=10000, type=int, help="Number of review lines to load")
@click.pass_context
def extract(ctx: click.Context, stop: int) -> None:
    """Extract raw reviews into parquet."""
    run_extract(ctx.obj, stop=stop)


# ── tokenize ─────────────────────────────────


@cli.command()
@click.option(
    "--resume",
    is_flag=True,
    help="Update the existing dictionary from new data",
)
@click.pass_context
def tokenize(ctx: click.Context, resume: bool) -> None:
    """Build/update dictionary and write processed parquet."""
    run_tokenize(ctx.obj, resume=resume)


# ── train ────────────────────────────────────


@cli.command()
@shared_train_options
@click.option(
    "--resume",
    is_flag=True,
    help="Resume training from latest checkpoint in --checkpoint-dir",
)
@click.pass_context
def train(ctx: click.Context, resume: bool, **kwargs: Any) -> None:
    """Fit the model (single-node or distributed)."""
    run_train(ctx.obj, resume=resume, **kwargs)


# ── tune ─────────────────────────────────────


@cli.command()
@click.option(
    "--mode",
    default="agent",
    type=click.Choice(["agent", "standalone"]),
    help="Tuning mode: agent (LLM-guided loop) or standalone (Ray Tune)",
)
@click.option("--agent-config", default=None, type=str, help="Path to agent config YAML")
@click.option(
    "--samples", "tune_samples", default=20, type=int, help="Ray Tune trials per iteration"
)
@click.option(
    "--max-iterations",
    "tune_max_iterations",
    default=5,
    type=int,
    help="Maximum agent tuning iterations",
)
@click.option(
    "--output-dir",
    "tune_output_dir",
    default="tuning_results",
    type=str,
    help="Directory for tuning results",
)
@click.option("--no-validate", is_flag=True, help="Skip model prediction validation")
@click.option(
    "--validation-threshold",
    default=0.75,
    type=float,
    help="Minimum fraction of correct predictions to pass validation",
)
@click.option("--max-retries", default=2, type=int, help="Max re-tuning attempts on failure")
@click.option("--save/--no-save", default=False, help="Save best model after tuning")
@click.option("--push-to-hub", is_flag=True, help="Push model weights to Hugging Face Hub")
@click.option("--balance-classes", is_flag=True, help="Enable class balancing in training data")
@click.option("--balance-seed", default=42, type=int, help="Random seed for class balancing")
@click.option("--pos-weight", default=0.0, type=float, help="Loss weight for positive class")
@click.pass_context
def tune(ctx: click.Context, mode: str, **kwargs: Any) -> None:
    """Hyperparameter tuning (LLM-guided agent or standalone Ray Tune run)."""
    run_tune(ctx.obj, mode=mode, **kwargs)


# ── hf ───────────────────────────────────────


@cli.group()
def hf() -> None:
    """Hugging Face Hub operations."""


@hf.command("push")
@click.option("--repo-id", default=None, type=str, help="Hugging Face repository ID")
@click.pass_context
def hf_push(ctx: click.Context, repo_id: str | None) -> None:
    """Push model weights to Hugging Face Hub."""
    run_hf_push(ctx.obj, repo_id=repo_id)


@hf.command("pull")
@click.option("--repo-id", default=None, type=str, help="Hugging Face repository ID")
@click.pass_context
def hf_pull(ctx: click.Context, repo_id: str | None) -> None:
    """Pull model weights from Hugging Face Hub."""
    run_hf_pull(ctx.obj, repo_id=repo_id)


# ── diagnose ─────────────────────────────────


@cli.group()
def diagnose() -> None:
    """Pipeline diagnostics."""


@diagnose.command("env")
@click.pass_context
def diagnose_env(ctx: click.Context) -> None:
    """Fast environment check (Python / CUDA paths / Ray env vars). No torch/ray."""
    run_diagnose_env(ctx.obj)


@diagnose.command("pipeline")
@click.pass_context
def diagnose_pipeline(ctx: click.Context) -> None:
    """Heavy pipeline check. Imports the ML stack."""
    run_diagnose_pipeline(ctx.obj)


# ── run (full pipeline) ──────────────────────


@cli.command()
@click.option("--stop", default=10000, type=int, help="Number of review lines to load")
@click.option(
    "--resume-tokenize",
    is_flag=True,
    help="Update existing dictionary during tokenize stage",
)
@click.option(
    "--resume-train",
    is_flag=True,
    help="Resume training from checkpoint",
)
@shared_train_options
@click.pass_context
def run(
    ctx: click.Context,
    stop: int,
    resume_tokenize: bool,
    resume_train: bool,
    **train_kwargs: Any,
) -> None:
    """Run the full pipeline: extract → tokenize → train.

    Mirrors today's default ``python driver.py --type new`` invocation.
    The single legacy ``--resume`` flag is split into stage-specific flags
    so each stage's resume can be controlled independently.
    """
    run_extract(ctx.obj, stop=stop)
    run_tokenize(ctx.obj, resume=resume_tokenize)
    run_train(ctx.obj, resume=resume_train, **train_kwargs)


# ──────────────────────────────────────────────
# 9. Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cli()
