"""Train stage: fit the model (single-node or distributed via Ray Train).

Heavy imports (torch, ray, config) are DEFERRED to function bodies to avoid
importing the ML stack at module level. Do NOT add module-level imports
of torch, ray, gensim, or sentimentizer.config here.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from workflows.helpers import _load_model
from workflows.lifecycle import State, _cuda_cleanup, _ensure_ray_initialized, logger

_METRICS_PERSISTENCE_PATH = Path("/tmp/sentimentizer_training_metrics.json")


def _reset_stale_metrics(model_type: str) -> None:
    """Clear stale persisted metrics and reset Prometheus gauges for *model_type*.

    When training starts for a given model type, any residual metrics from a
    previous run (persisted in the JSON file and cached in prometheus_client
    gauges) would otherwise appear on the dashboard as "current" data.  This
    function:

    1. Writes a zeroed-out entry for *model_type* in the persisted JSON file
       so the standalone exporter pushes zeroes to Prometheus on its next
       scrape cycle, overwriting the stale values.
    2. Resets the driver-level ``prometheus_client`` gauges for *model_type*
       to 0 so a Prometheus scrape that happens mid-training doesn't show the
       old values.  (The gauges are re-populated during training.)
    3. Invalidates the Ray ``_RAY_GAUGES`` cache entry for *model_type* so a
       fresh gauge is lazily created in the current Ray worker context.
    """
    zeroed_metrics = {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "cohen_kappa": 0.0,
        "auc_roc": None,
        "positive_accuracy": 0.0,
        "negative_accuracy": 0.0,
        "epoch": 0,
    }

    # --- 1. Persisted JSON file ---
    # Write a zeroed entry instead of deleting: the standalone exporter only
    # updates gauge labels for model types present in the file.  If we delete
    # the entry, the exporter's in-process gauges keep serving the last value.
    try:
        data: dict = {}
        if _METRICS_PERSISTENCE_PATH.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                data = json.loads(_METRICS_PERSISTENCE_PATH.read_text())
        if not isinstance(data, dict):
            data = {}
        data[model_type] = zeroed_metrics
        _METRICS_PERSISTENCE_PATH.write_text(json.dumps(data, indent=2))
        logger.info(
            "stale_metrics_cleared",
            path=str(_METRICS_PERSISTENCE_PATH),
            model_type=model_type,
        )
    except OSError as exc:
        logger.warning("failed_to_clear_stale_metrics_file: %s", exc)

    # --- 2. prometheus_client gauges (driver process) ---
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

        lbl = {"model_type": model_type}
        for gauge in (
            TRAINING_TRAIN_LOSS,
            TRAINING_VAL_LOSS,
            TRAINING_VAL_ACCURACY,
            TRAINING_VAL_PRECISION,
            TRAINING_VAL_RECALL,
            TRAINING_VAL_F1,
            TRAINING_VAL_COHEN_KAPPA,
            TRAINING_VAL_POSITIVE_ACCURACY,
            TRAINING_VAL_NEGATIVE_ACCURACY,
            TRAINING_EPOCH,
        ):
            gauge.labels(**lbl).set(0)
        # AUC-ROC uses None / no value as sentinel — clear it
        TRAINING_VAL_AUC_ROC.labels(**lbl).set(0)
    except ImportError:
        pass

    # --- 3. Ray _RAY_GAUGES cache ---
    try:
        from sentimentizer.trainer import _RAY_GAUGES

        if model_type in _RAY_GAUGES:
            del _RAY_GAUGES[model_type]
    except ImportError:
        pass


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

    # Clear stale metrics from previous training runs for this model type
    # so the dashboard doesn't show residual data from other model types.
    _reset_stale_metrics(state.model)

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
        from workflows.lifecycle import is_ray_available

        if not is_ray_available():
            raise ImportError(
                "Ray is required for distributed training. Install with: pip install '.[ray]'"
            )
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
