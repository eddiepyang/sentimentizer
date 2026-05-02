"""Ray Tune + Optuna hyperparameter search for sentimentizer models.

Translates the YAML-driven search space configuration into Ray Tune
search spaces, runs trials with ASHA scheduling, and returns the best
configuration found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ray import tune
from ray.tune import CLIReporter, RunConfig
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler, MedianStoppingRule
from ray.tune.search.optuna import OptunaSearch

from sentimentizer import new_logger
from sentimentizer.agent.loader import (
    AgentConfig,
    TunerConfig,
    load_agent_config,
    load_search_space,
)
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


def build_search_space(
    model_type: str,
    overrides: dict[str, dict[str, Any]] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Ray Tune search space dict from the YAML config.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        overrides: Optional dict of parameter overrides from the agent.
                   Keys are parameter names, values are search space dicts
                   (same format as YAML). These override the YAML defaults.
        config_path: Path to agent config YAML (uses default if None).

    Returns:
        Dict mapping parameter names to Ray Tune search space objects.
    """
    raw_space = load_search_space(model_type, config_path=config_path)

    # Apply agent overrides (narrowing/widening the search space)
    if overrides:
        for param_name, override_spec in overrides.items():
            if param_name in raw_space:
                raw_space[param_name] = override_spec

    search_space: dict[str, Any] = {}
    for param_name, spec in raw_space.items():
        param_type = spec["type"]

        if param_type == "loguniform":
            search_space[param_name] = tune.loguniform(spec["low"], spec["high"])
        elif param_type == "uniform":
            search_space[param_name] = tune.uniform(spec["low"], spec["high"])
        elif param_type == "choice":
            search_space[param_name] = tune.choice(spec["values"])
        elif param_type == "randint":
            search_space[param_name] = tune.randint(spec["low"], spec["high"])
        else:
            raise ValueError(f"Unknown search space type: {param_type}")

    return search_space


def _get_scheduler(
    tuner_config: TunerConfig,
) -> ASHAScheduler | HyperBandScheduler | MedianStoppingRule:
    """Create the appropriate Ray Tune scheduler from config."""
    if tuner_config.scheduler == "asha":
        return ASHAScheduler(
            time_attr="training_iteration",
            metric=tuner_config.metric,
            mode=tuner_config.mode,
            max_t=tuner_config.num_samples,
            grace_period=tuner_config.grace_period,
            reduction_factor=tuner_config.reduction_factor,
        )
    elif tuner_config.scheduler == "hyperband":
        return HyperBandScheduler(
            time_attr="training_iteration",
            metric=tuner_config.metric,
            mode=tuner_config.mode,
            max_t=tuner_config.num_samples,
            reduction_factor=tuner_config.reduction_factor,
        )
    elif tuner_config.scheduler == "median":
        return MedianStoppingRule(
            time_attr="training_iteration",
            metric=tuner_config.metric,
            mode=tuner_config.mode,
            grace_period=tuner_config.grace_period,
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {tuner_config.scheduler}. "
            f"Use 'asha', 'hyperband', or 'median'."
        )


def _trainable_wrapper(config: dict) -> None:
    """Ray Tune trainable that wraps sentimentizer training.

    This function is executed by each Ray Tune trial. It:
    1. Constructs a model with the trial's hyperparameters
    2. Trains for one epoch at a time (reporting metrics per epoch)
    3. Reports validation accuracy and loss back to Ray Tune

    The config dict must contain:
    - All model hyperparameters from the search space
    - 'model_type': str
    - 'device': str
    - 'dict_path': str
    - 'embeddings_file_path': str
    - 'embeddings_sub_file_path': str
    - 'embeddings_emb_length': int
    - 'input_len': int
    """
    from sentimentizer.config import DriverConfig, EmbeddingsConfig, TrainerConfig
    from sentimentizer.loader import load_train_val_corpus_datasets
    from sentimentizer.trainer import new_trainer

    model_type = config["model_type"]
    dict_path = config["dict_path"]
    embeddings_config = EmbeddingsConfig(
        file_path=config["embeddings_file_path"],
        sub_file_path=config["embeddings_sub_file_path"],
        emb_length=config["embeddings_emb_length"],
    )
    input_len = config["input_len"]
    device = config.get("device", "cpu")

    # Create model with trial hyperparameters
    model_config = _build_model_config(model_type, config)

    if model_type == "rnn":
        from sentimentizer.models.rnn import new_model
    elif model_type == "encoder":
        from sentimentizer.models.encoder import new_model
    elif model_type == "decoder":
        from sentimentizer.models.decoder import new_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = new_model(
        dict_path=dict_path,
        embeddings_config=embeddings_config,
        batch_size=config.get("batch_size", 64),
        input_len=input_len,
        model_config=model_config,
    )

    # Set up trainer with trial hyperparameters
    trainer_config = TrainerConfig(
        batch_size=config.get("batch_size", 64),
        epochs=config.get("epochs", 4),
        device=device,
        dataloader_workers=0,  # Use 0 for Ray Tune workers to avoid multiprocessing issues
    )

    trainer = new_trainer(
        model=model,
        cfg=trainer_config,
        model_type=model_type,
    )

    # Load data
    train_dataset, val_dataset = load_train_val_corpus_datasets(
        DriverConfig.files.processed_reviews_file_path
    )

    # Create data loaders
    from sentimentizer.trainer import _new_loaders

    train_loader, val_loader = _new_loaders(train_dataset, val_dataset, trainer_config)
    model.to(device)

    from sentimentizer.config import default_epochs

    epochs = config.get("epochs", default_epochs(model_type))
    best_val_loss = float("inf")

    for epoch in range(epochs):
        trainer._train_epoch(model, train_loader)  # noqa: SLF001
        trainer.eval(model, val_loader)

        # Compute validation accuracy
        val_accuracy = _compute_accuracy(model, val_loader, device)
        val_loss = trainer.val_loss

        # Track best loss for early stopping within the trial
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Report metrics to Ray Tune
        tune.report(
            val_accuracy=val_accuracy,
            val_loss=val_loss,
            train_loss=trainer.losses[-1] if trainer.losses else 0.0,
            epoch=epoch,
        )


def _build_model_config(
    model_type: str, config: dict
) -> Any:
    """Build a model config dataclass from the trial hyperparameters."""
    if model_type == "rnn":
        from sentimentizer.config import RNNConfig

        return RNNConfig(
            hidden_size=config.get("hidden_size", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
        )
    elif model_type == "encoder":
        from sentimentizer.config import EncoderConfig

        return EncoderConfig(
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            dropout=config.get("dropout", 0.2),
            ff_multiplier=config.get("ff_multiplier", 4),
        )
    elif model_type == "decoder":
        from sentimentizer.config import DecoderConfig

        return DecoderConfig(
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            n_encoder_layers=config.get("n_encoder_layers", 2),
            n_decoder_layers=config.get("n_decoder_layers", 4),
            dropout=config.get("dropout", 0.2),
            ff_multiplier=config.get("ff_multiplier", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _compute_accuracy(
    model: Any, dataloader: Any, device: str
) -> float:
    """Compute validation accuracy from a model and dataloader."""
    import torch

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs.to(device))
            preds = torch.sigmoid(outputs) >= 0.5
            targets_binary = targets.to(device) >= 0.5
            correct += (preds == targets_binary).sum().item()
            total += targets_binary.numel()
    return correct / max(total, 1)


def tune_model(
    model_type: str,
    agent_config: AgentConfig | None = None,
    tuner_config: TunerConfig | None = None,
    search_space_overrides: dict[str, dict[str, Any]] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run hyperparameter tuning using Ray Tune + Optuna.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        agent_config: Agent configuration (loaded from YAML if None).
        tuner_config: Tuner configuration (loaded from YAML if None).
        search_space_overrides: Optional overrides from the agent's TuningDecision.
        config_path: Path to agent config YAML.

    Returns:
        Dict with keys:
        - 'best_config': dict of best hyperparameters
        - 'best_accuracy': float
        - 'best_loss': float
        - 'trial_count': int
        - 'results': list of all trial results
    """
    # Load configs if not provided
    if agent_config is None or tuner_config is None:
        loaded_agent_cfg, loaded_tuner_cfg = load_agent_config(config_path)
        agent_config = agent_config or loaded_agent_cfg
        tuner_config = tuner_config or loaded_tuner_cfg

    # Build search space
    search_space = build_search_space(
        model_type,
        overrides=search_space_overrides,
        config_path=config_path,
    )

    # Add fixed config that trials need
    from sentimentizer.config import DriverConfig

    search_space["model_type"] = model_type
    search_space["device"] = tuner_config.mode  # will be overridden by device arg
    search_space["dict_path"] = DriverConfig.files.dictionary_file_path
    search_space["embeddings_file_path"] = DriverConfig.embeddings.file_path
    search_space["embeddings_sub_file_path"] = DriverConfig.embeddings.sub_file_path
    search_space["embeddings_emb_length"] = DriverConfig.embeddings.emb_length
    search_space["input_len"] = DriverConfig.tokenizer.max_len

    # Create scheduler and search algorithm
    scheduler = _get_scheduler(tuner_config)
    search_alg = OptunaSearch(metric=tuner_config.metric, mode=tuner_config.mode)

    # Reporter for CLI output
    reporter = CLIReporter(
        metric_columns=["val_accuracy", "val_loss", "train_loss", "epoch"],
    )

    logger.info(
        "starting_ray_tune",
        model_type=model_type,
        num_samples=tuner_config.num_samples,
        scheduler=tuner_config.scheduler,
        search_algorithm="optuna",
    )

    # Run Ray Tune
    result = tune.Tuner(
        tune.with_resources(
            _trainable_wrapper,
            resources={"cpu": 1, "gpu": 1 if _gpu_available() else 0},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=tuner_config.num_samples,
            metric=tuner_config.metric,
            mode=tuner_config.mode,
        ),
        run_config=RunConfig(
            progress_reporter=reporter,
            name=f"sentimentizer_{model_type}_tune",
        ),
    ).fit()

    # Extract best results
    best_result = result.get_best_result()
    best_config = best_result.config
    best_metrics = best_result.metrics

    # Remove internal config keys from best_config
    internal_keys = {
        "model_type", "device", "dict_path",
        "embeddings_file_path", "embeddings_sub_file_path",
        "embeddings_emb_length", "input_len",
    }
    clean_config = {k: v for k, v in best_config.items() if k not in internal_keys}

    # Collect all trial results
    all_results = []
    for trial_result in result:
        trial_config = {
            k: v for k, v in trial_result.config.items() if k not in internal_keys
        }
        all_results.append({
            "config": trial_config,
            "metrics": trial_result.metrics,
        })

    output = {
        "best_config": clean_config,
        "best_accuracy": best_metrics.get("val_accuracy", 0.0),
        "best_loss": best_metrics.get("val_loss", float("inf")),
        "trial_count": len(result),
        "results": all_results,
    }

    logger.info(
        "tuning_complete",
        best_accuracy=output["best_accuracy"],
        best_loss=output["best_loss"],
        trial_count=output["trial_count"],
        best_config=clean_config,
    )

    return output


def _gpu_available() -> bool:
    """Check if a GPU is available for Ray Tune trials."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
