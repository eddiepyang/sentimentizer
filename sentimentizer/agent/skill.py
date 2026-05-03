"""TuningRun skill: create a tuning run and work with the agent until convergence.

This module provides a high-level API for orchestrating hyperparameter
tuning runs. It supports two modes:

1. **Agent-guided** — uses the Pydantic AI + LangGraph agent loop
   (analyze → decide → tune → evaluate) to iteratively refine the
   search space until convergence or max iterations.

2. **Standalone** — runs a single Ray Tune + Optuna search without
   the LLM agent, useful for quick sweeps or when Ollama is unavailable.

Both modes produce a ``TuningRunResult`` with the best configuration,
metrics, and an optional trained model saved to disk.

The skill also includes model validation: after tuning, it trains a final
model with the best config and validates predictions against known
sentiment examples. If the model doesn't meet quality criteria, it can
re-tune with adjusted parameters.

Usage::

    from sentimentizer.agent.skill import TuningRun, TuningRunConfig

    config = TuningRunConfig(model_type="rnn")
    run = TuningRun(config)
    result = run.execute()
    print(f"Best accuracy: {result.best_accuracy:.4f}")
    print(f"Best config: {result.best_config}")
    print(f"Validation passed: {result.validation_passed}")
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentimentizer import new_logger
from sentimentizer.agent.loader import AgentConfig, load_agent_config
from sentimentizer.agent.models import AgentRunResult, TuningResult
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    DriverConfig,
    EmbeddingsConfig,
    TrainerConfig,
    auto_detect_device,
    default_epochs,
    weights_path_for,
)
from sentimentizer.tuner import TunerConfig, tune_model

logger = new_logger(DEFAULT_LOG_LEVEL)


# ---------------------------------------------------------------------------
# Known sentiment examples for validation
# ---------------------------------------------------------------------------

# These are used to verify the model produces sensible predictions.
# Each example has text and the expected sentiment direction:
#   positive → expect score > 0.5
#   negative → expect score < 0.5
KNOWN_SENTIMENT_EXAMPLES: list[dict[str, Any]] = [
    {"text": "amazing food great service", "expected": "positive"},
    {"text": "terrible experience worst ever", "expected": "negative"},
    {"text": "no good", "expected": "negative"},
    {"text": "loved it so much", "expected": "positive"},
    {"text": "awful and disgusting", "expected": "negative"},
    {"text": "best restaurant in town", "expected": "positive"},
    {"text": "would not recommend", "expected": "negative"},
    {"text": "fantastic place", "expected": "positive"},
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TuningRunConfig:
    """Configuration for a ``TuningRun``.

    Attributes:
        model_type: Model architecture to tune ('rnn', 'encoder', 'decoder').
        config_path: Path to agent config YAML. Uses default if None.
        mode: 'agent' for LLM-guided loop, 'standalone' for single Ray Tune run.
        max_iterations: Maximum agent loop iterations (agent mode only).
        convergence_threshold: Min accuracy improvement to continue (agent mode).
        num_samples: Number of Ray Tune trials per iteration.
        scheduler: Ray Tune scheduler ('asha', 'hyperband', 'median').
        device: Compute device ('auto', 'cpu', 'cuda', 'mps').
        save_best_model: Whether to train and save a final model with best config.
        output_dir: Directory to save results and model weights.
        search_space_overrides: Manual overrides for the YAML search space.
        stop: Number of data rows to load for training.
        validate_predictions: Whether to validate model predictions after training.
        validation_threshold: Minimum fraction of correct predictions to pass.
        max_retries: Maximum number of re-tuning attempts if validation fails.
    """

    model_type: str = "rnn"
    config_path: str | None = None
    mode: str = "agent"
    max_iterations: int = 5
    convergence_threshold: float = 0.005
    num_samples: int = 20
    scheduler: str = "asha"
    device: str = "auto"
    save_best_model: bool = True
    output_dir: str = "tuning_results"
    search_space_overrides: dict[str, dict[str, Any]] | None = None
    stop: int = 100000
    validate_predictions: bool = True
    validation_threshold: float = 0.75
    max_retries: int = 2
    balance_classes: bool = True
    balance_seed: int = 42
    pos_weight: float = 1.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TuningRunResult:
    """Result of a ``TuningRun.execute()`` call.

    Attributes:
        best_config: Best hyperparameter configuration found.
        best_accuracy: Best validation accuracy achieved.
        best_loss: Best validation loss achieved.
        best_precision: Best positive-class precision achieved.
        best_recall: Best positive-class recall achieved.
        best_f1: Best positive-class F1 score achieved.
        best_cohen_kappa: Best Cohen's kappa coefficient achieved.
        best_positive_accuracy: Best accuracy on positive samples.
        best_negative_accuracy: Best accuracy on negative samples.
        iterations_completed: Number of tuning iterations completed.
        converged: Whether the agent converged before max iterations.
        history: List of TuningResult dicts from each iteration.
        model_path: Path to the saved model weights (empty string if not saved).
        results_path: Path to the saved JSON results file.
        elapsed_seconds: Wall-clock time for the entire run.
        validation_passed: Whether model predictions passed validation.
        validation_results: Detailed validation results per example.
        validation_metrics: Classification metrics from model validation.
        retry_count: Number of re-tuning attempts due to failed validation.
    """

    best_config: dict[str, Any] = field(default_factory=dict)
    best_accuracy: float = 0.0
    best_loss: float = float("inf")
    best_precision: float = 0.0
    best_recall: float = 0.0
    best_f1: float = 0.0
    best_cohen_kappa: float = 0.0
    best_positive_accuracy: float = 0.0
    best_negative_accuracy: float = 0.0
    iterations_completed: int = 0
    converged: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    model_path: str = ""
    results_path: str = ""
    elapsed_seconds: float = 0.0
    validation_passed: bool = False
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    validation_metrics: dict[str, Any] | None = None
    retry_count: int = 0


# ---------------------------------------------------------------------------
# TuningRun
# ---------------------------------------------------------------------------


class TuningRun:
    """Orchestrate a hyperparameter tuning run from start to finish.

    This is the main entry point for creating a tuning run and working
    with the agent until a good model is found. It handles:

    - Loading configuration from YAML
    - Running agent-guided or standalone tuning
    - Training a final model with the best configuration
    - Validating model predictions against known sentiment examples
    - Re-tuning if validation fails (up to max_retries)
    - Saving results and model weights to disk

    Args:
        config: A ``TuningRunConfig`` with run settings.

    Example::

        run = TuningRun(TuningRunConfig(model_type="encoder", mode="agent"))
        result = run.execute()
        if result.validation_passed:
            print(f"Model ready! Accuracy: {result.best_accuracy:.4f}")
    """

    def __init__(self, config: TuningRunConfig) -> None:
        self.config = config
        self._resolve_device()
        self._output_dir = Path(config.output_dir)
        self._agent_config: AgentConfig | None = None
        self._tuner_config: TunerConfig | None = None

    def _resolve_device(self) -> None:
        """Resolve 'auto' device to the best available."""
        if self.config.device == "auto":
            self.config.device = auto_detect_device()
        logger.info("device_resolved", device=self.config.device)

    def _load_configs(self) -> tuple[AgentConfig, TunerConfig]:
        """Load agent and tuner configs, applying overrides from TuningRunConfig."""
        agent_config, tuner_config = load_agent_config(self.config.config_path)

        # Apply TuningRunConfig overrides to the agent config
        if self.config.max_iterations != 5:  # non-default
            agent_config.max_iterations = self.config.max_iterations
        if self.config.convergence_threshold != 0.005:  # non-default
            agent_config.convergence_threshold = self.config.convergence_threshold

        # Apply TuningRunConfig overrides to the tuner config
        if self.config.num_samples != 20:  # non-default
            tuner_config.num_samples = self.config.num_samples
        if self.config.scheduler != "asha":  # non-default
            tuner_config.scheduler = self.config.scheduler

        self._agent_config = agent_config
        self._tuner_config = tuner_config
        return agent_config, tuner_config

    def execute(self) -> TuningRunResult:
        """Execute the full tuning run with validation and retry logic.

        Runs either agent-guided or standalone tuning based on
        ``self.config.mode``, then optionally trains a final model
        with the best configuration found, validates predictions,
        and re-tunes if validation fails.

        Returns:
            ``TuningRunResult`` with best config, metrics, and file paths.
        """
        start_time = time.time()
        logger.info(
            "tuning_run_start",
            model_type=self.config.model_type,
            mode=self.config.mode,
            device=self.config.device,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)

        retry_count = 0
        result = self._run_tuning()

        # Train final model and validate predictions
        if self.config.save_best_model and result.best_accuracy > 0.0:
            while retry_count <= self.config.max_retries:
                model_path = self._train_final_model(result.best_config)
                result.model_path = str(model_path)

                if self.config.validate_predictions:
                    validation_passed, validation_results, validation_metrics = (
                        self._validate_model(model_path)
                    )
                    result.validation_passed = validation_passed
                    result.validation_results = validation_results
                    result.validation_metrics = validation_metrics

                    if validation_passed:
                        logger.info(
                            "validation_passed",
                            model_type=self.config.model_type,
                            retry_count=retry_count,
                        )
                        break

                    logger.warning(
                        "validation_failed",
                        model_type=self.config.model_type,
                        retry_count=retry_count,
                        max_retries=self.config.max_retries,
                    )
                    retry_count += 1

                    if retry_count <= self.config.max_retries:
                        # Re-tune with narrowed search space
                        result = self._run_tuning()
                        if result.best_accuracy <= 0.0:
                            logger.warning("re_tuning_produced_no_results")
                            break
                else:
                    # No validation requested — mark as passed
                    result.validation_passed = True
                    break

        result.retry_count = retry_count

        # Copy best weights to model-specific path if validation passed
        if result.validation_passed and result.model_path:
            model_weights_path = Path(weights_path_for(self.config.model_type))
            try:
                model_weights_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(result.model_path, model_weights_path)
                logger.info(
                    "weights_copied_to_default",
                    source=result.model_path,
                    destination=str(model_weights_path),
                )
            except OSError as e:
                logger.warning(f"failed_to_copy_weights: {e}")

        # Save results to JSON
        results_path = self._save_results(result)
        result.results_path = str(results_path)

        result.elapsed_seconds = time.time() - start_time
        logger.info(
            "tuning_run_complete",
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
            elapsed_seconds=result.elapsed_seconds,
        )

        return result

    # ------------------------------------------------------------------
    # Tuning dispatch
    # ------------------------------------------------------------------

    def _run_tuning(self) -> TuningRunResult:
        """Run the appropriate tuning mode (agent or standalone)."""
        if self.config.mode == "agent":
            return self._run_agent_guided()
        elif self.config.mode == "standalone":
            return self._run_standalone()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode!r}. Use 'agent' or 'standalone'.")

    # ------------------------------------------------------------------
    # Agent-guided mode
    # ------------------------------------------------------------------

    def _run_agent_guided(self) -> TuningRunResult:
        """Run the full agent-guided tuning loop.

        Uses the LangGraph state graph (analyze → decide → tune → evaluate)
        with Pydantic AI agents powered by GLM 5.1 via Ollama.
        """
        import asyncio

        from sentimentizer.agent.graph import run_agent_tuning

        agent_config, tuner_config = self._load_configs()

        logger.info(
            "starting_agent_guided_tuning",
            model_type=self.config.model_type,
            max_iterations=agent_config.max_iterations,
            convergence_threshold=agent_config.convergence_threshold,
            num_samples=tuner_config.num_samples,
        )

        agent_result: AgentRunResult = asyncio.run(
            run_agent_tuning(
                model_type=self.config.model_type,
                config_path=self.config.config_path,
            )
        )

        return TuningRunResult(
            best_config=agent_result.best_config,
            best_accuracy=agent_result.best_accuracy,
            best_loss=agent_result.best_loss,
            iterations_completed=agent_result.iterations_completed,
            converged=agent_result.converged,
            history=[
                h.model_dump() if hasattr(h, "model_dump") else h for h in agent_result.history
            ],
        )

    # ------------------------------------------------------------------
    # Standalone mode
    # ------------------------------------------------------------------

    def _run_standalone(self) -> TuningRunResult:
        """Run a single Ray Tune + Optuna search without the LLM agent.

        This is useful for quick sweeps or when Ollama is not available.
        """
        _, tuner_config = self._load_configs()

        logger.info(
            "starting_standalone_tuning",
            model_type=self.config.model_type,
            num_samples=tuner_config.num_samples,
            scheduler=tuner_config.scheduler,
        )

        result = tune_model(
            model_type=self.config.model_type,
            tuner_config=tuner_config,
            search_space_overrides=self.config.search_space_overrides,
            config_path=self.config.config_path,
            balance_classes=self.config.balance_classes,
            balance_seed=self.config.balance_seed,
            pos_weight=self.config.pos_weight,
        )

        tuning_result = TuningResult(
            best_accuracy=result["best_accuracy"],
            best_loss=result["best_loss"],
            best_config=result["best_config"],
            trial_count=result["trial_count"],
            improvement_over_last=result["best_accuracy"],
        )

        return TuningRunResult(
            best_config=result["best_config"],
            best_accuracy=result["best_accuracy"],
            best_loss=result["best_loss"],
            best_precision=result.get("best_precision", 0.0),
            best_recall=result.get("best_recall", 0.0),
            best_f1=result.get("best_f1", 0.0),
            best_cohen_kappa=result.get("best_cohen_kappa", 0.0),
            best_positive_accuracy=result.get("best_positive_accuracy", 0.0),
            best_negative_accuracy=result.get("best_negative_accuracy", 0.0),
            iterations_completed=1,
            converged=True,  # single run is always "converged"
            history=[tuning_result.model_dump()],
        )

    # ------------------------------------------------------------------
    # Final model training
    # ------------------------------------------------------------------

    def _train_final_model(self, best_config: dict[str, Any]) -> Path:
        """Train a final model with the best configuration found.

        Creates a model using the best hyperparameters from the tuning
        run, trains it on the full dataset, and saves the weights.

        Args:
            best_config: Best hyperparameter configuration from tuning.

        Returns:
            Path to the saved model weights.
        """
        import torch

        from sentimentizer.loader import load_train_val_corpus_datasets
        from sentimentizer.trainer import new_trainer

        model_type = self.config.model_type
        device = self.config.device

        logger.info(
            "training_final_model",
            model_type=model_type,
            device=device,
            best_config=best_config,
        )

        # Build model config from best hyperparameters
        model_config = _build_model_config(model_type, best_config)

        # Create model
        model = _create_model(model_type, model_config)
        model.to(device)

        # Determine epochs — use more epochs for final training
        epochs = default_epochs(model_type)
        # For final training, use 2x the default epochs for better convergence
        final_epochs = epochs * 2

        trainer_config = TrainerConfig(
            epochs=final_epochs,
            device=device,
            early_stopping_patience=3,
            checkpoint_dir=str(self._output_dir / "checkpoints"),
            checkpoint_every=1,
            checkpoint_best=True,
            pos_weight=self.config.pos_weight,
        )

        trainer = new_trainer(
            model=model,
            cfg=trainer_config,
            model_type=model_type,
        )

        # Load data
        train_dataset, val_dataset = load_train_val_corpus_datasets(
            DriverConfig.files.processed_reviews_file_path,
            balance_classes=self.config.balance_classes,
            random_state=self.config.balance_seed,
        )

        # Train
        trainer.fit(model, train_data=train_dataset, val_data=val_dataset)

        # Save model weights
        model_path = self._output_dir / f"best_model_{model_type}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"final_model_saved: {model_path}")

        return model_path

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    def _validate_model(self, model_path: str) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
        """Validate model predictions against known sentiment examples.

        Loads the trained model and tests it against a set of known
        positive and negative phrases. A prediction is correct if:
        - For positive text: model output > 0.5
        - For negative text: model output < 0.5

        Also computes comprehensive classification metrics including
        per-class accuracy, precision/recall/F1, Cohen's kappa, and
        AUC-ROC.

        Args:
            model_path: Path to the saved model weights.

        Returns:
            Tuple of (passed, results, metrics) where passed is True if
            the fraction of correct predictions >= validation_threshold,
            results is a list of per-example validation details, and
            metrics is a dict of ClassificationMetrics fields.
        """
        import torch

        from sentimentizer.metrics import compute_metrics_from_examples

        model_type = self.config.model_type
        device = self.config.device

        logger.info(
            "validating_model",
            model_type=model_type,
            model_path=model_path,
            threshold=self.config.validation_threshold,
        )

        # Load tokenizer
        from sentimentizer.tokenizer import get_trained_tokenizer

        try:
            tokenizer = get_trained_tokenizer()
        except Exception as e:
            logger.warning(f"tokenizer_load_failed: {e}")
            return False, [{"error": str(e)}], {}

        # Load model
        try:
            model = _load_trained_model(model_type, model_path, device)
        except Exception as e:
            logger.warning(f"model_load_failed: {e}")
            return False, [{"error": str(e)}], {}

        model.to(device)
        model.eval()

        results: list[dict[str, Any]] = []
        correct = 0

        for example in KNOWN_SENTIMENT_EXAMPLES:
            text = example["text"]
            expected = example["expected"]

            # Tokenize
            token_ids = tokenizer.tokenize_text(text)
            input_tensor = torch.from_numpy(token_ids).long().to(device)

            # Predict
            with torch.no_grad():
                logit = model(input_tensor)
                score = torch.sigmoid(logit).item()

            # Determine if prediction is correct
            is_correct = score > 0.5 if expected == "positive" else score < 0.5

            if is_correct:
                correct += 1

            results.append(
                {
                    "text": text,
                    "expected": expected,
                    "score": round(score, 4),
                    "correct": is_correct,
                }
            )

            logger.info(
                "validation_example",
                text=text,
                expected=expected,
                score=round(score, 4),
                correct=is_correct,
            )

        accuracy = correct / len(KNOWN_SENTIMENT_EXAMPLES)
        passed = accuracy >= self.config.validation_threshold

        # Compute comprehensive classification metrics
        metrics_obj = compute_metrics_from_examples(results)
        metrics_dict = metrics_obj.to_dict()

        logger.info(
            "validation_summary",
            correct=correct,
            total=len(KNOWN_SENTIMENT_EXAMPLES),
            accuracy=round(accuracy, 4),
            passed=passed,
            f1=metrics_obj.f1,
            cohen_kappa=metrics_obj.cohen_kappa,
            positive_accuracy=metrics_obj.positive_accuracy,
            negative_accuracy=metrics_obj.negative_accuracy,
        )

        return passed, results, metrics_dict

    # ------------------------------------------------------------------
    # Results persistence
    # ------------------------------------------------------------------

    def _save_results(self, result: TuningRunResult) -> Path:
        """Save tuning results to a JSON file.

        Args:
            result: The ``TuningRunResult`` to save.

        Returns:
            Path to the saved JSON file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        results_path = self._output_dir / f"tuning_results_{self.config.model_type}.json"

        output = {
            "model_type": self.config.model_type,
            "mode": self.config.mode,
            "device": self.config.device,
            "best_config": result.best_config,
            "best_accuracy": result.best_accuracy,
            "best_loss": result.best_loss,
            "best_precision": result.best_precision,
            "best_recall": result.best_recall,
            "best_f1": result.best_f1,
            "best_cohen_kappa": result.best_cohen_kappa,
            "best_positive_accuracy": result.best_positive_accuracy,
            "best_negative_accuracy": result.best_negative_accuracy,
            "iterations_completed": result.iterations_completed,
            "converged": result.converged,
            "history": result.history,
            "model_path": result.model_path,
            "validation_passed": result.validation_passed,
            "validation_results": result.validation_results,
            "validation_metrics": result.validation_metrics,
            "retry_count": result.retry_count,
            "elapsed_seconds": result.elapsed_seconds,
        }

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"results_saved: {results_path}")
        return results_path

    @staticmethod
    def load_results(path: str | Path) -> dict[str, Any]:
        """Load tuning results from a JSON file.

        Args:
            path: Path to the JSON results file.

        Returns:
            Dict with the saved tuning results.
        """
        path = Path(path)
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Module-level helpers (shared with tuner.py)
# ---------------------------------------------------------------------------


def _build_model_config(model_type: str, config: dict[str, Any]) -> Any:
    """Build a model config dataclass from the best hyperparameters."""
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


def _create_model(model_type: str, model_config: Any) -> Any:
    """Create a model instance from the model type and config."""
    embeddings_config = EmbeddingsConfig()
    dict_path = DriverConfig.files.dictionary_file_path
    input_len = DriverConfig.tokenizer.max_len

    if model_type == "rnn":
        from sentimentizer.models.rnn import new_model

        return new_model(
            dict_path=dict_path,
            embeddings_config=embeddings_config,
            input_len=input_len,
            model_config=model_config,
        )
    elif model_type == "encoder":
        from sentimentizer.models.encoder import new_model

        return new_model(
            dict_path=dict_path,
            embeddings_config=embeddings_config,
            input_len=input_len,
            model_config=model_config,
        )
    elif model_type == "decoder":
        from sentimentizer.models.decoder import new_model

        return new_model(
            dict_path=dict_path,
            embeddings_config=embeddings_config,
            input_len=input_len,
            model_config=model_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_trained_model(model_type: str, model_path: str, device: str) -> Any:
    """Load a trained model from saved weights.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        model_path: Path to the saved model weights (.pth).
        device: Device to load the model onto.

    Returns:
        Model with loaded weights in eval mode.
    """
    import torch

    weights = torch.load(model_path, map_location=device, weights_only=True)

    # Infer model dimensions from weights
    emb_shape = weights["embed_layer.weight"].shape

    if model_type == "rnn":
        from sentimentizer.models.rnn import RNN

        hidden_size = weights["classifier.0.weight"].shape[1] // 2
        model = RNN(
            emb_weights=torch.zeros(emb_shape),
            hidden_size=hidden_size,
        )
    elif model_type == "encoder":
        from sentimentizer.models.encoder import Encoder

        d_model = weights["proj.weight"].shape[0]
        model = Encoder(
            input_len=200,
            emb_weights=torch.zeros(emb_shape),
            d_model=d_model,
        )
    elif model_type == "decoder":
        from sentimentizer.models.decoder import Decoder

        d_model = weights["proj.weight"].shape[0]
        model = Decoder(
            input_len=200,
            emb_weights=torch.zeros(emb_shape),
            d_model=d_model,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(weights)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def create_tuning_run(
    model_type: str = "rnn",
    mode: str = "agent",
    config_path: str | None = None,
    num_samples: int = 20,
    max_iterations: int = 5,
    device: str = "auto",
    save_best_model: bool = True,
    output_dir: str = "tuning_results",
    search_space_overrides: dict[str, dict[str, Any]] | None = None,
    validate_predictions: bool = True,
    validation_threshold: float = 0.75,
    max_retries: int = 2,
    balance_classes: bool = True,
    balance_seed: int = 42,
) -> TuningRunResult:
    """Create and execute a tuning run with sensible defaults.

    This is the simplest way to start a tuning run. It creates a
    ``TuningRunConfig``, builds a ``TuningRun``, and executes it.

    The skill works with the agent until a good model is found:
    it tunes hyperparameters, trains a final model, validates
    predictions, and re-tunes if validation fails.

    Args:
        model_type: Model architecture ('rnn', 'encoder', 'decoder').
        mode: 'agent' for LLM-guided loop, 'standalone' for single search.
        config_path: Path to agent config YAML (uses default if None).
        num_samples: Number of Ray Tune trials per iteration.
        max_iterations: Max agent loop iterations (agent mode).
        device: Compute device ('auto', 'cpu', 'cuda', 'mps').
        save_best_model: Whether to train and save a final model.
        output_dir: Directory to save results and model weights.
        search_space_overrides: Manual overrides for the search space.
        validate_predictions: Whether to validate model predictions.
        validation_threshold: Minimum fraction of correct predictions.
        max_retries: Max re-tuning attempts if validation fails.
        balance_classes: Whether to balance classes by undersampling (default: True).
        balance_seed: Random seed for class balancing (default: 42).

    Returns:
        ``TuningRunResult`` with best config, metrics, and file paths.

    Example::

        # Agent-guided tuning with validation
        result = create_tuning_run(model_type="rnn", mode="agent")
        if result.validation_passed:
            print(f"Model ready! Accuracy: {result.best_accuracy:.4f}")

        # Quick standalone sweep
        result = create_tuning_run(model_type="encoder", mode="standalone")
    """
    config = TuningRunConfig(
        model_type=model_type,
        mode=mode,
        config_path=config_path,
        num_samples=num_samples,
        max_iterations=max_iterations,
        device=device,
        save_best_model=save_best_model,
        output_dir=output_dir,
        search_space_overrides=search_space_overrides,
        validate_predictions=validate_predictions,
        validation_threshold=validation_threshold,
        max_retries=max_retries,
        balance_classes=balance_classes,
        balance_seed=balance_seed,
    )
    run = TuningRun(config)
    return run.execute()
