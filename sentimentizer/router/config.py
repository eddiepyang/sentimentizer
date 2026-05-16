"""Configuration for the SetFit router training pipeline.

Default base model is BAAI/bge-base-en-v1.5 (109M params, 768-dim
embeddings, strong MTEB scores). Switch to mxbai-embed-large-v1
(335M params, ~1.3GB) only if evaluation fails to meet thresholds.

Config can be loaded from YAML (router/config.yaml) or constructed
directly via dataclass defaults. CLI flags override YAML values when
provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SetFitConfig:
    """Configuration for the SetFit router training pipeline.

    Attributes:
        base_model: HuggingFace model ID for the sentence-transformer backbone.
        num_iterations: Number of contrastive pairs generated per example.
        num_epochs: Number of fine-tuning epochs.
        batch_size: Training batch size.
        max_seq_length: Maximum token sequence length for the tokenizer.
        seed: Random seed for reproducibility.
        output_dir: Directory to save the trained model.
    """

    base_model: str = "BAAI/bge-base-en-v1.5"
    num_iterations: int = 20
    num_epochs: int = 1
    batch_size: int = 16
    max_seq_length: int = 512
    seed: int = 42
    output_dir: Path = Path("models/router")


@dataclass(frozen=True)
class AugmentConfig:
    """Configuration for the router augmentation pipeline (GLM 5.1 via Ollama).

    Attributes:
        model: Ollama model name for augmentation.
        ollama_url: Ollama API endpoint URL.
        variations_per_seed: Number of variations to generate per seed utterance.
        output_path: Default output JSONL file path.
        batch_size: Number of seeds to process per API batch.
    """

    model: str = "glm-5.1:cloud"
    ollama_url: str = "http://localhost:11434/api/generate"
    variations_per_seed: int = 50
    output_path: str = "augmented_yelp.jsonl"
    batch_size: int = 5


@dataclass(frozen=True)
class RouteLabels:
    """Route category labels for the Yelp review classifier.

    Categories:
        dietary (0): Food allergies, celiac, FODMAP, ingredient safety
        service (1): Wait times, staff behavior, reservation issues
        general (2): Ambiance, price, general food quality
    """

    dietary: int = 0
    service: int = 1
    general: int = 2

    @classmethod
    def label_names(cls) -> dict[int, str]:
        """Return mapping from label ID to name."""
        return {0: "dietary", 1: "service", 2: "general"}

    @classmethod
    def num_classes(cls) -> int:
        """Return the number of route categories."""
        return 3


def _get_default_config_path() -> Path:
    """Return path to the default config.yaml bundled with the package."""
    return Path(__file__).parent / "config.yaml"


def load_router_config(
    path: str | Path | None = None,
) -> tuple[SetFitConfig, AugmentConfig]:
    """Load router configuration from a YAML file.

    Args:
        path: Path to YAML config file. If None, uses the default
              config.yaml bundled with the package.

    Returns:
        Tuple of (SetFitConfig, AugmentConfig) with all settings populated.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = _get_default_config_path() if path is None else Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Router config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    training_dict = raw.get("training", {})
    # Convert output_dir string to Path
    if "output_dir" in training_dict:
        training_dict["output_dir"] = Path(training_dict["output_dir"])

    augmentation_dict = raw.get("augmentation", {})

    return SetFitConfig(**training_dict), AugmentConfig(**augmentation_dict)
