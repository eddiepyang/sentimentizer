"""Configuration for the router training pipeline.

Default base model is BAAI/bge-base-en-v1.5 (109M params, 768-dim
embeddings, strong MTEB scores). Switch to mxbai-embed-large-v1
(335M params, ~1.3GB) only if evaluation fails to meet thresholds.

Defaults live in ``config.yaml`` next to this module. Loading order
mirrors ``sentimentizer.serve.config``: dataclass defaults < YAML
values < environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for the router training pipeline.

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


# Backward-compatible alias — existing code that references SetFitConfig
# will continue to work.
SetFitConfig = RouterConfig


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


# Environment-variable overrides. (env_var → (section, field, coerce_fn)).
# Pattern: SENTIMENTIZER_ROUTER_<FIELD> for training, SENTIMENTIZER_AUGMENT_<FIELD>
# for augmentation. Type coercion happens at parse time so errors surface early.
_ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "SENTIMENTIZER_ROUTER_BASE_MODEL": ("training", "base_model", str),
    "SENTIMENTIZER_ROUTER_NUM_ITERATIONS": ("training", "num_iterations", int),
    "SENTIMENTIZER_ROUTER_NUM_EPOCHS": ("training", "num_epochs", int),
    "SENTIMENTIZER_ROUTER_BATCH_SIZE": ("training", "batch_size", int),
    "SENTIMENTIZER_ROUTER_MAX_SEQ_LENGTH": ("training", "max_seq_length", int),
    "SENTIMENTIZER_ROUTER_SEED": ("training", "seed", int),
    "SENTIMENTIZER_ROUTER_OUTPUT_DIR": ("training", "output_dir", str),
    "SENTIMENTIZER_AUGMENT_MODEL": ("augmentation", "model", str),
    "SENTIMENTIZER_AUGMENT_OLLAMA_URL": ("augmentation", "ollama_url", str),
    "SENTIMENTIZER_AUGMENT_VARIATIONS_PER_SEED": ("augmentation", "variations_per_seed", int),
    "SENTIMENTIZER_AUGMENT_OUTPUT_PATH": ("augmentation", "output_path", str),
    "SENTIMENTIZER_AUGMENT_BATCH_SIZE": ("augmentation", "batch_size", int),
}


def _default_config_path() -> Path:
    """Return path to the default config.yaml bundled with the package."""
    return Path(__file__).parent / "config.yaml"


def load_router_config(
    path: str | Path | None = None,
) -> tuple[RouterConfig, AugmentConfig]:
    """Load router configuration from YAML with environment variable overrides.

    Precedence (highest wins):
        1. Environment variables (SENTIMENTIZER_ROUTER_*, SENTIMENTIZER_AUGMENT_*)
        2. YAML config file values
        3. Dataclass defaults

    Args:
        path: Path to YAML config file. If None, uses the bundled config.yaml.

    Returns:
        Tuple of (RouterConfig, AugmentConfig) with all settings populated.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = _default_config_path() if path is None else Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Router config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    sections: dict[str, dict[str, Any]] = {
        "training": dict(raw.get("training", {})),
        "augmentation": dict(raw.get("augmentation", {})),
    }

    for env_var, (section, field_name, coerce) in _ENV_OVERRIDES.items():
        env_value = os.environ.get(env_var)
        if env_value is None:
            continue
        try:
            sections[section][field_name] = coerce(env_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {env_var}={env_value!r}: expected {coerce.__name__}"
            ) from exc

    if "output_dir" in sections["training"]:
        sections["training"]["output_dir"] = Path(sections["training"]["output_dir"])

    return RouterConfig(**sections["training"]), AugmentConfig(**sections["augmentation"])
