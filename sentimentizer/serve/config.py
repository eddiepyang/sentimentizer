"""Configuration for the Ray Serve deployment.

Loads defaults from serve_config.yaml, with environment variables
taking priority. Use ``ServeConfig.from_yaml()`` to load with the
standard precedence: YAML defaults < env vars.

Environment variables:
    SENTIMENTIZER_SERVE_CONFIG  -- Path to custom YAML config file
    SENTIMENTIZER_DEFAULT_MODEL -- Override default_model
    SENTIMENTIZER_MAX_BATCH_SIZE -- Override max_batch_size
    SENTIMENTIZER_MAX_TEXT_LENGTH -- Override max_text_length
    SENTIMENTIZER_PREDICT_BATCH_SIZE -- Override predict_batch_size
    SENTIMENTIZER_PREDICT_BATCH_WAIT_S -- Override predict_batch_wait_s
    SENTIMENTIZER_CLASSIFY_BATCH_SIZE -- Override classify_batch_size
    SENTIMENTIZER_CLASSIFY_BATCH_WAIT_S -- Override classify_batch_wait_s
    ROUTER_MODEL_PATH -- Override router_model_path
    SENTIMENTIZER_CORS_ORIGINS -- Override cors_origins (comma-separated)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ServeConfig:
    """Configuration for the Sentimentizer Ray Serve deployment.

    Attributes:
        default_model: Default sentiment model to load (rnn, encoder, decoder).
        router_model_path: Path to trained router model directory.
        max_batch_size: Maximum texts per /v1/batch request.
        max_text_length: Maximum characters per text input.
        predict_batch_size: Maximum requests collected per /v1/predict batch.
        predict_batch_wait_s: Seconds to wait before processing a partial batch.
        classify_batch_size: Maximum requests collected per /v1/router/predict batch.
        classify_batch_wait_s: Seconds to wait before processing a partial router batch.
        cors_origins: Allowed CORS origins (comma-separated for env var).
    """

    default_model: str = "encoder"
    router_model_path: str = "models/router"
    max_batch_size: int = 64
    max_text_length: int = 10000
    predict_batch_size: int = 32
    predict_batch_wait_s: float = 0.05
    classify_batch_size: int = 32
    classify_batch_wait_s: float = 0.05
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


def _parse_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


# Mapping from env var name to ServeConfig field name
_ENV_OVERRIDES: dict[str, str] = {
    "SENTIMENTIZER_DEFAULT_MODEL": "default_model",
    "ROUTER_MODEL_PATH": "router_model_path",
    "SENTIMENTIZER_MAX_BATCH_SIZE": "max_batch_size",
    "SENTIMENTIZER_MAX_TEXT_LENGTH": "max_text_length",
    "SENTIMENTIZER_PREDICT_BATCH_SIZE": "predict_batch_size",
    "SENTIMENTIZER_PREDICT_BATCH_WAIT_S": "predict_batch_wait_s",
    "SENTIMENTIZER_CLASSIFY_BATCH_SIZE": "classify_batch_size",
    "SENTIMENTIZER_CLASSIFY_BATCH_WAIT_S": "classify_batch_wait_s",
    "SENTIMENTIZER_CORS_ORIGINS": "cors_origins",
}

# Type coercion for non-string fields
# list[str] fields use _parse_list for comma-separated env var parsing
_FIELD_TYPES: dict[str, type] = {
    "max_batch_size": int,
    "max_text_length": int,
    "predict_batch_size": int,
    "predict_batch_wait_s": float,
    "classify_batch_size": int,
    "classify_batch_wait_s": float,
    "cors_origins": list,
}


def _default_config_path() -> Path:
    """Return path to the default serve_config.yaml bundled with the package."""
    return Path(__file__).parent / "serve_config.yaml"


def load_serve_config(path: str | Path | None = None) -> ServeConfig:
    """Load serve configuration from YAML with environment variable overrides.

    Precedence (highest wins):
        1. Environment variables (SENTIMENTIZER_DEFAULT_MODEL, etc.)
        2. YAML config file values
        3. Dataclass defaults

    Args:
        path: Path to YAML config file. If None, checks the
              SENTIMENTIZER_SERVE_CONFIG env var, then falls back to
              the bundled serve_config.yaml.

    Returns:
        A ServeConfig instance with resolved values.
    """
    if path is None:
        env_path = os.environ.get("SENTIMENTIZER_SERVE_CONFIG")
        path = Path(env_path) if env_path else _default_config_path()
    else:
        path = Path(path)

    # Start with dataclass defaults
    values: dict[str, object] = {}

    # Layer 1: YAML file overrides (if it exists)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        values.update(raw)

    # Layer 2: Environment variable overrides
    for env_var, field_name in _ENV_OVERRIDES.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            coerce = _FIELD_TYPES.get(field_name, str)
            if coerce is list:
                values[field_name] = _parse_list(env_value)
            else:
                values[field_name] = coerce(env_value)

    return ServeConfig(**values)
