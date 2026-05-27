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
        sd_enabled: Enable Stable Diffusion 2.1 deployment.
        sd_model_id: HuggingFace model ID for SD.
        flux_enabled: Enable FLUX.1-dev deployment.
        flux_model_path: Path to FLUX weights (e.g. GGUF file).
        sd35_enabled: Enable SD 3.5 Medium deployment.
        sd35_model_id: HuggingFace model ID for SD 3.5 Medium.
        sd35_cpu_offload: Diffusers CPU offload mode for SD 3.5. One of
            "" (full GPU, default), "model" (whole-module swap), or
            "sequential" (submodule swap, lowest VRAM, slowest).
        sdxl_models: Named SDXL slots as "name:model_id" entries, e.g.
            ["anime:John6666/noob-sdxl-v10", "base:stabilityai/stable-diffusion-xl-base-1.0"].
            Each slot spawns its own GPU deployment and is addressable by name
            in image generation requests. Fits on an 11 GB card (~6.5 GB per slot).
        default_image_model: Default image model ("sd", "flux", "sd35", or any SDXL slot name).
        request_timeout_s: HTTP request timeout in seconds.
        api_keys: Comma-separated list of valid API keys (for image routes).
        rate_limit_per_min: Rate limit per API key per minute.
        rate_limit_burst: Token bucket burst size.
        idempotency_ttl_s: Cache TTL for idempotency keys.
        job_ttl_s: TTL in seconds before terminal job records are reaped.
        prompt_blocklist_path: Path to prompt blocklist file.
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
    sd_enabled: bool = False
    sd_model_id: str = "stabilityai/stable-diffusion-2-1"
    flux_enabled: bool = False
    flux_model_path: str = ""
    sd35_enabled: bool = False
    sd35_model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    sd35_cpu_offload: str = ""
    sdxl_models: list[str] = field(default_factory=list)
    default_image_model: str = "sd35"
    request_timeout_s: int = 600
    api_keys: list[str] = field(default_factory=list)
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 10
    idempotency_ttl_s: int = 600
    job_ttl_s: int = 3600
    prompt_blocklist_path: str = ""

    def __post_init__(self) -> None:
        """Validate that numeric fields are positive."""
        _positive_ints = {
            "max_batch_size": self.max_batch_size,
            "max_text_length": self.max_text_length,
            "predict_batch_size": self.predict_batch_size,
            "classify_batch_size": self.classify_batch_size,
        }
        _positive_floats = {
            "predict_batch_wait_s": self.predict_batch_wait_s,
            "classify_batch_wait_s": self.classify_batch_wait_s,
        }
        for name, val in _positive_ints.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        for name, val in _positive_floats.items():
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")


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
    "SENTIMENTIZER_SD_ENABLED": "sd_enabled",
    "SENTIMENTIZER_SD_MODEL_ID": "sd_model_id",
    "SENTIMENTIZER_FLUX_ENABLED": "flux_enabled",
    "SENTIMENTIZER_FLUX_MODEL_PATH": "flux_model_path",
    "SENTIMENTIZER_SD35_ENABLED": "sd35_enabled",
    "SENTIMENTIZER_SD35_MODEL_ID": "sd35_model_id",
    "SENTIMENTIZER_SD35_CPU_OFFLOAD": "sd35_cpu_offload",
    "SENTIMENTIZER_SDXL_MODELS": "sdxl_models",
    "SENTIMENTIZER_DEFAULT_IMAGE_MODEL": "default_image_model",
    "SENTIMENTIZER_REQUEST_TIMEOUT_S": "request_timeout_s",
    "SENTIMENTIZER_API_KEYS": "api_keys",
    "SENTIMENTIZER_RATE_LIMIT_PER_MIN": "rate_limit_per_min",
    "SENTIMENTIZER_RATE_LIMIT_BURST": "rate_limit_burst",
    "SENTIMENTIZER_IDEMPOTENCY_TTL_S": "idempotency_ttl_s",
    "SENTIMENTIZER_JOB_TTL_S": "job_ttl_s",
    "SENTIMENTIZER_PROMPT_BLOCKLIST_PATH": "prompt_blocklist_path",
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
    "sd_enabled": bool,
    "flux_enabled": bool,
    "sd35_enabled": bool,
    "sdxl_models": list,
    "request_timeout_s": int,
    "api_keys": list,
    "rate_limit_per_min": int,
    "rate_limit_burst": int,
    "idempotency_ttl_s": int,
    "job_ttl_s": int,
}


def parse_sdxl_models(entries: list[str]) -> dict[str, str]:
    """Parse ``["name:model_id", ...]`` into ``{name: model_id}``.

    Entries that don't contain a colon are silently skipped.
    """
    result: dict[str, str] = {}
    for entry in entries:
        name, sep, model_id = entry.partition(":")
        if sep and name.strip() and model_id.strip():
            result[name.strip()] = model_id.strip()
    return result


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
                try:
                    values[field_name] = coerce(env_value)
                except ValueError:
                    raise ValueError(
                        f"Invalid {env_var}={env_value!r}: " f"expected {coerce.__name__}"
                    ) from None

    return ServeConfig(**values)
