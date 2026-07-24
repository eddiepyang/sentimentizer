"""Configuration for the Ray Serve deployment.

Loads defaults from service.yaml, with environment variables
taking priority. Use ``ServeConfig.from_yaml()`` to load with the
standard precedence: YAML defaults < env vars.

Environment variables:
    SENTIMENTIZER_SERVE_CONFIG  -- Path to custom YAML config file
    SENTIMENTIZER_SERVE_HOST -- Override serve_host
    SENTIMENTIZER_SERVE_PORT -- Override serve_port
    SENTIMENTIZER_RAY_OBJECT_STORE_MEMORY_MB -- Override ray_object_store_memory_mb
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
        serve_host: Address used by the Ray Serve HTTP proxy.
        serve_port: Port used by the Ray Serve HTTP proxy.
        ray_object_store_memory_mb: Ray object-store allocation in MiB.
        default_model: Default sentiment model to load (rnn, encoder, decoder).
        router_model_path: Path to trained router model directory.
        max_batch_size: Maximum texts per /v1/batch request.
        max_text_length: Maximum characters per text input.
        predict_batch_size: Maximum requests collected per /v1/predict batch.
        predict_batch_wait_s: Seconds to wait before processing a partial batch.
        classify_batch_size: Maximum requests collected per /v1/router/predict batch.
        classify_batch_wait_s: Seconds to wait before processing a partial router batch.
        cors_origins: Allowed CORS origins (comma-separated for env var).
        krea_2_enabled: Enable Krea 2 Turbo through headless ComfyUI.
        krea_2_license_accepted: Confirms the operator has accepted the Krea 2
            Community License and is eligible for the intended use.
        ideogram_4_enabled: Enable Ideogram 4 through headless ComfyUI.
        ideogram_4_license_accepted: Confirms the operator has accepted the
            Ideogram 4 non-commercial model license.
        comfyui_base_url: URL of the separately managed headless ComfyUI process.
        comfyui_temp_directory: Shared path to ComfyUI's temporary output directory.
        image_moderation_url: Operator-provided output moderation endpoint.
        image_moderation_api_key: Optional bearer credential for that endpoint.
        image_moderation_timeout_s: Output moderation request timeout.
        default_image_model: Default image model ("krea_2" or "ideogram_4").
        request_timeout_s: HTTP request timeout in seconds.
        api_keys: Comma-separated list of valid API keys (for image routes).
        rate_limit_per_min: Rate limit per API key per minute.
        rate_limit_burst: Token bucket burst size.
        idempotency_ttl_s: Cache TTL for idempotency keys.
        job_ttl_s: TTL in seconds before terminal job records are reaped.
        prompt_blocklist_path: Path to prompt blocklist file.
        bge_m3_num_replicas: Number of BGE-M3 Ray Serve replicas.
        bge_m3_max_ongoing_requests: Maximum concurrent requests per BGE-M3 replica.
        bge_m3_num_cpus: Ray CPU reservation per BGE-M3 replica.
        bge_m3_num_gpus: Ray GPU reservation per BGE-M3 replica.
    """

    serve_host: str = "0.0.0.0"
    serve_port: int = 8000
    ray_object_store_memory_mb: int = 384
    default_model: str = "encoder"
    router_model_path: str = "models/router"
    max_batch_size: int = 64
    max_text_length: int = 10000
    predict_batch_size: int = 32
    predict_batch_wait_s: float = 0.05
    classify_batch_size: int = 32
    classify_batch_wait_s: float = 0.05
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    krea_2_enabled: bool = False
    krea_2_license_accepted: bool = False
    ideogram_4_enabled: bool = False
    ideogram_4_license_accepted: bool = False
    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_timeout_s: float = 600.0
    comfyui_poll_interval_s: float = 0.25
    comfyui_temp_directory: str = ""
    image_moderation_url: str = ""
    image_moderation_api_key: str = ""
    image_moderation_timeout_s: float = 10.0
    default_image_model: str = "krea_2"
    request_timeout_s: int = 600
    api_keys: list[str] = field(default_factory=list)
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 10
    idempotency_ttl_s: int = 600
    job_ttl_s: int = 3600
    prompt_blocklist_path: str = ""
    embeddings_enabled: bool = False
    embeddings_device: str = "auto"
    dense_embedding_model_id: str = "nomic-ai/nomic-embed-text-v1.5"
    dense_embedding_revision: str = "161a1515eaa28482dfe6ceae96cdefdd20dabcbd"
    bge_m3_enabled: bool = False
    bge_m3_model_id: str = "BAAI/bge-m3"
    bge_m3_use_fp16: bool = False
    bge_m3_batch_size: int = 8
    bge_m3_batch_wait_s: float = 0.01
    bge_m3_num_replicas: int = 1
    bge_m3_max_ongoing_requests: int = 64
    bge_m3_num_cpus: float = 2.0
    bge_m3_num_gpus: float = 0.0

    def __post_init__(self) -> None:
        """Validate that numeric fields are positive."""
        _positive_ints = {
            "serve_port": self.serve_port,
            "ray_object_store_memory_mb": self.ray_object_store_memory_mb,
            "max_batch_size": self.max_batch_size,
            "max_text_length": self.max_text_length,
            "predict_batch_size": self.predict_batch_size,
            "classify_batch_size": self.classify_batch_size,
            "bge_m3_batch_size": self.bge_m3_batch_size,
            "bge_m3_num_replicas": self.bge_m3_num_replicas,
            "bge_m3_max_ongoing_requests": self.bge_m3_max_ongoing_requests,
        }
        _positive_floats = {
            "predict_batch_wait_s": self.predict_batch_wait_s,
            "classify_batch_wait_s": self.classify_batch_wait_s,
            "bge_m3_batch_wait_s": self.bge_m3_batch_wait_s,
            "comfyui_timeout_s": self.comfyui_timeout_s,
            "comfyui_poll_interval_s": self.comfyui_poll_interval_s,
            "image_moderation_timeout_s": self.image_moderation_timeout_s,
        }
        for name, val in _positive_ints.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        for name, val in _positive_floats.items():
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if not self.serve_host.strip():
            raise ValueError("serve_host must not be empty")
        if self.serve_port > 65535:
            raise ValueError(f"serve_port must be <= 65535, got {self.serve_port}")
        if not self.comfyui_base_url.startswith(("http://", "https://")):
            raise ValueError("comfyui_base_url must be an http(s) URL")
        if self.image_moderation_url and not self.image_moderation_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("image_moderation_url must be an http(s) URL")
        if self.krea_2_enabled and not self.krea_2_license_accepted:
            raise ValueError(
                "krea_2_enabled requires krea_2_license_accepted=true; review the "
                "Krea 2 Community License and confirm commercial-use eligibility"
            )
        if self.krea_2_enabled and not self.image_moderation_url:
            raise ValueError(
                "krea_2_enabled requires image_moderation_url so generated output "
                "is checked before release"
            )
        if self.ideogram_4_enabled and not self.ideogram_4_license_accepted:
            raise ValueError(
                "ideogram_4_enabled requires ideogram_4_license_accepted=true; "
                "the checkpoint is licensed for non-commercial use"
            )
        enabled_image_models = {
            name
            for name, enabled in {
                "krea_2": self.krea_2_enabled,
                "ideogram_4": self.ideogram_4_enabled,
            }.items()
            if enabled
        }
        if enabled_image_models and not self.comfyui_temp_directory:
            raise ValueError(
                "enabled image models require comfyui_temp_directory so temporary "
                "PreviewImage artifacts are removed after each request"
            )
        if enabled_image_models and self.default_image_model not in enabled_image_models:
            raise ValueError(
                f"default_image_model={self.default_image_model!r} is not enabled; "
                f"enabled image models: {sorted(enabled_image_models)}"
            )
        for name, val in {
            "bge_m3_num_cpus": self.bge_m3_num_cpus,
            "bge_m3_num_gpus": self.bge_m3_num_gpus,
        }.items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")


_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def _parse_bool(value: str) -> bool:
    """Parse a string to boolean, accepting common truthy/falsy values.

    Raises ValueError for unrecognized strings (e.g. "maybe").
    This avoids the Python footgun where ``bool("false")`` is ``True``.
    """
    lower = value.lower().strip()
    if lower in _TRUTHY:
        return True
    if lower in _FALSY:
        return False
    raise ValueError(f"Cannot interpret {value!r} as boolean")


def _parse_list(value: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


# Mapping from env var name to ServeConfig field name
_ENV_OVERRIDES: dict[str, str] = {
    "SENTIMENTIZER_SERVE_HOST": "serve_host",
    "SENTIMENTIZER_SERVE_PORT": "serve_port",
    "SENTIMENTIZER_RAY_OBJECT_STORE_MEMORY_MB": "ray_object_store_memory_mb",
    "SENTIMENTIZER_DEFAULT_MODEL": "default_model",
    "ROUTER_MODEL_PATH": "router_model_path",
    "SENTIMENTIZER_MAX_BATCH_SIZE": "max_batch_size",
    "SENTIMENTIZER_MAX_TEXT_LENGTH": "max_text_length",
    "SENTIMENTIZER_PREDICT_BATCH_SIZE": "predict_batch_size",
    "SENTIMENTIZER_PREDICT_BATCH_WAIT_S": "predict_batch_wait_s",
    "SENTIMENTIZER_CLASSIFY_BATCH_SIZE": "classify_batch_size",
    "SENTIMENTIZER_CLASSIFY_BATCH_WAIT_S": "classify_batch_wait_s",
    "SENTIMENTIZER_CORS_ORIGINS": "cors_origins",
    "SENTIMENTIZER_KREA_2_ENABLED": "krea_2_enabled",
    "SENTIMENTIZER_KREA_2_LICENSE_ACCEPTED": "krea_2_license_accepted",
    "SENTIMENTIZER_IDEOGRAM_4_ENABLED": "ideogram_4_enabled",
    "SENTIMENTIZER_IDEOGRAM_4_LICENSE_ACCEPTED": "ideogram_4_license_accepted",
    "SENTIMENTIZER_COMFYUI_BASE_URL": "comfyui_base_url",
    "SENTIMENTIZER_COMFYUI_TIMEOUT_S": "comfyui_timeout_s",
    "SENTIMENTIZER_COMFYUI_POLL_INTERVAL_S": "comfyui_poll_interval_s",
    "SENTIMENTIZER_COMFYUI_TEMP_DIRECTORY": "comfyui_temp_directory",
    "SENTIMENTIZER_IMAGE_MODERATION_URL": "image_moderation_url",
    "SENTIMENTIZER_IMAGE_MODERATION_API_KEY": "image_moderation_api_key",
    "SENTIMENTIZER_IMAGE_MODERATION_TIMEOUT_S": "image_moderation_timeout_s",
    "SENTIMENTIZER_DEFAULT_IMAGE_MODEL": "default_image_model",
    "SENTIMENTIZER_REQUEST_TIMEOUT_S": "request_timeout_s",
    "SENTIMENTIZER_API_KEYS": "api_keys",
    "SENTIMENTIZER_RATE_LIMIT_PER_MIN": "rate_limit_per_min",
    "SENTIMENTIZER_RATE_LIMIT_BURST": "rate_limit_burst",
    "SENTIMENTIZER_IDEMPOTENCY_TTL_S": "idempotency_ttl_s",
    "SENTIMENTIZER_JOB_TTL_S": "job_ttl_s",
    "SENTIMENTIZER_PROMPT_BLOCKLIST_PATH": "prompt_blocklist_path",
    "SENTIMENTIZER_EMBEDDINGS_ENABLED": "embeddings_enabled",
    "SENTIMENTIZER_EMBEDDINGS_DEVICE": "embeddings_device",
    "SENTIMENTIZER_DENSE_EMBEDDING_MODEL_ID": "dense_embedding_model_id",
    "SENTIMENTIZER_DENSE_EMBEDDING_REVISION": "dense_embedding_revision",
    "SENTIMENTIZER_BGE_M3_ENABLED": "bge_m3_enabled",
    "SENTIMENTIZER_BGE_M3_MODEL_ID": "bge_m3_model_id",
    "SENTIMENTIZER_BGE_M3_USE_FP16": "bge_m3_use_fp16",
    "SENTIMENTIZER_BGE_M3_BATCH_SIZE": "bge_m3_batch_size",
    "SENTIMENTIZER_BGE_M3_BATCH_WAIT_S": "bge_m3_batch_wait_s",
    "SENTIMENTIZER_BGE_M3_NUM_REPLICAS": "bge_m3_num_replicas",
    "SENTIMENTIZER_BGE_M3_MAX_ONGOING_REQUESTS": "bge_m3_max_ongoing_requests",
    "SENTIMENTIZER_BGE_M3_NUM_CPUS": "bge_m3_num_cpus",
    "SENTIMENTIZER_BGE_M3_NUM_GPUS": "bge_m3_num_gpus",
}

# Type coercion for non-string fields
# list[str] fields use _parse_list for comma-separated env var parsing
_FIELD_TYPES: dict[str, type | callable] = {
    "serve_port": int,
    "ray_object_store_memory_mb": int,
    "max_batch_size": int,
    "max_text_length": int,
    "predict_batch_size": int,
    "predict_batch_wait_s": float,
    "classify_batch_size": int,
    "classify_batch_wait_s": float,
    "cors_origins": list,
    "krea_2_enabled": _parse_bool,
    "krea_2_license_accepted": _parse_bool,
    "ideogram_4_enabled": _parse_bool,
    "ideogram_4_license_accepted": _parse_bool,
    "comfyui_timeout_s": float,
    "comfyui_poll_interval_s": float,
    "image_moderation_timeout_s": float,
    "request_timeout_s": int,
    "api_keys": list,
    "rate_limit_per_min": int,
    "rate_limit_burst": int,
    "idempotency_ttl_s": int,
    "job_ttl_s": int,
    "embeddings_enabled": _parse_bool,
    "bge_m3_enabled": _parse_bool,
    "bge_m3_use_fp16": _parse_bool,
    "bge_m3_batch_size": int,
    "bge_m3_batch_wait_s": float,
    "bge_m3_num_replicas": int,
    "bge_m3_max_ongoing_requests": int,
    "bge_m3_num_cpus": float,
    "bge_m3_num_gpus": float,
}


def _default_config_path() -> Path:
    """Return path to the default service.yaml bundled with the package."""
    return Path(__file__).parent / "service.yaml"


def load_serve_config(path: str | Path | None = None) -> ServeConfig:
    """Load serve configuration from YAML with environment variable overrides.

    Precedence (highest wins):
        1. Environment variables (SENTIMENTIZER_DEFAULT_MODEL, etc.)
        2. YAML config file values
        3. Dataclass defaults

    Args:
        path: Path to YAML config file. If None, checks the
              SENTIMENTIZER_SERVE_CONFIG env var, then falls back to
              the bundled service.yaml.

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
                        f"Invalid {env_var}={env_value!r}: expected {coerce.__name__}"
                    ) from None

    return ServeConfig(**values)


# Global singleton instance loaded once at import time
cfg = load_serve_config()
