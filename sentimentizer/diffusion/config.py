"""Configuration dataclasses for diffusion models.

Defaults live in ``diffusion_config.yaml`` next to this module and are
loaded by ``load_diffusion_config()``. Module-level constants
(``SD35_DEFAULT_CONFIG``, ``SDXL_DEFAULT_CONFIG``, ``FLUX2_KLEIN_DEFAULT_CONFIG``)
are eager-loaded aliases for the supported models.

Operational overrides (which models to enable, API keys, rate limits,
model_id / cpu_offload) live in ``sentimentizer.serve.config`` and are
applied on top of these defaults in the dispatcher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import yaml

from sentimentizer.diffusion.mlx_compat import MFLUX_AVAILABLE, is_mlx_device


@dataclass(frozen=True)
class DiffusionModelConfig:
    """Configuration for a single diffusion model.

    Attributes:
        model_id: HuggingFace model ID or local path for from_pretrained.
        model_path: Path to weights file (e.g. GGUF for FLUX).
        dtype: Torch dtype for inference (e.g. torch.bfloat16).
        quantization: Quantization config string (e.g. "Q8_0") or None.
        default_steps: Default number of denoising steps.
        default_guidance: Default classifier-free guidance scale.
        max_pixels: Maximum width*height allowed (e.g. 1048576 for 1024²).
        dim_alignment: Dimension alignment requirement (8 for SD, 16 for FLUX).
        cpu_offload: Diffusers CPU offload mode. One of None (full GPU),
            "model" (whole-module swap, modest VRAM win), or "sequential"
            (submodule swap, biggest VRAM win, slowest).
        backend: Backend to use ("diffusers", "mlx", or "auto").
    """

    model_id: str = ""
    model_path: str = ""
    dtype: str = "bfloat16"
    quantization: str | None = None
    default_steps: int = 30
    default_guidance: float = 7.5
    max_pixels: int = 1048576
    dim_alignment: int = 8
    cpu_offload: str | None = None
    backend: Literal["diffusers", "mlx", "auto"] = "auto"


BACKEND_REGISTRY: dict[str, list[str]] = {
    "sdxl": ["diffusers"],
    "sd35": ["diffusers"],
    "flux2_klein": ["diffusers"] + (["mlx"] if MFLUX_AVAILABLE else []),
}


def resolve_backend(model_key: str, backend: str = "auto") -> str:
    """Resolve 'auto' to a concrete backend based on model and device.

    For SDXL and SD3.5, always returns 'diffusers' (no MLX implementation).
    For FLUX.2 Klein, returns 'mlx' on Apple Silicon when mflux is installed,
    otherwise 'diffusers'.
    """
    available = BACKEND_REGISTRY.get(model_key, ["diffusers"])
    if backend != "auto":
        if backend not in available:
            raise ValueError(
                f"backend={backend!r} not available for {model_key}. " f"Available: {available}"
            )
        return backend
    if "mlx" in available and is_mlx_device():
        return "mlx"
    return "diffusers"


# Per-model env-var overrides for model-internal defaults. Pattern:
# SENTIMENTIZER_DIFFUSION_<MODEL>_<FIELD>. Only fields that don't already
# have a higher-priority override in ServeConfig are listed here; identity
# fields (model_id / model_path / cpu_offload) are owned by ServeConfig.
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "SENTIMENTIZER_DIFFUSION_SD35_DEFAULT_STEPS": ("sd35", "default_steps"),
    "SENTIMENTIZER_DIFFUSION_SD35_DEFAULT_GUIDANCE": ("sd35", "default_guidance"),
    "SENTIMENTIZER_DIFFUSION_SD35_MAX_PIXELS": ("sd35", "max_pixels"),
    "SENTIMENTIZER_DIFFUSION_SDXL_DEFAULT_STEPS": ("sdxl", "default_steps"),
    "SENTIMENTIZER_DIFFUSION_SDXL_DEFAULT_GUIDANCE": ("sdxl", "default_guidance"),
    "SENTIMENTIZER_DIFFUSION_SDXL_MAX_PIXELS": ("sdxl", "max_pixels"),
    "SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_DEFAULT_STEPS": ("flux2_klein", "default_steps"),
    "SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_DEFAULT_GUIDANCE": ("flux2_klein", "default_guidance"),
    "SENTIMENTIZER_DIFFUSION_FLUX2_KLEIN_MAX_PIXELS": ("flux2_klein", "max_pixels"),
}

# With `from __future__ import annotations` above, dataclass field
# `.type` is the annotation source string (e.g. "int", "float",
# "str | None"), not the resolved type object — so we compare against
# strings in _coerce below.
_FIELD_TYPES: dict[str, str] = {f.name: f.type for f in fields(DiffusionModelConfig)}


def _default_config_path() -> Path:
    """Return path to the bundled diffusion_config.yaml."""
    return Path(__file__).parent / "diffusion_config.yaml"


def _coerce(field_name: str, value: str) -> Any:
    """Coerce a string env value to the dataclass field's type."""
    annotation = _FIELD_TYPES.get(field_name, "str")
    if annotation == "int":
        return int(value)
    if annotation == "float":
        return float(value)
    return value


def load_diffusion_config(
    path: str | Path | None = None,
) -> dict[str, DiffusionModelConfig]:
    """Load diffusion model configs from YAML with env-var overrides.

    Precedence (highest wins):
        1. Environment variables (SENTIMENTIZER_DIFFUSION_<MODEL>_<FIELD>)
        2. YAML config file values
        3. Dataclass defaults

    Args:
        path: Path to YAML config. If None, uses the bundled
              ``diffusion_config.yaml``.

    Returns:
        Dict of ``{"sd": DiffusionModelConfig, "flux": ..., "sd35": ...}``.
    """
    path = _default_config_path() if path is None else Path(path)

    raw: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    per_model: dict[str, dict[str, Any]] = {
        name: dict(raw.get(name, {})) for name in ("sd35", "sdxl", "flux2_klein")
    }

    for env_var, (model_name, field_name) in _ENV_OVERRIDES.items():
        env_value = os.environ.get(env_var)
        if env_value is None:
            continue
        try:
            per_model[model_name][field_name] = _coerce(field_name, env_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {env_var}={env_value!r}: expected {_FIELD_TYPES[field_name]}"
            ) from exc

    return {name: DiffusionModelConfig(**values) for name, values in per_model.items()}


_DEFAULTS = load_diffusion_config()
SD35_DEFAULT_CONFIG = _DEFAULTS["sd35"]
SDXL_DEFAULT_CONFIG = _DEFAULTS["sdxl"]
FLUX2_KLEIN_DEFAULT_CONFIG = _DEFAULTS["flux2_klein"]
