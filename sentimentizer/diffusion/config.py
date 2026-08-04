"""Krea 2 and Ideogram 4 headless workflow configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ImageModelConfig:
    """Checkpoint names and generation limits for a ComfyUI image model."""

    transformer: str
    text_encoder: str
    vae: str
    unconditional_transformer: str = ""
    default_steps: int = 20
    default_guidance: float = 1.0
    max_pixels: int = 1024 * 1024
    dim_alignment: int = 16
    quantization: str = "int8_convrot"


_MODEL_NAMES = ("krea_2", "ideogram_4")
_FIELD_NAMES = {field.name for field in fields(ImageModelConfig)}


def _default_config_path() -> Path:
    return Path(__file__).parent / "diffusion_config.yaml"


def _coerce(field_name: str, value: str) -> Any:
    if field_name in {"default_steps", "max_pixels", "dim_alignment"}:
        return int(value)
    if field_name == "default_guidance":
        return float(value)
    return value


def load_diffusion_config(path: str | Path | None = None) -> dict[str, ImageModelConfig]:
    """Load checkpoint/default settings with per-field env overrides."""
    config_path = _default_config_path() if path is None else Path(path)
    with config_path.open() as config_file:
        raw = yaml.safe_load(config_file) or {}
    configs: dict[str, ImageModelConfig] = {}
    for model_name in _MODEL_NAMES:
        values = dict(raw.get(model_name, {}))
        for field_name in _FIELD_NAMES:
            env_name = f"SENTIMENTIZER_DIFFUSION_{model_name.upper()}_{field_name.upper()}"
            env_value = os.environ.get(env_name)
            if env_value is not None:
                values[field_name] = _coerce(field_name, env_value)
        configs[model_name] = ImageModelConfig(**values)
    return configs


IMAGE_MODEL_CONFIGS = load_diffusion_config()
KREA_2_CONFIG = IMAGE_MODEL_CONFIGS["krea_2"]
IDEOGRAM_4_CONFIG = IMAGE_MODEL_CONFIGS["ideogram_4"]
