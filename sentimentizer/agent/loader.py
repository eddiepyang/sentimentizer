"""Load agent and tuner configuration from YAML.

Provides dataclass-backed configuration loading with sensible defaults,
so the YAML file only needs to override what differs from defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CheckpointConfig:
    """LangGraph checkpointing configuration."""

    enabled: bool = True
    db_path: str = "agent_checkpoints.db"


@dataclass
class AgentConfig:
    """Configuration for the Pydantic AI tuning agent.

    Connects to GLM 5.1 (or any model) via Ollama's OpenAI-compatible API.
    """

    model_name: str = "glm5.1"
    ollama_base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.3
    max_tokens: int = 2048
    max_iterations: int = 5
    convergence_threshold: float = 0.005
    initial_search_strategy: str = "bayesian"
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    human_in_the_loop: bool = False


@dataclass
class TunerConfig:
    """Ray Tune + Optuna hyperparameter search configuration."""

    scheduler: str = "asha"
    metric: str = "val_accuracy"
    mode: str = "max"
    num_samples: int = 20
    grace_period: int = 2
    reduction_factor: int = 3
    search_spaces: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    Lists and scalar values in override replace those in base.
    Nested dicts are merged recursively.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_default_config_path() -> Path:
    """Return path to the default config.yaml bundled with the package."""
    return Path(__file__).parent / "config.yaml"


def _config_dict_to_agent_config(agent_dict: dict[str, Any]) -> AgentConfig:
    """Convert the agent section of the YAML dict into an AgentConfig dataclass."""
    cp_dict = agent_dict.pop("checkpointing", {})
    checkpoint_cfg = CheckpointConfig(**cp_dict)
    return AgentConfig(checkpointing=checkpoint_cfg, **agent_dict)


def _config_dict_to_tuner_config(tuner_dict: dict[str, Any]) -> TunerConfig:
    """Convert the tuner section of the YAML dict into a TunerConfig dataclass."""
    return TunerConfig(**tuner_dict)


def load_agent_config(
    path: str | Path | None = None,
) -> tuple[AgentConfig, TunerConfig]:
    """Load agent and tuner configuration from a YAML file.

    Args:
        path: Path to YAML config file. If None, uses the default config.yaml
              bundled with the package.

    Returns:
        Tuple of (AgentConfig, TunerConfig) with all settings populated.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = _get_default_config_path() if path is None else Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Agent config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    agent_cfg = _config_dict_to_agent_config(raw.get("agent", {}))
    tuner_cfg = _config_dict_to_tuner_config(raw.get("tuner", {}))

    return agent_cfg, tuner_cfg


def load_search_space(
    model_type: str,
    tuner_config: TunerConfig | None = None,
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the search space for a specific model type.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.
        tuner_config: Optional pre-loaded TunerConfig. If None, loads from config_path.
        config_path: Path to YAML config file (only used if tuner_config is None).

    Returns:
        Dict mapping parameter names to their search space specs,
        e.g. {'lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2}, ...}

    Raises:
        ValueError: If model_type is not found in the search spaces.
    """
    if tuner_config is None:
        _, tuner_config = load_agent_config(config_path)

    if model_type not in tuner_config.search_spaces:
        available = list(tuner_config.search_spaces.keys())
        raise ValueError(
            f"No search space defined for model type '{model_type}'. Available: {available}"
        )

    return tuner_config.search_spaces[model_type]


# Allow overriding config path via environment variable
CONFIG_PATH_ENV = "SENTIMENTIZER_AGENT_CONFIG"


def get_config_path() -> Path | None:
    """Get config path from environment variable, or return None for default."""
    env_path = os.environ.get(CONFIG_PATH_ENV)
    if env_path:
        return Path(env_path)
    return None
