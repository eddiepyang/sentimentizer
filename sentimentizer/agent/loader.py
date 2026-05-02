"""Load agent and tuner configuration from YAML.

Provides dataclass-backed configuration loading with sensible defaults,
so the YAML file only needs to override what differs from defaults.

TunerConfig and load_search_space live in sentimentizer.tuner (the
standalone tuning module). This module re-exports them for convenience
and adds AgentConfig, which is agent-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sentimentizer.tuner import TunerConfig


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

    model_name: str = "glm-5.1:cloud"
    ollama_base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.3
    max_tokens: int = 2048
    max_iterations: int = 5
    convergence_threshold: float = 0.005
    initial_search_strategy: str = "bayesian"
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    human_in_the_loop: bool = False


def _get_default_config_path() -> Path:
    """Return path to the default config.yaml bundled with the package."""
    return Path(__file__).parent / "config.yaml"


def _config_dict_to_agent_config(agent_dict: dict[str, Any]) -> AgentConfig:
    """Convert the agent section of the YAML dict into an AgentConfig."""
    cp_dict = agent_dict.pop("checkpointing", {})
    checkpoint_cfg = CheckpointConfig(**cp_dict)
    return AgentConfig(checkpointing=checkpoint_cfg, **agent_dict)


def load_agent_config(
    path: str | Path | None = None,
) -> tuple[AgentConfig, TunerConfig]:
    """Load agent and tuner configuration from a YAML file.

    Args:
        path: Path to YAML config file. If None, uses the default
              config.yaml bundled with the package.

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
    tuner_cfg = TunerConfig(**raw.get("tuner", {}))

    return agent_cfg, tuner_cfg


# Allow overriding config path via environment variable
CONFIG_PATH_ENV = "SENTIMENTIZER_AGENT_CONFIG"
