"""Pydantic AI agent definitions for the tuning agent.

Two agents work together:
1. AnalysisAgent — examines training metrics and diagnoses issues
2. StrategyAgent — decides the next tuning strategy and search space

Both use GLM 5.1 via Ollama's OpenAI-compatible API.
Pydantic AI validates the LLM's structured output, rejecting
hallucinated or invalid configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from sentimentizer import new_logger
from sentimentizer.agent.loader import AgentConfig
from sentimentizer.agent.models import AnalysisResult, TuningDecision
from sentimentizer.agent.prompts import ANALYSIS_SYSTEM_PROMPT, STRATEGY_SYSTEM_PROMPT
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


@dataclass
class TuningDeps:
    """Dependencies injected into both agents.

    Provides the agents with access to tuning history, current
    metrics, and the model type being tuned.
    """

    model_type: str = "rnn"
    history: list[dict[str, Any]] | None = None
    current_metrics: dict[str, Any] | None = None
    search_space_defaults: dict[str, dict[str, Any]] | None = None


def _create_model(config: AgentConfig) -> OpenAIModel:
    """Create an OpenAI-compatible model pointed at Ollama.

    Ollama exposes an OpenAI-compatible API at /v1, so we
    use pydantic-ai-slim's OpenAIModel with OpenAIProvider
    configured with Ollama's base_url.
    """
    provider = OpenAIProvider(
        base_url=config.ollama_base_url,
        api_key="ollama",  # Ollama doesn't require a real key
    )
    return OpenAIModel(
        model_name=config.model_name,
        provider=provider,
    )


def create_analysis_agent(config: AgentConfig) -> Agent[TuningDeps, AnalysisResult]:
    """Create the analysis agent that examines training metrics.

    This agent:
    - Reads validation loss, accuracy, and training loss curves
    - Detects overfitting (val_loss rising while train_loss falls)
    - Detects underfitting (both losses remain high)
    - Assesses whether learning rate is appropriate
    - Suggests which parameters to focus on next

    Returns a Pydantic-validated AnalysisResult.
    """
    model = _create_model(config)

    agent = Agent(
        model=model,
        output_type=AnalysisResult,
        deps_type=TuningDeps,
        system_prompt=ANALYSIS_SYSTEM_PROMPT,
        model_settings={"temperature": config.temperature, "max_tokens": config.max_tokens},
    )

    @agent.tool
    def get_previous_results(ctx: RunContext[TuningDeps]) -> list[dict[str, Any]]:
        """Get the history of past tuning iterations."""
        return ctx.deps.history or []

    @agent.tool
    def get_current_metrics(ctx: RunContext[TuningDeps]) -> dict[str, Any]:
        """Get the current iteration's training metrics."""
        return ctx.deps.current_metrics or {}

    @agent.tool
    def get_model_type(ctx: RunContext[TuningDeps]) -> str:
        """Get the model type being tuned (rnn, encoder, decoder)."""
        return ctx.deps.model_type

    return agent


def create_strategy_agent(config: AgentConfig) -> Agent[TuningDeps, TuningDecision]:
    """Create the strategy agent that decides the next tuning action.

    This agent:
    - Receives the analysis of training metrics
    - Decides whether to widen, narrow, change focus, or stop
    - Produces a new search space configuration
    - Sets the number of trials for the next Ray Tune run

    Returns a Pydantic-validated TuningDecision, ensuring
    the search space and strategy are always valid.
    """
    model = _create_model(config)

    agent = Agent(
        model=model,
        output_type=TuningDecision,
        deps_type=TuningDeps,
        system_prompt=STRATEGY_SYSTEM_PROMPT,
        model_settings={"temperature": config.temperature, "max_tokens": config.max_tokens},
    )

    @agent.tool
    def get_analysis(ctx: RunContext[TuningDeps]) -> dict[str, Any]:
        """Get the current analysis results from the analysis agent."""
        return ctx.deps.current_metrics or {}

    @agent.tool
    def get_previous_results(ctx: RunContext[TuningDeps]) -> list[dict[str, Any]]:
        """Get the history of past tuning iterations."""
        return ctx.deps.history or []

    @agent.tool
    def get_default_search_space(ctx: RunContext[TuningDeps]) -> dict[str, dict[str, Any]]:
        """Get the default search space parameters from the YAML config."""
        return ctx.deps.search_space_defaults or {}

    @agent.tool
    def get_model_type(ctx: RunContext[TuningDeps]) -> str:
        """Get the model type being tuned (rnn, encoder, decoder)."""
        return ctx.deps.model_type

    return agent
