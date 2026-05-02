"""LangGraph agent state definition.

The AgentState TypedDict flows through the LangGraph state graph,
carrying tuning history, analysis results, and the current decision
between nodes.
"""

from __future__ import annotations

from typing import Any, TypedDict

from sentimentizer.agent.models import AgentRunResult, AnalysisResult, TuningDecision, TuningResult


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph tuning agent.

    Each node reads from and writes to this state. LangGraph
    checkpointing persists this state between runs.

    Attributes:
        iteration: Current iteration number (0-based).
        model_type: Model being tuned (rnn, encoder, decoder).
        history: List of TuningResult dicts from completed iterations.
        current_analysis: Latest AnalysisResult from the analysis agent.
        current_decision: Latest TuningDecision from the strategy agent.
        current_result: Latest TuningResult from Ray Tune.
        best_config: Best hyperparameter configuration found so far.
        best_accuracy: Best validation accuracy achieved so far.
        best_loss: Best validation loss achieved so far.
        search_space_overrides: Current search space overrides from the agent.
        agent_config_path: Path to the agent config YAML file.
        converged: Whether the tuning loop has converged.
        final_result: Complete result when the loop ends.
    """

    iteration: int
    model_type: str
    history: list[dict[str, Any]]
    current_analysis: AnalysisResult
    current_decision: TuningDecision
    current_result: TuningResult
    best_config: dict[str, float | int]
    best_accuracy: float
    best_loss: float
    search_space_overrides: dict[str, dict[str, Any]]
    agent_config_path: str | None
    converged: bool
    final_result: AgentRunResult
