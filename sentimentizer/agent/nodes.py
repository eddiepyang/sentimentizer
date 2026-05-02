"""LangGraph node functions for the tuning agent.

Each node is an async function that:
1. Reads from AgentState
2. Calls a Pydantic AI agent or Ray Tune
3. Returns state updates (partial dict)

Nodes:
- analyze: Uses the analysis agent to examine metrics
- decide: Uses the strategy agent to choose next action
- tune: Executes Ray Tune with the chosen search space
- evaluate: Checks convergence and updates best results
"""

from __future__ import annotations

from typing import Any

from sentimentizer import new_logger
from sentimentizer.agent.agents import TuningDeps, create_analysis_agent, create_strategy_agent
from sentimentizer.agent.loader import load_agent_config
from sentimentizer.agent.models import AgentRunResult, AnalysisResult, TuningDecision, TuningResult
from sentimentizer.agent.state import AgentState
from sentimentizer.config import DEFAULT_LOG_LEVEL
from sentimentizer.tuner import load_search_space

logger = new_logger(DEFAULT_LOG_LEVEL)


async def analyze(state: AgentState) -> dict[str, Any]:
    """Use the Pydantic AI analysis agent to examine training results.

    Calls GLM 5.1 to analyze metrics history and diagnose issues
    like overfitting, underfitting, or suboptimal learning rate.
    """
    config_path = state.get("agent_config_path")
    agent_config, _ = load_agent_config(config_path)

    analysis_agent = create_analysis_agent(agent_config)
    deps = TuningDeps(
        model_type=state.get("model_type", "rnn"),
        history=state.get("history", []),
        current_metrics={
            "best_accuracy": state.get("best_accuracy", 0.0),
            "best_loss": state.get("best_loss", float("inf")),
            "best_config": state.get("best_config", {}),
        },
    )

    iteration = state.get("iteration", 0)
    prompt = f"Analyze the tuning results after iteration {iteration}."
    if state.get("history"):
        prompt += " Use get_previous_results and get_current_metrics to examine the data."

    result = await analysis_agent.run(prompt, deps=deps)
    analysis: AnalysisResult = result.output

    logger.info(
        "analysis_complete",
        summary=analysis.summary,
        overfitting=analysis.overfitting,
        underfitting=analysis.underfitting,
        lr_status=analysis.lr_status,
        suggested_focus=analysis.suggested_focus,
    )

    return {"current_analysis": analysis}


async def decide(state: AgentState) -> dict[str, Any]:
    """Use the Pydantic AI strategy agent to decide the next action.

    Calls GLM 5.1 to choose a tuning strategy (widen, narrow,
    change_focus, increase_epochs, stop) and produce a validated
    TuningDecision with an updated search space.
    """
    config_path = state.get("agent_config_path")
    agent_config, tuner_config = load_agent_config(config_path)

    strategy_agent = create_strategy_agent(agent_config)

    # Get default search space for the model type
    model_type = state.get("model_type", "rnn")
    try:
        default_space = load_search_space(model_type, tuner_config=tuner_config)
    except ValueError:
        default_space = {}

    deps = TuningDeps(
        model_type=model_type,
        history=state.get("history", []),
        current_metrics={
            "analysis": state.get(
                "current_analysis",
                AnalysisResult(summary="No analysis yet"),
            ).model_dump(),
            "best_accuracy": state.get("best_accuracy", 0.0),
            "best_config": state.get("best_config", {}),
        },
        search_space_defaults=default_space,
    )

    prompt = "Based on the analysis, decide the next tuning strategy."
    if state.get("current_analysis"):
        prompt += f" The analysis found: {state['current_analysis'].summary}"

    result = await strategy_agent.run(prompt, deps=deps)
    decision: TuningDecision = result.output

    logger.info(
        "decision_made",
        strategy=decision.strategy,
        reasoning=decision.reasoning,
        num_samples=decision.num_samples,
        search_space_params=list(decision.search_space.keys()),
    )

    # Convert Pydantic SearchSpaceParam objects to plain dicts for Ray Tune
    overrides = {}
    for param_name, param_spec in decision.search_space.items():
        overrides[param_name] = param_spec.model_dump(exclude_none=True)

    return {
        "current_decision": decision,
        "search_space_overrides": overrides,
    }


def tune(state: AgentState) -> dict[str, Any]:
    """Execute Ray Tune with the search space chosen by the agent.

    This is a synchronous node (no LLM needed) that runs the
    actual hyperparameter search using Ray Tune + Optuna.
    """
    config_path = state.get("agent_config_path")
    agent_config, tuner_config = load_agent_config(config_path)

    model_type = state.get("model_type", "rnn")
    overrides = state.get("search_space_overrides")

    # Override num_samples if the agent decided differently
    decision = state.get("current_decision")
    if decision:
        tuner_config.num_samples = decision.num_samples

    logger.info(
        "running_tuning",
        model_type=model_type,
        num_samples=tuner_config.num_samples,
        has_overrides=overrides is not None,
    )

    # Lazy import to break circular dependency
    from sentimentizer.tuner import tune_model

    result = tune_model(
        model_type=model_type,
        tuner_config=tuner_config,
        search_space_overrides=overrides,
        config_path=config_path,
    )

    # Convert to TuningResult for the agent state
    previous_best = state.get("best_accuracy", 0.0)
    tuning_result = TuningResult(
        best_accuracy=result["best_accuracy"],
        best_loss=result["best_loss"],
        best_config=result["best_config"],
        trial_count=result["trial_count"],
        improvement_over_last=result["best_accuracy"] - previous_best,
    )

    logger.info(
        "tuning_iteration_complete",
        best_accuracy=tuning_result.best_accuracy,
        best_loss=tuning_result.best_loss,
        trial_count=tuning_result.trial_count,
        improvement=tuning_result.improvement_over_last,
    )

    return {"current_result": tuning_result}


def evaluate(state: AgentState) -> dict[str, Any]:
    """Evaluate results and check convergence.

    Updates the best config if the current result is better,
    and checks if the tuning loop should stop.
    """
    result: TuningResult | None = state.get("current_result")
    if result is None:
        return {"converged": False}

    # Update best if current is better
    current_best_accuracy = state.get("best_accuracy", 0.0)
    current_best_config = state.get("best_config", {})
    current_best_loss = state.get("best_loss", float("inf"))

    if result.best_accuracy > current_best_accuracy:
        new_best_accuracy = result.best_accuracy
        new_best_config = result.best_config
        new_best_loss = result.best_loss
    else:
        new_best_accuracy = current_best_accuracy
        new_best_config = current_best_config
        new_best_loss = current_best_loss

    # Append to history
    history = list(state.get("history", []))
    history.append(result.model_dump())

    # Check convergence
    iteration = state.get("iteration", 0) + 1
    config_path = state.get("agent_config_path")
    agent_config, _ = load_agent_config(config_path)

    converged = _check_convergence(
        history=history,
        threshold=agent_config.convergence_threshold,
        max_iterations=agent_config.max_iterations,
        iteration=iteration,
        decision=state.get("current_decision"),
    )

    # Build final result if converged
    final_result = None
    if converged:
        final_result = AgentRunResult(
            best_config=new_best_config,
            best_accuracy=new_best_accuracy,
            best_loss=new_best_loss,
            iterations_completed=iteration,
            converged=converged,
            history=[TuningResult(**h) for h in history],
        )

    logger.info(
        "evaluation_complete",
        iteration=iteration,
        best_accuracy=new_best_accuracy,
        best_loss=new_best_loss,
        converged=converged,
    )

    updates: dict[str, Any] = {
        "iteration": iteration,
        "history": history,
        "best_accuracy": new_best_accuracy,
        "best_config": new_best_config,
        "best_loss": new_best_loss,
        "converged": converged,
    }

    if final_result is not None:
        updates["final_result"] = final_result

    return updates


def _check_convergence(
    history: list[dict[str, Any]],
    threshold: float,
    max_iterations: int,
    iteration: int,
    decision: TuningDecision | None = None,
) -> bool:
    """Check if the tuning loop should stop.

    Convergence is reached if:
    1. The agent explicitly decided to stop
    2. Max iterations reached
    3. Improvement over last 3 iterations is below threshold
    """
    # Agent explicitly decided to stop
    if decision and decision.strategy == "stop":
        logger.info("converged_agent_stop", reason="Agent decided to stop")
        return True

    # Max iterations reached
    if iteration >= max_iterations:
        logger.info("converged_max_iterations", iteration=iteration, max=max_iterations)
        return True

    # Check improvement over last 3 iterations
    if len(history) >= 3:
        recent = history[-3:]
        improvements = [h.get("improvement_over_last", 0.0) for h in recent]
        avg_improvement = sum(improvements) / len(improvements)
        if abs(avg_improvement) < threshold:
            logger.info(
                "converged_threshold",
                avg_improvement=avg_improvement,
                threshold=threshold,
            )
            return True

    return False
