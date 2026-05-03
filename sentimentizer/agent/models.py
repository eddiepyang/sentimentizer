"""Pydantic models for the tuning agent's structured input/output.

These models validate the LLM's responses and the tuning results,
ensuring that only valid configurations are passed to Ray Tune.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchSpaceParam(BaseModel):
    """A single parameter's search space specification.

    Produced by the LLM agent when it decides to narrow/widen
    the search space for a hyperparameter.
    """

    type: Literal["loguniform", "uniform", "choice", "randint"]
    low: float | None = None
    high: float | None = None
    values: list[int] | list[float] | list[str] | None = None


class TuningDecision(BaseModel):
    """Structured output from the strategy agent.

    The LLM produces this after analyzing training metrics.
    Pydantic validates it, rejecting hallucinated or invalid configs.
    """

    reasoning: str = Field(
        ...,
        description="Brief explanation of why this strategy was chosen",
    )
    strategy: Literal["widen", "narrow", "change_focus", "increase_epochs", "stop"] = Field(
        ...,
        description="The tuning strategy to apply next",
    )
    search_space: dict[str, SearchSpaceParam] = Field(
        ...,
        description="Updated search space parameters for Ray Tune",
    )
    num_samples: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of trials for Ray Tune to run",
    )


class TuningResult(BaseModel):
    """Result from a single Ray Tune tuning run.

    Fed back to the LLM agent for analysis in the next iteration.
    """

    best_accuracy: float = Field(..., description="Best validation accuracy achieved")
    best_loss: float = Field(..., description="Best validation loss achieved")
    best_config: dict[str, float | int] = Field(
        ..., description="Best hyperparameter configuration found"
    )
    trial_count: int = Field(..., description="Number of trials completed")
    improvement_over_last: float = Field(
        default=0.0, description="Accuracy improvement vs. previous best"
    )


class AnalysisResult(BaseModel):
    """Structured output from the analysis agent.

    The LLM produces this after examining training metrics and history.
    """

    summary: str = Field(..., description="Brief summary of the training results")
    overfitting: bool = Field(
        default=False, description="Whether the model appears to be overfitting"
    )
    underfitting: bool = Field(
        default=False, description="Whether the model appears to be underfitting"
    )
    lr_status: Literal["too_high", "too_low", "appropriate", "unclear"] = Field(
        default="unclear", description="Assessment of the learning rate"
    )
    suggested_focus: list[str] = Field(
        default_factory=list,
        description="Parameters to focus on in the next iteration",
    )


class AgentRunResult(BaseModel):
    """Final result from the complete agent tuning loop.

    Returned when the agent converges or reaches max iterations.
    """

    best_config: dict[str, float | int]
    best_accuracy: float
    best_loss: float
    iterations_completed: int
    converged: bool
    history: list[TuningResult] = Field(default_factory=list)
