"""Click CLI: commands and shared options.

IMPORTANT: This module imports workflows.lifecycle at the top level
(which triggers env setup and cleanup registration). It does NOT import
any stage modules at module level — stages are imported inside command
handlers to keep ``--help`` fast and avoid pulling in torch/ray.
"""

from __future__ import annotations

from typing import Any

import click

from workflows.lifecycle import State

# ──────────────────────────────────────────────
# Shared decorator for train/run flags
# ──────────────────────────────────────────────


def shared_train_options(func: click.Command) -> click.Command:
    """Shared flags for ``train`` and ``run`` — keeps their surfaces in lockstep.

    ``--resume`` is intentionally NOT in this list because ``run`` exposes
    it as split flags (--resume-tokenize / --resume-train) while ``train``
    keeps the unified ``--resume``.
    """
    options = [
        click.option("--distributed", is_flag=True, help="Use Ray Train for distributed training"),
        click.option("--num-workers", default=1, type=int, help="Number of Ray Train workers"),
        click.option("--save/--no-save", default=False, help="Save model weights after training"),
        click.option("--checkpoint-dir", default="", type=str, help="Directory for checkpoints"),
        click.option(
            "--checkpoint-every", default=1, type=int, help="Save checkpoint every N epochs"
        ),
        click.option(
            "--balance-classes",
            is_flag=True,
            help="Enable class balancing (undersampling majority)",
        ),
        click.option(
            "--balance-seed", default=42, type=int, help="Random seed for class balancing"
        ),
        click.option(
            "--pos-weight",
            default=0.0,
            type=float,
            help="Loss weight for positive class (0 = auto-calculate)",
        ),
        click.option("--push-to-hub", is_flag=True, help="Push model weights to Hugging Face Hub"),
        click.option(
            "--pull-from-hub", is_flag=True, help="Pull model weights from Hugging Face Hub"
        ),
        click.option("--hf-repo", default=None, type=str, help="Hugging Face repository ID"),
    ]
    for option in reversed(options):
        func = option(func)  # type: ignore[misc]
    return func


# ──────────────────────────────────────────────
# CLI group definition
# ──────────────────────────────────────────────


@click.group()
@click.option(
    "--model",
    default="rnn",
    type=click.Choice(["rnn", "encoder", "decoder"]),
    envvar="SENTIMENTIZER_MODEL",
    help="Model architecture: rnn, encoder, or decoder",
)
@click.option(
    "--device",
    default="auto",
    envvar="SENTIMENTIZER_DEVICE",
    help="Compute device: auto, cuda, mps, or cpu (resolved lazily)",
)
@click.option(
    "--run-type",
    default="new",
    type=click.Choice(["new", "update"]),
    help="new = build/train fresh; update = reuse existing artifacts",
)
@click.pass_context
def cli(ctx: click.Context, model: str, device: str, run_type: str) -> None:
    """Sentimentizer ML pipeline."""
    # Store raw values only. Do NOT call auto_detect_device here or import
    # sentimentizer.config — it transitively imports torch, which would
    # trigger torch loading for every <subcommand> --help.
    ctx.obj = State(model=model, device=device, run_type=run_type)


# ── extract ──────────────────────────────────


@cli.command()
@click.option("--stop", default=10000, type=int, help="Number of review lines to load")
@click.pass_context
def extract(ctx: click.Context, stop: int) -> None:
    """Extract raw reviews into parquet."""
    from workflows.stages.extract import run_extract

    run_extract(ctx.obj, stop=stop)


# ── tokenize ─────────────────────────────────


@cli.command()
@click.option(
    "--resume",
    is_flag=True,
    help="Update the existing dictionary from new data",
)
@click.pass_context
def tokenize(ctx: click.Context, resume: bool) -> None:
    """Build/update dictionary and write processed parquet."""
    from workflows.stages.tokenize import run_tokenize

    run_tokenize(ctx.obj, resume=resume)


# ── train ────────────────────────────────────


@cli.command()
@shared_train_options
@click.option(
    "--resume",
    is_flag=True,
    help="Resume training from latest checkpoint in --checkpoint-dir",
)
@click.pass_context
def train(ctx: click.Context, resume: bool, **kwargs: Any) -> None:
    """Fit the model (single-node or distributed)."""
    from workflows.stages.train import run_train

    run_train(ctx.obj, resume=resume, **kwargs)


# ── tune ─────────────────────────────────────


@cli.command()
@click.option(
    "--mode",
    default="agent",
    type=click.Choice(["agent", "standalone"]),
    help="Tuning mode: agent (LLM-guided loop) or standalone (Ray Tune)",
)
@click.option("--agent-config", default=None, type=str, help="Path to agent config YAML")
@click.option(
    "--samples", "tune_samples", default=20, type=int, help="Ray Tune trials per iteration"
)
@click.option(
    "--max-iterations",
    "tune_max_iterations",
    default=5,
    type=int,
    help="Maximum agent tuning iterations",
)
@click.option(
    "--output-dir",
    "tune_output_dir",
    default="tuning_results",
    type=str,
    help="Directory for tuning results",
)
@click.option("--no-validate", is_flag=True, help="Skip model prediction validation")
@click.option(
    "--validation-threshold",
    default=0.75,
    type=float,
    help="Minimum fraction of correct predictions to pass validation",
)
@click.option("--max-retries", default=2, type=int, help="Max re-tuning attempts on failure")
@click.option("--save/--no-save", default=False, help="Save best model after tuning")
@click.option("--push-to-hub", is_flag=True, help="Push model weights to Hugging Face Hub")
@click.option("--balance-classes", is_flag=True, help="Enable class balancing in training data")
@click.option("--balance-seed", default=42, type=int, help="Random seed for class balancing")
@click.option("--pos-weight", default=0.0, type=float, help="Loss weight for positive class")
@click.pass_context
def tune(ctx: click.Context, mode: str, **kwargs: Any) -> None:
    """Hyperparameter tuning (LLM-guided agent or standalone Ray Tune run)."""
    from workflows.stages.tune import run_tune

    run_tune(ctx.obj, mode=mode, **kwargs)


# ── hf ───────────────────────────────────────


@cli.group()
def hf() -> None:
    """Hugging Face Hub operations."""


@hf.command("push")
@click.option("--repo-id", default=None, type=str, help="Hugging Face repository ID")
@click.pass_context
def hf_push(ctx: click.Context, repo_id: str | None) -> None:
    """Push model weights to Hugging Face Hub."""
    from workflows.stages.hf import run_hf_push

    run_hf_push(ctx.obj, repo_id=repo_id)


@hf.command("pull")
@click.option("--repo-id", default=None, type=str, help="Hugging Face repository ID")
@click.pass_context
def hf_pull(ctx: click.Context, repo_id: str | None) -> None:
    """Pull model weights from Hugging Face Hub."""
    from workflows.stages.hf import run_hf_pull

    run_hf_pull(ctx.obj, repo_id=repo_id)


# ── diagnose ─────────────────────────────────


@cli.group()
def diagnose() -> None:
    """Pipeline diagnostics."""


@diagnose.command("env")
@click.pass_context
def diagnose_env(ctx: click.Context) -> None:
    """Fast environment check (Python / CUDA paths / Ray env vars). No torch/ray."""
    from workflows.stages.diagnose import run_diagnose_env

    run_diagnose_env(ctx.obj)


@diagnose.command("pipeline")
@click.pass_context
def diagnose_pipeline(ctx: click.Context) -> None:
    """Heavy pipeline check. Imports the ML stack."""
    from workflows.stages.diagnose import run_diagnose_pipeline

    run_diagnose_pipeline(ctx.obj)


# ── run (full pipeline) ──────────────────────


@cli.command()
@click.option("--stop", default=10000, type=int, help="Number of review lines to load")
@click.option(
    "--resume-tokenize",
    is_flag=True,
    help="Update existing dictionary during tokenize stage",
)
@click.option(
    "--resume-train",
    is_flag=True,
    help="Resume training from checkpoint",
)
@shared_train_options
@click.pass_context
def run(
    ctx: click.Context,
    stop: int,
    resume_tokenize: bool,
    resume_train: bool,
    **train_kwargs: Any,
) -> None:
    """Run the full pipeline: extract → tokenize → train.

    Mirrors today's default ``python driver.py --type new`` invocation.
    The single legacy ``--resume`` flag is split into stage-specific flags
    so each stage's resume can be controlled independently.
    """
    from workflows.stages.extract import run_extract
    from workflows.stages.tokenize import run_tokenize
    from workflows.stages.train import run_train

    run_extract(ctx.obj, stop=stop)
    run_tokenize(ctx.obj, resume=resume_tokenize)
    run_train(ctx.obj, resume=resume_train, **train_kwargs)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cli()
