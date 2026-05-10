"""Backward-compatibility shim.

All public symbols are now in sub-modules. This file re-exports them
so that existing imports like ``from workflows.driver import cli`` and
``from workflows.driver import State`` continue to work.

New code should import from the canonical locations:
  - ``workflows.cli`` for CLI commands
  - ``workflows.lifecycle`` for State, env setup, and cleanup
  - ``workflows.helpers`` for utility functions
  - ``workflows.stages.*`` for pipeline stage functions
"""

# Re-export CLI entry point
from workflows.cli import cli  # noqa: F401

# Re-export shared state
from workflows.lifecycle import State  # noqa: F401

# Re-export pipeline stage functions
from workflows.stages.diagnose import run_diagnose_env, run_diagnose_pipeline  # noqa: F401
from workflows.stages.extract import run_extract  # noqa: F401
from workflows.stages.hf import run_hf_pull, run_hf_push  # noqa: F401
from workflows.stages.tokenize import run_tokenize  # noqa: F401
from workflows.stages.train import run_train  # noqa: F401
from workflows.stages.tune import run_tune  # noqa: F401

if __name__ == "__main__":
    cli()
