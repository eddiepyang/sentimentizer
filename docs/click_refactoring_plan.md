# Refactoring `workflows/driver.py` with Click for Lazy Loading

This document outlines the architectural plan to refactor the main CLI driver to use `click`. Goals: faster CLI feedback (`--help` and lightweight subcommands no longer load the ML stack), per-command flag scoping with native validation, and **strict behavior parity** with the existing `argparse` driver.

## Step 1: Restructure into a Command Group

Replace the single `argparse` namespace with a `click.Group` and per-stage subcommands:

- `extract` — pull raw reviews into parquet
- `tokenize` — build/update dictionary and write processed parquet
- `train` — fit (single-node or distributed)
- `tune` — hyperparameter tuning (`--mode agent|standalone`)
- `hf push` / `hf pull` — Hugging Face Hub operations
- `diagnose` — pipeline health checks
- **`run` — chain `extract → tokenize → train`** (preserves the current default invocation)

`run` is mandatory: today `python driver.py --type new` runs the full pipeline in one call. Without `run`, every existing user has to learn three commands and the Makefile has to grow three times.

## Step 2: Scope Arguments to Specific Commands

Common flags that affect multiple stages stay on the root group:

- `--model {rnn,encoder,decoder}` — used by `tokenize`, `train`, `tune`, `hf`, `diagnose`
- `--device {auto,cuda,mps,cpu}` — used by `train`, `tune`
- `--run-type {new,update}` — used by `extract`, `tokenize`, `train` (controls fresh-build vs. reuse). **Renamed from `--type`** to avoid collision with `tokenize --type` in the original draft.

Stage-specific flags attach to their command (full inventory in the appendix).

## Step 3: Defer Heavy Imports (Lazy Loading)

`torch`, `ray`, and `gensim` move out of the module top and into the command bodies that use them. `--help` and per-subcommand `--help` render without importing the ML stack.

**`diagnose` is a group, not a single command.** `diagnose_training_issues` from `sentimentizer.agent.skill` transitively imports torch, so we split:

- `diagnose env` — fast: prints CUDA / Ray / Python versions, validates env vars and paths. No torch / ray imports.
- `diagnose pipeline` — heavy: runs the existing `diagnose_training_issues` checks (dictionary alignment, embedding shape, class balance, token-id range).

Bare `sentimentizer diagnose` prints the group help. The old `python driver.py --diagnose` maps to `sentimentizer diagnose pipeline`.

## Step 4: Centralize Ray Init and Environment Setup

Today `main()` runs `ray.init(_metrics_export_port=8080, runtime_env={...})` once with `LD_LIBRARY_PATH` propagated to workers ([driver.py:891-897](../workflows/driver.py#L891-L897)). Splitting commands must not regress this — `extract` and `tokenize` use `ray.data`, which auto-inits Ray *without* the metrics port or LD path → Prometheus scrape would silently break and CUDA libs wouldn't reach workers.

A single helper handles this for every Ray-using command:

```python
def _ensure_ray_initialized() -> None:
    import ray
    if ray.is_initialized():
        return
    from sentimentizer.env import ensure_nvidia_ld_library_path
    ld_path = ensure_nvidia_ld_library_path()
    runtime_env = {"env_vars": {"LD_LIBRARY_PATH": ld_path}} if ld_path else {}
    _cleanup_stale_ray_sessions()
    ray.init(_metrics_export_port=8080, runtime_env=runtime_env)
```

Pre-`ray.init` env vars (`RAY_GRAFANA_HOST`, `RAY_PROMETHEUS_HOST`, `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION`, `RAY_ENABLE_UV_RUN_RUNTIME_ENV`) stay at module top — they must be set *before* anything imports `ray`. The original draft only listed `RAY_ENABLE_UV_RUN_RUNTIME_ENV`.

**Responsibility rule:** every `run_*` helper that touches `ray.data` or `ray.train` calls `_ensure_ray_initialized()` as its first line. Click command shells stay thin (one-line forwarders) and don't call it themselves — that way you can't accidentally use a helper from another helper or a test without bringing Ray with it.

### Lazy device resolution

`auto_detect_device()` lives in `sentimentizer/config.py`, which `import torch`s at module top. Calling it from the group callback — or even importing it there — forces torch to load every time click invokes the group callback, which **includes every `<subcommand> --help`** (only the bare `sentimentizer --help` skips the callback). That undoes Step 3.

Fix: do not resolve in the group. Store `state.device = "auto"` as-is. Commands that need a concrete device (`train`, `tune`) resolve it themselves — they import torch anyway, so the resolution is free.

For the resolver to be importable without dragging in torch, extract it to a torch-free module — e.g. `sentimentizer/device.py`:

```python
# sentimentizer/device.py — no module-level torch import
def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

`config.py`'s `auto_detect_device` becomes a re-export (`from sentimentizer.device import resolve_device as auto_detect_device`) so existing call sites keep working. New click commands import from `sentimentizer.device` directly.

## Step 5: Preserve Cleanup Handlers

The current driver registers `_cuda_cleanup` / `_ray_cleanup` with `atexit`, installs a SIGINT handler, and calls `_cleanup_stale_ray_sessions()` at startup ([driver.py:163-181, 885](../workflows/driver.py#L163-L181)). These are load-bearing — CLAUDE.md flags `/tmp/ray` growth as a recurring 5+ GB-per-session problem. They stay at module load:

```python
# Module top — runs once per process regardless of subcommand
atexit.register(_cuda_cleanup)
atexit.register(_ray_cleanup)
signal.signal(signal.SIGINT, _sigint_handler)
```

`_cleanup_stale_ray_sessions()` runs inside `_ensure_ray_initialized()` (just before `ray.init`), not at module top — that keeps `--help` instant.

## Step 6: Hard Cutover for Flags, No Silent Aliases

The flag surface changes (`--type` → `--run-type`, `--hf-pull` → `hf pull`, `--agent-tune` removed). Old invocations like `python driver.py --type train ...` will **not** be silently aliased.

Migration plan:

1. Land the click driver, update the `Makefile` and any internal scripts in the **same PR**.
2. Communicate the new commands and remove the original argparse parser.
3. (Optional) Expose `python -m workflows.driver` as a thin shim that prints the equivalent new command and exits non-zero — helps stragglers without faking compatibility.

The original plan's claim "the old invocation will still work during migration" was wrong and is removed.

## Step 7: Add `console_scripts` Entry Point

```toml
[project.scripts]
sentimentizer = "workflows.driver:cli"
```

Enables:

```bash
sentimentizer run --stop 5000              # full pipeline (default workflow)
sentimentizer train --distributed --save   # train only
sentimentizer hf push --repo-id user/model
sentimentizer diagnose
```

## Step 8: CLI Tests

`tests/test_cli.py` covers:

- `sentimentizer --help` renders without importing `torch`/`ray` (asserted via `sys.modules` in a fresh subprocess — see Testing section).
- Each subcommand's `--help` is non-empty and lists its expected flags.
- `click.Choice` rejects invalid `--model`, `--run-type`, `tune --mode`.
- `hf` errors when no subcommand is given.
- One mocked `run` test that verifies `extract → tokenize → train` chaining.

## Step 9: Decide on `exporter.py` CLI

Leave as-is. The exporter is a long-lived HTTP process, not a pipeline step. Revisit if it grows more subcommands.

---

## Conceptual Code Architecture

```python
import atexit
import os
import signal
import sys
from dataclasses import dataclass

import click
from dotenv import load_dotenv

# 1. Environment setup — must happen before ANY ray/torch import
load_dotenv()
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION", "0.75")
os.environ.setdefault("RAY_GRAFANA_HOST", "http://localhost:3000")
os.environ.setdefault("RAY_PROMETHEUS_HOST", "http://localhost:9090")

# Note: do NOT call logging.basicConfig() here or in the cli callback.
# Logging is configured by structlog in sentimentizer/__init__.py at import
# time; basicConfig wouldn't reach structlog's logger factory and would
# silently shadow the existing config for any stdlib-logging consumers.


# 2. Cleanup handlers — registered once at module load
def _cuda_cleanup() -> None:
    """Guard: skip if torch was never imported (lazy-loading safe)."""
    if "torch" not in sys.modules:
        return
    import torch
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

def _ray_cleanup() -> None:
    """Guard: skip if ray was never imported (lazy-loading safe)."""
    if "ray" not in sys.modules:
        return
    import ray
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    # ... stale session cleanup omitted for brevity (see Step 5) ...

def _cleanup_stale_ray_sessions() -> None: ...
def _sigint_handler(signum, frame) -> None: ...

atexit.register(_cuda_cleanup)
atexit.register(_ray_cleanup)
signal.signal(signal.SIGINT, _sigint_handler)


# 3. Shared Ray bootstrap — every run_* helper calls this first
def _ensure_ray_initialized() -> None:
    import ray
    if ray.is_initialized():
        return
    from sentimentizer.env import ensure_nvidia_ld_library_path
    ld_path = ensure_nvidia_ld_library_path()
    runtime_env = {"env_vars": {"LD_LIBRARY_PATH": ld_path}} if ld_path else {}
    _cleanup_stale_ray_sessions()
    ray.init(_metrics_export_port=8080, runtime_env=runtime_env)


# 4. Typed shared state (replaces raw dict ctx.obj)
@dataclass
class State:
    model: str
    device: str  # raw value: "auto" | "cuda" | "mps" | "cpu" — resolve lazily
    run_type: str  # "new" | "update"


# 5. Shared decorator: keeps `train` and `run` flag surfaces in lockstep.
# `--resume` is intentionally NOT in this list because `run` exposes it as
# split flags (--resume-tokenize / --resume-train) while `train` keeps the
# unified `--resume`. Each command applies @click.option("--resume", ...)
# itself.
def shared_train_options(func):
    options = [
        click.option("--distributed", is_flag=True),
        click.option("--num-workers", default=1, type=int),
        click.option("--save/--no-save", default=False),
        click.option("--checkpoint-dir", default=""),
        click.option("--checkpoint-every", default=1, type=int),
        click.option("--balance-classes", is_flag=True),
        click.option("--balance-seed", default=42, type=int),
        click.option("--weight-smoothing", default=0.5, type=float),
        click.option("--loss-type", default="cross_entropy",
                      type=click.Choice(["cross_entropy", "focal"])),
        click.option("--label-smoothing", default=0.1, type=float),
        click.option("--neutral-oversample-ratio", default=0.0, type=float),
        click.option("--push-to-hub", is_flag=True),
        click.option("--pull-from-hub", is_flag=True),
        click.option("--hf-repo", default=None),
    ]
    for option in reversed(options):
        func = option(func)
    return func


# 6. Root group — must NOT import torch, ray, or sentimentizer.config
@click.group()
@click.option("--model", default="rnn",
              type=click.Choice(["rnn", "encoder", "decoder"]),
              envvar="SENTIMENTIZER_MODEL")
@click.option("--device", default="auto", envvar="SENTIMENTIZER_DEVICE",
              help="auto, cuda, mps, or cpu (resolved lazily by commands that need torch)")
@click.option("--run-type", default="new",
              type=click.Choice(["new", "update"]),
              help="new = build/train fresh; update = reuse existing artifacts")
@click.pass_context
def cli(ctx, model, device, run_type):
    """Sentimentizer ML pipeline."""
    # Store raw values only. Do NOT call auto_detect_device here or import
    # sentimentizer.config — it transitively imports torch, which would
    # trigger torch loading for every `<subcommand> --help`.
    ctx.obj = State(model=model, device=device, run_type=run_type)


# 7. Subcommands — thin shells that forward to run_* helpers.
# Each helper owns _ensure_ray_initialized() if it touches ray.

@cli.command()
@click.option("--stop", default=10000, type=int)
@click.pass_context
def extract(ctx, stop):
    """Extract raw reviews into parquet."""
    run_extract(ctx.obj, stop=stop)


@cli.command()
@click.option("--resume", is_flag=True,
              help="Update the existing dictionary from new data")
@click.pass_context
def tokenize(ctx, resume):
    """Build/update dictionary and write processed parquet."""
    run_tokenize(ctx.obj, resume=resume)


@cli.command()
@shared_train_options
@click.option("--resume", is_flag=True,
              help="Resume training from latest checkpoint in --checkpoint-dir")
@click.pass_context
def train(ctx, resume, **kwargs):
    """Fit the model (single-node or distributed)."""
    run_train(ctx.obj, resume=resume, **kwargs)


@cli.command()
@click.option("--mode", default="agent",
              type=click.Choice(["agent", "standalone"]))
@click.option("--agent-config", default=None)
@click.option("--samples", "tune_samples", default=20, type=int)
@click.option("--max-iterations", "tune_max_iterations", default=5, type=int)
@click.option("--output-dir", "tune_output_dir", default="tuning_results")
@    click.option("--no-validate", is_flag=True)
@click.option("--validation-threshold", default=0.75, type=float)
@click.option("--max-retries", default=2, type=int)
@click.option("--save/--no-save", default=False)
@click.option("--push-to-hub", is_flag=True)
@click.option("--balance-classes", is_flag=True)
@click.option("--balance-seed", default=42, type=int)
@click.option("--weight-smoothing", default=0.5, type=float)
@click.option("--loss-type", default="cross_entropy",
              type=click.Choice(["cross_entropy", "focal"]))
@click.option("--label-smoothing", default=0.1, type=float)
@click.option("--neutral-oversample-ratio", default=0.0, type=float)
@click.pass_context
def tune(ctx, mode, **kwargs):
    """Hyperparameter tuning (LLM-guided agent or standalone Ray Tune run).

    Replaces both `--tune` (TuningRun skill) and `--agent-tune`. The legacy
    `--agent-tune` path that called run_agent_tuning directly is removed;
    `tune --mode agent` covers it via TuningRun.
    """
    run_tune(ctx.obj, mode=mode, **kwargs)


@cli.group()
def hf():
    """Hugging Face Hub operations."""


@hf.command("push")
@click.option("--repo-id", default=None)
@click.pass_context
def hf_push(ctx, repo_id):
    run_hf_push(ctx.obj, repo_id=repo_id)


@hf.command("pull")
@click.option("--repo-id", default=None)
@click.pass_context
def hf_pull(ctx, repo_id):
    run_hf_pull(ctx.obj, repo_id=repo_id)


@cli.group()
def diagnose():
    """Pipeline diagnostics."""


@diagnose.command("env")
@click.pass_context
def diagnose_env(ctx):
    """Fast environment check (Python / CUDA paths / Ray env vars). No torch/ray."""
    run_diagnose_env(ctx.obj)


@diagnose.command("pipeline")
@click.pass_context
def diagnose_pipeline(ctx):
    """Heavy pipeline check. Imports the ML stack."""
    run_diagnose_pipeline(ctx.obj)


@cli.command()
@click.option("--stop", default=10000, type=int)
@click.option("--resume-tokenize", is_flag=True,
              help="Update existing dictionary during tokenize stage")
@click.option("--resume-train", is_flag=True,
              help="Resume training from checkpoint")
@shared_train_options
@click.pass_context
def run(ctx, stop, resume_tokenize, resume_train, **train_kwargs):
    """Run the full pipeline: extract → tokenize → train.

    Mirrors today's default `python driver.py --type new` invocation.
    The single legacy `--resume` flag is split into stage-specific flags
    so each stage's resume can be controlled independently.
    """
    run_extract(ctx.obj, stop=stop)
    run_tokenize(ctx.obj, resume=resume_tokenize)
    run_train(ctx.obj, resume=resume_train, **train_kwargs)


if __name__ == "__main__":
    cli()
```

### Composition pattern: helpers, not `ctx.invoke`

Each subcommand is a thin shell that bundles its options into kwargs and calls a helper function (`run_extract`, `run_tokenize`, `run_train`, etc.). `run` calls the same helpers directly. We **do not** use `ctx.invoke` for command composition.

`ctx.invoke(train, save=True)` does technically work — Click fills in defaults from `train`'s `params` for anything you don't pass — but it's the wrong tool here:

1. **Silent default drift.** If someone changes `train`'s `--num-workers` default from 1 to 4, every `run` invocation silently goes distributed-by-4 with no test failure. There's no link between `run`'s flag surface and what `ctx.invoke` injects.
2. **Hidden coupling.** `run`'s behavior depends on `train`'s decorator stack. Refactoring `train`'s options breaks `run` in non-obvious ways.
3. **No type checking.** `ctx.invoke` is `**kwargs`-typed; static analysis can't catch a misnamed parameter.

The helper-function pattern matches the current argparse driver's structure ([`run_extract`, `run_tokenize`, `_run_fit_single` / `_run_fit_distributed`](../workflows/driver.py#L440)) — keep it.

```python
# Business logic — no click decorators, fully typed.
# Helpers that touch ray.data / ray.train call _ensure_ray_initialized()
# as their first line.
def run_extract(state: State, *, stop: int) -> None: ...
def run_tokenize(state: State, *, resume: bool) -> None: ...
def run_train(state: State, *, distributed: bool, num_workers: int,
              save: bool, checkpoint_dir: str, checkpoint_every: int,
              resume: bool, balance_classes: bool, balance_seed: int,
              weight_smoothing: float, loss_type: str, label_smoothing: float,
              neutral_oversample_ratio: float,
              push_to_hub: bool, pull_from_hub: bool,
              hf_repo: str | None) -> None: ...
def run_tune(state: State, *, mode: str, agent_config: str | None,
             tune_samples: int, tune_max_iterations: int, tune_output_dir: str,
             no_validate: bool, validation_threshold: float, max_retries: int,
             save: bool, push_to_hub: bool, balance_classes: bool,
             balance_seed: int, weight_smoothing: float, loss_type: str,
             label_smoothing: float, neutral_oversample_ratio: float) -> None: ...
def run_hf_push(state: State, *, repo_id: str | None) -> None: ...
def run_hf_pull(state: State, *, repo_id: str | None) -> None: ...
def run_diagnose_env(state: State) -> None: ...        # no torch/ray
def run_diagnose_pipeline(state: State) -> None: ...   # imports ML stack

# Click commands — one-liners that forward to helpers
@cli.command()
@shared_train_options
@click.option("--resume", is_flag=True)
@click.pass_context
def train(ctx, resume, **kwargs):
    run_train(ctx.obj, resume=resume, **kwargs)
```

Two side benefits of `shared_train_options`:

- `train` and `run` cannot drift — adding/renaming a flag updates both call sites at once.
- The helpers (`run_train`, `run_tune`) become a unit-testable surface independent of click. Without `run_tune`, tune would be the one stage you can only test through `CliRunner`.

---

## Migration Table

| Old invocation | New invocation |
|---|---|
| `python workflows/driver.py --type new` | `sentimentizer run` |
| `python workflows/driver.py --type new --stop 5000` | `sentimentizer run --stop 5000` |
| `python workflows/driver.py --type new --save` | `sentimentizer run --save` |
| `python workflows/driver.py --type update --save` | `sentimentizer --run-type update train --save` |
| `python workflows/driver.py --type new --distributed --num-workers 4 --save` | `sentimentizer run --distributed --save` (or `train` directly for more flags) |
| `python workflows/driver.py --type extract --stop 5000` | `sentimentizer extract --stop 5000` |
| `python workflows/driver.py --type tokenize` | `sentimentizer tokenize` |
| `python workflows/driver.py --tune --tune-mode agent` | `sentimentizer tune --mode agent` |
| `python workflows/driver.py --agent-tune` | `sentimentizer tune --mode agent` *(`--agent-tune` removed)* |
| `python workflows/driver.py --diagnose` | `sentimentizer diagnose pipeline` *(bare `diagnose` prints group help)* |
| `python workflows/driver.py --hf-push --repo-id user/model` | `sentimentizer hf push --repo-id user/model` |
| `python workflows/driver.py --hf-pull --repo-id user/model` | `sentimentizer hf pull --repo-id user/model` |
| `python workflows/driver.py --type train --pull-from-hub` | `sentimentizer train --pull-from-hub` (single invocation, same semantics as today) |
| `python workflows/driver.py --type new --resume` | `sentimentizer run --resume-tokenize --resume-train` *(see resume-flag split below)* |
| `python workflows/driver.py --type tokenize --resume` | `sentimentizer tokenize --resume` |
| `python workflows/driver.py --type train --resume` | `sentimentizer train --resume` |

### Resume-flag split (semantic change)

Today's `--resume` does two unrelated things depending on the stage:

- In tokenize, `args.resume` triggers `tokenizer.update_from_dataset(...)` ([driver.py:482-486](../workflows/driver.py#L482-L486)) — i.e., extend the dictionary with new vocabulary.
- In train, `args.resume` loads the latest checkpoint ([driver.py:543-553](../workflows/driver.py#L543-L553)).

When a user ran `--type new --resume`, both fired together. The new `run` command exposes `--resume-tokenize` and `--resume-train` so each stage can be controlled independently. `tokenize` and `train` keep a single `--resume` (unambiguous in context). This is a deliberate UX improvement, not a rename — call it out in release notes.

The `Makefile` and any CI scripts must be updated in the same PR.

---

## Behavior Parity Checklist

For implementers — items the new CLI must match against the old driver before merge:

- [ ] Every `run_*` helper that touches `ray.data` or `ray.train` calls `_ensure_ray_initialized()` as its first line (`run_extract`, `run_tokenize`, `run_train`, `run_tune`, `run_diagnose_pipeline`). Click command shells stay one-line forwarders — they do **not** call it themselves. Init uses `_metrics_export_port=8080` and propagates `LD_LIBRARY_PATH` via `runtime_env`.
- [ ] `_cleanup_stale_ray_sessions()` runs immediately before each `ray.init()`.
- [ ] `atexit` handlers (`_cuda_cleanup`, `_ray_cleanup`) and the SIGINT handler are registered at module load time. **Both `_cuda_cleanup` and `_ray_cleanup` must guard against their dependency not being imported yet** — e.g. `_cuda_cleanup` should check `if "torch" not in sys.modules: return` before calling `torch.cuda.is_available()`, and `_ray_cleanup` should check `if "ray" not in sys.modules: return` before calling `ray.is_initialized()`. Without these guards, the atexit handlers would force a torch/ray import at process exit even for `sentimentizer diagnose env` or `sentimentizer --help`, defeating lazy loading.
- [ ] `train --pull-from-hub` pulls before fit and forces `run_type=update` (preserves [driver.py:858-877](../workflows/driver.py#L858-L877)).
- [ ] `train --push-to-hub` only fires when `--save` is also set (preserves current gating in `_run_fit_single` / `_run_fit_distributed`).
- [ ] `tokenize --resume` updates the existing dictionary; `--run-type update` (without `--resume`) reuses it without updating.
- [ ] `run --resume-tokenize` triggers `update_from_dataset` in the tokenize stage; `run --resume-train` loads the latest checkpoint in train. Each is independent.
- [ ] `run_diagnose_env` does **not** import torch, ray, or `sentimentizer.config` (verified by extending the parametrized `--help` test to also invoke `diagnose env` itself, not just its `--help`).
- [ ] `extract` / `tokenize` skip when parquet row count ≥ `--stop`.
- [ ] `--device auto` is **stored unresolved** on `ctx.obj`; `train` / `tune` (or any torch-using command) resolve it via `sentimentizer.device.resolve_device()` after they've already imported torch. Group callback must not import `sentimentizer.config` or `torch`.
- [ ] `<subcommand> --help` does not import torch or ray — verify with the subprocess test in the Testing section, parameterized over every subcommand.
- [ ] `--agent-tune` removed; `tune --mode agent` produces the `best_config.json` artifact and the same logged `agent_tuning_complete` keys.
- [ ] `RAY_GRAFANA_HOST`, `RAY_PROMETHEUS_HOST`, `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION` are set at module top, before any `import ray`.
- [ ] `tune` with `--save` writes the model to `tune_output_dir` and (if `--push-to-hub`) pushes it.
- [ ] CUDA cleanup runs in `finally` blocks around `trainer.fit()` and `ray_trainer.fit()`.
- [ ] `--hf-repo` defaults to `None` in the Click CLI (instead of `DriverConfig.hf.repo_id`); `run_hf_push` / `run_hf_pull` / `_run_fit_single` / `_run_fit_distributed` must resolve `None` to `DriverConfig.hf.repo_id` internally. This avoids importing `DriverConfig` (which transitively imports torch) at module top, but preserves the behavior where omitting `--hf-repo` uses the configured default.

---

## Testing Strategy

```python
import subprocess
import sys

from click.testing import CliRunner

from workflows.driver import cli


SUBCOMMANDS_TO_CHECK = [
    [],                              # bare group help
    ["extract", "--help"],
    ["tokenize", "--help"],
    ["train", "--help"],
    ["tune", "--help"],
    ["hf", "--help"],
    ["hf", "push", "--help"],
    ["hf", "pull", "--help"],
    ["diagnose", "--help"],
    ["diagnose", "env", "--help"],
    ["diagnose", "env"],             # actually run the fast path — must not import ML stack
    ["diagnose", "pipeline", "--help"],
    ["run", "--help"],
]


@pytest.mark.parametrize("argv", SUBCOMMANDS_TO_CHECK)
def test_help_does_not_import_ml_stack(argv):
    """Every <subcommand> --help must render without pulling in torch or ray.

    Critical: click runs the group callback before subcommand help, so a
    single `import torch` (or `from sentimentizer.config import ...`) in
    the group callback poisons every subcommand's --help latency.

    Spawned in a fresh subprocess because pytest's other tests may have
    already imported torch/ray in the current process.
    """
    args_repr = repr(argv if argv else ["--help"])
    code = (
        "import sys, time\n"
        "from workflows.driver import cli\n"
        "from click.testing import CliRunner\n"
        f"start = time.time()\n"
        f"result = CliRunner().invoke(cli, {args_repr})\n"
        f"duration = time.time() - start\n"
        "assert result.exit_code == 0, result.output\n"
        "assert 'torch' not in sys.modules, 'torch leaked'\n"
        "assert 'ray' not in sys.modules, 'ray leaked'\n"
        "assert 'sentimentizer.config' not in sys.modules, 'config (which imports torch) leaked'\n"
        f"assert duration < 1.0, f'help took {{duration:.2f}}s — likely slow import leak'\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_train_help_lists_flags():
    result = CliRunner().invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--distributed" in result.output
    assert "--checkpoint-dir" in result.output


def test_invalid_model_rejected():
    result = CliRunner().invoke(cli, ["--model", "invalid", "train"])
    assert result.exit_code != 0


def test_hf_requires_subcommand():
    result = CliRunner().invoke(cli, ["hf"])
    assert result.exit_code != 0


def test_diagnose_env_no_ml_imports():
    """Verify that actually running `diagnose env` (not just --help) imports
    neither torch nor ray. This is separate from the parametrized
    test_help_does_not_import_ml_stack because that test covers --help
    rendering, while this one covers command execution."""
    code = (
        "import sys, time\n"
        "from workflows.driver import cli\n"
        "from click.testing import CliRunner\n"
        "start = time.time()\n"
        "result = CliRunner().invoke(cli, ['diagnose', 'env'])\n"
        "duration = time.time() - start\n"
        "assert result.exit_code == 0, result.output\n"
        "assert 'torch' not in sys.modules, 'torch leaked'\n"
        "assert 'ray' not in sys.modules, 'ray leaked'\n"
        "assert 'sentimentizer.config' not in sys.modules, 'config leaked'\n"
        "assert duration < 1.0, f'diagnose env took {duration:.2f}s'\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_run_chains_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr("workflows.driver.run_extract", lambda *a, **kw: calls.append("extract"))
    monkeypatch.setattr("workflows.driver.run_tokenize", lambda *a, **kw: calls.append("tokenize"))
    monkeypatch.setattr("workflows.driver.run_train", lambda *a, **kw: calls.append("train"))
    result = CliRunner().invoke(cli, ["run", "--stop", "10"])
    assert result.exit_code == 0
    assert calls == ["extract", "tokenize", "train"]
```

---

## Why this revision is safer than the original draft

1. **Pipeline preserved.** `sentimentizer run` keeps the default `extract → tokenize → train` workflow that today's `python driver.py --type new` provides — without it, every existing user would have to learn three commands.
2. **Ray init centralized.** `_ensure_ray_initialized()` handles metrics port + runtime_env for every Ray-using command — `extract`/`tokenize` no longer silently lose Prometheus scraping or LD path propagation.
3. **Cleanup preserved.** atexit/SIGINT handlers stay at module load; `_cleanup_stale_ray_sessions` runs before each `ray.init` so `/tmp/ray/` doesn't fill the disk.
4. **Flag collision removed.** Global `--type` is renamed to `--run-type`; tokenize no longer has a clashing per-command `--type`.
5. **Hard cutover, named.** No false claim of argparse fallback. Makefile and scripts are part of the same PR; an optional shim prints the new command and exits.
6. **Diagnose honesty.** No false speed claim; offers an explicit `env`-only split if a fast path is wanted later.
7. **Pre-init env vars complete.** `RAY_GRAFANA_HOST`, `RAY_PROMETHEUS_HOST`, `RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION` are documented as module-top requirements.
8. **Behavior parity checklist.** Implementers have a concrete list to verify against, including the subtle items (`--push-to-hub` gated on `--save`, `--pull-from-hub` forcing `run_type=update`, `tokenize --resume` semantics).
9. **`shared_train_options` decorator.** `train` and `run` share a single options list — adding/renaming a flag updates both call sites at once, eliminating the drift risk that would exist if `run`'s flags were copy-pasted from `train`'s.
10. **Resume-flag split.** Today's single `--resume` controls two unrelated things (dictionary update + checkpoint restore). `run --resume-tokenize` / `run --resume-train` makes each independently controllable; `tokenize` / `train` keep a unified `--resume` since each is unambiguous in isolation.
11. **No false `logging.basicConfig`.** The cli callback does not configure stdlib logging — the project uses structlog, configured at import time in `sentimentizer/__init__.py`. A `basicConfig` call in the callback would fire on every `<sub> --help`, wouldn't reach structlog's logger factory, and would silently shadow the existing config for any stdlib-logging consumers.

---

## Appendix: Per-subcommand flag inventory

| Subcommand | Flags |
|---|---|
| (root) | `--model`, `--device`, `--run-type` |
| `extract` | `--stop` |
| `tokenize` | `--resume` |
| `train` | `shared_train_options` + `--resume` |
| `tune` | `--mode`, `--agent-config`, `--samples`, `--max-iterations`, `--output-dir`, `--no-validate`, `--validation-threshold`, `--max-retries`, `--save/--no-save`, `--push-to-hub`, `--balance-classes`, `--balance-seed`, `--weight-smoothing`, `--loss-type`, `--label-smoothing`, `--neutral-oversample-ratio` |
| `hf push` | `--repo-id` |
| `hf pull` | `--repo-id` |
| `diagnose` (group) | (group itself takes no flags) |
| `diagnose env` | (none beyond root) — fast, no torch/ray |
| `diagnose pipeline` | (none beyond root) — heavy, imports ML stack |
| `run` | `--stop`, `--resume-tokenize`, `--resume-train`, `shared_train_options`; calls `run_extract` / `run_tokenize` / `run_train` helpers directly, not via `ctx.invoke` |

**`shared_train_options`** expands to: `--distributed`, `--num-workers`, `--save/--no-save`, `--checkpoint-dir`, `--checkpoint-every`, `--balance-classes`, `--balance-seed`, `--weight-smoothing`, `--loss-type`, `--label-smoothing`, `--neutral-oversample-ratio`, `--push-to-hub`, `--pull-from-hub`, `--hf-repo`. Applied to both `train` and `run` so the surfaces stay locked together. `--resume` is intentionally not in the shared list — `run` exposes it as split flags, `train` adds it back with command-specific help text.

**Envvar precedence:** CLI flag > envvar (`SENTIMENTIZER_MODEL`, `SENTIMENTIZER_DEVICE`) > default. This differs from the current driver, which only reads `os.getenv` ad-hoc — call this out in migration notes if any flags change resolution order.
