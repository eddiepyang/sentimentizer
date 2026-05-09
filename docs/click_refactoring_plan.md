# Refactoring `workflows/driver.py` with Click for Lazy Loading

This document outlines the architectural plan to refactor the main CLI driver to use `click`. This enables dramatically faster CLI responsiveness and cleaner command isolation by removing heavy inline imports from the top-level script.

## Step 1: Restructure into a Command Group
Currently, `argparse` collects every possible argument (for extraction, training, tuning, diagnostics) into one massive namespace, and the script uses `if/elif` blocks to figure out what to execute.

With `click`, we create a main `Group` and attach separate subcommands (`extract`, `tokenize`, `train`, `tune`, `diagnose`).

## Step 2: Scope Arguments to Specific Commands
Instead of having all flags available globally (even when they don't apply to the current run type), we attach options only to the commands that actually use them. Common options (like `--model` and `--device`) are attached to the root group and passed down via `click.Context`.

## Step 3: Defer Heavy Imports (Lazy Loading)
Heavy libraries (`torch`, `ray`, `gensim`) are removed from the top of the file and placed inside the specific command functions. This guarantees that lightweight commands like `--help` or `--diagnose` execute in milliseconds instead of waiting for the full machine learning stack to initialize.

---

## Conceptual Code Architecture

```python
import os
import atexit
import signal
from pathlib import Path
import click
from dotenv import load_dotenv

# 1. Environment setup (FAST - keep at the top)
load_dotenv()
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from sentimentizer.env import ensure_nvidia_ld_library_path
nvidia_ld_library_path = ensure_nvidia_ld_library_path()

# 2. Base Group & Shared State
@click.group()
@click.option("--model", default="rnn", type=click.Choice(["rnn", "encoder", "decoder"]))
@click.option("--device", default="auto", help="auto, cuda, mps, or cpu")
@click.pass_context
def cli(ctx, model, device):
    """Sentimentizer ML Pipeline."""
    # ctx.ensure_object(dict) passes shared state to subcommands
    ctx.ensure_object(dict)
    ctx.obj['model'] = model
    ctx.obj['device'] = device

# 3. Subcommands (Lazy Loaded)
@cli.command()
@click.option("--stop", default=10000, help="how many lines to load")
@click.pass_context
def extract(ctx, stop):
    """Extract raw dataset from archives."""
    # Heavy imports deferred until execution!
    import ray
    from sentimentizer.extractor import extract_data
    
    # Execution logic...
    click.echo(f"Extracting data for {ctx.obj['model']}...")

@cli.command()
@click.option("--type", "run_type", default="new", type=click.Choice(["new", "update"]))
@click.option("--resume", is_flag=True)
@click.pass_context
def tokenize(ctx, run_type, resume):
    """Process text into numerical tokens."""
    import ray
    from gensim import corpora
    from sentimentizer.tokenizer import Tokenizer
    # Execution logic...

@cli.command()
@click.option("--distributed", is_flag=True)
@click.option("--num-workers", default=1)
@click.option("--save/--no-save", default=False)
@click.pass_context
def train(ctx, distributed, num_workers, save):
    """Train the model."""
    import ray
    import torch
    
    # Initialize Ray if needed
    if distributed and not ray.is_initialized():
        ray.init(_metrics_export_port=8080)
        
    # Proceed to _run_fit_distributed or _run_fit_single
    click.echo(f"Training {ctx.obj['model']} on {ctx.obj['device']}...")

@cli.command()
@click.option("--mode", default="agent", type=click.Choice(["agent", "standalone"]))
@click.pass_context
def tune(ctx, mode):
    """Run hyperparameter tuning."""
    # Heavy LLM Agent or Ray Tune imports go here
    from sentimentizer.agent.skill import TuningRun, TuningRunConfig
    # Execution logic...

if __name__ == "__main__":
    cli()
```

---

## Why this is safer and cleaner:
1. **Instant CLI Feedback:** Since `torch` and `ray` are completely deferred into the commands, validation of flags and `--help` menus will render instantly.
2. **Eliminate `RunTypeError`:** We no longer need custom exceptions to manage conflicting flags. `click.Choice` automatically validates user input natively.
3. **No Unintended Side Effects:** You currently have to carefully manage `args.diagnose` so it runs before `ray.init()`. With `click`, `driver.py diagnose` is fully sandboxed in its own function.