# Project Rules

## After making code changes, always run tests

After modifying any source file in `sentimentizer/` or `workflows/`, run the test suite to verify nothing is broken:

```bash
uv run pytest tests/ -v --exitfirst --failed-first
```

If the change affects Ray Train or distributed training specifically, also run:

```bash
uv run pytest tests/ -v -k "Ray"
```

## Dependency management

This project uses **uv** for dependency management. Key commands:

```bash
# Install all dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Run a command in the virtual environment
uv run <command>
```

## Project structure

- `sentimentizer/` — Core library (models, trainer, tokenizer, extractor, loader, config)
- `workflows/` — CLI driver for the extract → tokenize → train pipeline
- `tests/` — Pytest tests
- `serve.py` — Ray Serve deployment
- `k8s/` — Kubernetes manifests

## Key conventions

- All data pipeline stages (extract, tokenize) use Ray Data (`ray.data`)
- Training supports both single-node (`Trainer`) and distributed (`TorchTrainer` via `--distributed` flag)
- Models are created on each Ray Train worker from a config dict (cannot pass PyTorch models across workers)
- Checkpoints use `Checkpoint.from_dict()` with `model.module.state_dict()` (unwrapping DDP)
- Config classes live in `sentimentizer/config.py` — use dataclasses
- Keep `ray.init(ignore_reinit_error=True)` in tests to avoid re-init errors
- The project requires Python 3.11+ (pinned in `.python-version`)