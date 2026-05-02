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

# Install with dev dependencies (includes ruff, black, pytest)
uv sync --extra dev

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Run a command in the virtual environment
uv run <command>
```

## Linting and formatting

This project uses **ruff** for linting and **black** for code formatting.

```bash
# Run ruff linter
uv run ruff check .

# Auto-fix ruff issues
uv run ruff check --fix .

# Check formatting with black (dry run)
uv run black --check .

# Apply black formatting
uv run black .
```

Both ruff and black are run in CI — PRs must pass lint and format checks before merging.

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

## Code quality principles

### Always use type hints

- All function and method signatures **must** include type hints for parameters and return values
- Use Python's `typing` module for complex types (e.g., `Optional`, `Union`, `Callable`, `TypeVar`)
- Class attributes should also be annotated with types
- Example:

```python
def tokenize(self, text: str, max_length: int = 512) -> list[int]:
    ...
```

### Follow DRY (Don't Repeat Yourself)

- Extract duplicated or near-duplicated logic into shared functions, classes, or mixins
- Prefer composition over copy-paste — if two modules need the same behavior, factor it into a utility or base class
- Configuration values that appear in multiple places should be defined once (e.g., in `config.py`) and referenced elsewhere

### Follow SOLID principles

- **Single Responsibility**: Each class/module should have one reason to change. Keep data loading, model definition, training, and serving logic in separate modules
- **Open/Closed**: Design classes to be extended (via subclassing or configuration) without modifying existing code. Use abstract base classes or protocols where appropriate
- **Liskov Substitution**: Subtypes must be substitutable for their base types. Don't override methods in ways that break the contract of the parent class
- **Interface Segregation**: Prefer small, focused interfaces (protocols/ABCs) over large, general-purpose ones
- **Dependency Inversion**: Depend on abstractions (protocols, ABCs) rather than concrete implementations. Inject dependencies via constructors or function arguments rather than creating them internally
