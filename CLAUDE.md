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
- Config classes live in `sentimentizer/config.py` — use dataclasses
- Keep `ray.init(ignore_reinit_error=True)` in tests to avoid re-init errors
- The project requires Python 3.11+ (pinned in `.python-version`)

## Ray 2.55 API conventions

This project uses **Ray 2.55.1**. The API has changed significantly from earlier versions.
Always verify against the installed source in `.venv/lib/python3.11/site-packages/ray/` if unsure.

### Ray Data (`ray.data.Dataset`)

- **`Dataset` objects are NOT iterable.** Never use `for row in ds:` or unpack them.
  Use `ds.iter_rows()`, `ds.iter_batches()`, or `ds.iter_torch_batches()` instead.
- **Split datasets** with `ds.train_test_split(test_size=0.2, shuffle=True, seed=42)`.
  Returns `(train_ds, val_ds)` tuple of `MaterializedDataset`.
  Do NOT use `ds.random_split()` — it does not exist on `Dataset`.
- **Sample a fraction** with `ds.random_sample(fraction, seed=42)`.
  Returns a single `Dataset` (NOT a tuple). Use for undersampling majority classes.
- **Filter rows** with `ds.filter(fn=)` or `ds.filter(expr=)` (prefer `expr` for performance).
- **Shuffle** with `ds.random_shuffle(seed=42)`.
- **Concatenate** with `ds.union(other_ds)`.
- **Count rows** with `ds.count()` (materializes the dataset).

### Ray Train (`ray.train`)

- **Checkpoints are directory-based only.** `Checkpoint.from_dict()` and `Checkpoint.to_dict()`
  were **removed in Ray 2.55+**. Use the directory-based API instead:

  ```python
  # Writing a checkpoint (inside a training function)
  import os, tempfile
  import ray.cloudpickle as pickle
  from ray.train import Checkpoint

  checkpoint_data = {
      "model_state_dict": model.module.state_dict(),  # unwrap DDP
      "optimizer_state_dict": optimizer.state_dict(),
      "epoch": epoch,
  }
  with tempfile.TemporaryDirectory() as checkpoint_dir:
      with open(os.path.join(checkpoint_dir, "data.pkl"), "wb") as fp:
          pickle.dump(checkpoint_data, fp)
      checkpoint = Checkpoint.from_directory(checkpoint_dir)
      tune.report({...}, checkpoint=checkpoint)
  ```

  ```python
  # Reading a checkpoint (on the driver)
  with result.checkpoint.as_directory() as checkpoint_dir:
      with open(os.path.join(checkpoint_dir, "data.pkl"), "rb") as fp:
          checkpoint_data = pickle.load(fp)
      model_state_dict = checkpoint_data["model_state_dict"]
  ```

- **`train.get_context()`** only works inside a Ray Train worker function
  (launched by `trainer.fit()`). Never call it from the driver process.
- **`prepare_model()`** wraps a model with DDP. Access the original model
  via `model.module` (e.g., `model.module.state_dict()`).

### Ray Tune (`ray.tune`)

- Use `tune.report({...})` to report metrics from trainables.
- Use `tune.Tuner` (not `tune.run`, which is deprecated).
- Use `tune.with_parameters()` to pass large objects (datasets) to trainables.
- Use `tune.with_resources()` to specify resource requirements.

### Common pitfalls

- **Never iterate a `Dataset` directly.** `for x in ds` raises `TypeError`.
- **Never use `ds.random_split()`.** It does not exist. Use `ds.train_test_split()`
  for splitting or `ds.random_sample()` for fractional sampling.
- **Never use `Checkpoint.from_dict()` or `Checkpoint.to_dict()`.** They were removed.
  Use `Checkpoint.from_directory()` / `checkpoint.as_directory()` with pickle.
- **Never call `train.get_context()` outside a worker.** It raises `RuntimeError`.
- **`random_sample(fraction)` returns a single `Dataset`, not a tuple.**
  Do not unpack it like `keep, _ = ds.random_sample(0.5)`.

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
