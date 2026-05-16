# Training Loop Unification Plan

## Problem

The codebase has **three** implementations of essentially the same training logic:

| Path | Location | Lines | Key Difference |
|------|----------|-------|----------------|
| **Single-node** | `Trainer.fit()` + `Trainer.evaluate()` | ~90 lines | Uses `DataLoader`, `Trainer` state object |
| **Distributed** | `_train_func()` | ~100 lines | Ray dataset shards, `prepare_model()` for DDP, `train.report()` |
| **Tuning** | `_trainable_wrapper()` in `tuner.py` | ~30 lines | Ray Tune trials, hyperparameter search, `tune.report()` |

> Note: The line counts are lower than a naive count because significant shared infrastructure already exists (see below).

## What's Already Shared (Pre-Unified)

| Asset | Location | Status |
|---|---|---|
| `train_step()` / `val_step()` | `trainer.py` | ✅ Shared primitives |
| `compute_epoch_metrics()` | `trainer.py` | ✅ Shared post-processing |
| `publish_epoch_metrics()` | `metrics_publisher.py` | ✅ Centralized publishing |
| `_get_opt_params()` / `_get_sched_params()` | `trainer.py` | ✅ Centralized factories |
| `_LinearWarmupCosineScheduler` | `trainer.py` | ✅ Shared scheduler |
| `save_checkpoint()` / `load_checkpoint()` | `trainer.py` | ✅ Shared I/O |
| `MODEL_REGISTRY` | `models/base.py` | ✅ Lazy model registry |
| `compute_metrics_from_model()` | `metrics.py` | ✅ Unified evaluation |

## What's Actually Duplicated

1. **Training loop** (`for epoch in range(1, epochs+1): ...`) — 3×
2. **Evaluation loop** (`model.eval()`, `torch.no_grad()`, forward + collect) — 2.5×
3. **Model creation** (`if model_type == "rnn": ... elif ...`) — 3×
4. **Optimizer/scheduler/loss setup** — 3×
5. **Metrics publishing call site** — 3× (despite `publish_epoch_metrics()` being centralized)

## What's Different (Path-Specific)

| Concern | Single-node | Distributed | Tuning |
|---------|-------------|-------------|--------|
| Data loading | `DataLoader` from `CorpusDataset` | `train_shard.iter_torch_batches()` | `DataLoader` (via `_new_loaders`) |
| Model setup | Created in `new_trainer()` | `new_model()` + `prepare_model()` | `new_model()` (no DDP) |
| Checkpointing | `save_checkpoint()` to disk | `train.report(checkpoint=...)` | `tune.report()` (no checkpoint) |
| Metrics reporting | `publish_epoch_metrics()` | `publish_epoch_metrics()` from rank 0 | `publish_epoch_metrics()` + `tune.report()` |
| Early stopping | Yes (patience counter) | No | No |
| CUDA cleanup | `try/finally` in `Trainer.fit()` | None | None |

## Proposed Architecture

```
sentimentizer/trainer.py
  ├── train_step()            # (existing, shared)
  ├── val_step()              # (existing, shared)
  ├── compute_epoch_metrics() # (existing, shared)
  │
  ├── EpochResult              # NEW: dataclass for per-epoch results
  │     epoch, train_loss, val_loss, metrics, lr
  │
  ├── TrainingState            # NEW: mutable training state container
  │     val_loss, latest_train_loss, latest_epoch, latest_metrics,
  │     best_val_loss, patience_counter, running_loss_mean, steps
  │
  ├── LoopConfig               # NEW: thin wrapper over existing TrainerConfig
  │     cfg: TrainerConfig, callbacks: list[TrainingCallback], device: str
  │
  ├── Trainer                  # High-level orchestrator
  │   ├── fit()                # Single-node path (delegates to _run_training_loop)
  │   └── evaluate()           # Single-node validation (delegates to compute_metrics_from_model)
  │
  ├── _run_training_loop()     # NEW: Shared core loop (pure PyTorch)
  │   for epoch in range(1, epochs+1):
  │       train_loss = _train_epoch(model, train_iter, ...)
  │       val_loss, metrics = _evaluate(model, val_iter, ...)  # compute_metrics_from_model
  │       result = EpochResult(...)
  │       if scheduler: scheduler.step()
  │       for cb in callbacks:
  │           if cb.on_epoch_end(result):
  │               break
  │       else:
  │           continue
  │       break
  │
  ├── _iter_batches()          # NEW: Generator normalizing DataLoader vs Ray shard iteration
  │
  ├── create_model_from_registry()  # NEW: Unified model creation via MODEL_REGISTRY
  │
  ├── _create_training_components() # NEW: Optimizer + scheduler + loss factory
  │
  ├── TrainingCallback         # Protocol for hooking into the loop
  │   ├── on_epoch_end(result: EpochResult) -> bool
  │   ├── on_train_begin(config: LoopConfig) -> None
  │   └── on_train_end(state: TrainingState) -> None
  │
  ├── MetricsCallback(rank=0)  # publish_epoch_metrics; no-op when rank != 0
  ├── CheckpointCallback       # Periodic + best model saves
  ├── EarlyStoppingCallback    # Patience counter; returns bool
  ├── RayReportCallback        # train.report() + Checkpoint.from_directory() (all ranks)
  └── LoggingCallback(rank=0)  # Structured logging; no-op when rank != 0
```

## New / Revised Components

### `_iter_batches()` (replaces `DataAdapter` protocol)

```python
def _iter_batches(
    data_source: DataLoader | ray.data.Dataset,
    batch_size: int,
    device: str,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield (data, target) tensors regardless of source type."""
    if isinstance(data_source, DataLoader):
        for sent, target in data_source:
            yield sent.to(device), target.to(device)
    else:  # Ray dataset shard
        for batch in data_source.iter_torch_batches(batch_size=batch_size):
            yield batch["data"].long().to(device), batch["target"].float().to(device)
```

Rationale: Only two data sources exist. A full `DataAdapter` protocol with two wrapper classes is overkill for this level of divergence.

### `EpochResult` and `TrainingState` dataclasses

```python
from dataclasses import dataclass, field

@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: ClassificationMetrics
    lr: float

@dataclass
class TrainingState:
    val_loss: float = float("inf")
    latest_train_loss: float = 0.0
    latest_epoch: int = 0
    latest_metrics: ClassificationMetrics | None = None
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    running_loss_mean: float = 0.0
    steps: int = 0
```

`TrainingState` replaces the mutable `Trainer` instance attributes. It is returned by `_run_training_loop()` and can be stored in `Trainer` for backward compatibility.

### `LoopConfig` (replaces `TrainingConfig`)

```python
@dataclass
class LoopConfig:
    """Loop-level configuration wrapping the existing TrainerConfig."""
    cfg: TrainerConfig
    callbacks: list[TrainingCallback]
    device: str
```

Rationale: `TrainerConfig` already exists in `config.py` and is used throughout the codebase. Creating a parallel `TrainingConfig` would duplicate batch_size, epochs, device, etc. `LoopConfig` adds only loop-specific params (callbacks) to the existing config.

## Implementation Phases

### Phase 1: Foundation (~1.5 hours)

1. **Add `_iter_batches()` helper** — Normalize DataLoader vs Ray shard iteration in a single generator
2. **Add `create_model_from_registry()`** — Use existing `MODEL_REGISTRY` in `base.py` to replace inline `if/elif` blocks in `_train_func()` and `_trainable_wrapper()`
3. **Add `_create_training_components()`** — Extract optimizer + scheduler + loss setup from `new_trainer()`, `_train_func()`, and `_trainable_wrapper()`
4. **Fix unbounded `self.losses`** — Replace unbounded `list` with an O(1) iterative running mean calculation in `TrainingState`

### Phase 2: Unified Evaluation (~1 hour)

1. **Refactor `Trainer.evaluate()`** to delegate to existing `compute_metrics_from_model()` from `metrics.py`
2. **Add `_iter_batches()` support** so `compute_metrics_from_model()` can work with Ray shards (or create a lightweight shard wrapper)
3. **Remove duplicate `compute_metrics_from_model()` call** from `tuner.py:_trainable_wrapper()` — the unified loop will evaluate once and report metrics to both Tune and the callback

### Phase 3: Callback Protocol + Core Loop (~2.5 hours)

1. **Define `EpochResult` and `TrainingState`** dataclasses
2. **Define `TrainingCallback` protocol**
3. **Implement callbacks**:
   - `MetricsCallback` — wraps `publish_epoch_metrics()`
   - `CheckpointCallback` — periodic + best model saves
   - `EarlyStoppingCallback` — sets patience counter and returns a bool to break loop
   - `RayReportCallback` — `train.report()` + `Checkpoint.from_directory()` (rank-gated)
   - `LoggingCallback` — structured logger calls
4. **Create `_run_training_loop()`** — Pure PyTorch loop with callback hooks; returns `TrainingState`

### Phase 4: Refactor Callers (~1.5 hours)

1. **`Trainer.fit()`** → Create `LoopConfig` with `[MetricsCallback, CheckpointCallback, EarlyStoppingCallback, LoggingCallback]`, call `_run_training_loop()`, store returned `TrainingState` in `self` for backward compatibility. Wrap in `try/finally` for CUDA cleanup.
2. **`_train_func()`** → Get dataset shards, call `prepare_model()`, create `LoopConfig` with `[MetricsCallback(ray_gauges=...), RayReportCallback]`, call `_run_training_loop()`
3. **`_trainable_wrapper()`** → Drop `Trainer` entirely; create model directly, create `LoopConfig` with `[MetricsCallback, RayReportCallback]`, call `_run_training_loop()`. Pass `tune.report()` via a callback.

### Phase 5: Testing (~2 hours)

#### 5.1 Integration Validation

- Run existing test suite: `pytest tests/ -v --exitfirst --failed-first`
- Run single-node training: `python -m workflows.driver --model rnn --device cpu --run-type new train`
- Verify distributed path: `pytest tests/ -v -k "Ray"`
- Verify tuning path: `pytest tests/ -v -k "tune"`

#### 5.2 Mock Testing Strategy for Callbacks

The callback architecture is designed for testability. Every callback should be unit-testable in isolation, and `_run_training_loop()` should be testable with mock callbacks. Below is the mock testing pattern and specific test cases to add.

---

**Core Principle: Mock at the callback boundary**

`_run_training_loop()` is pure PyTorch + callbacks. To test it:
- Mock the model (return fixed logits)
- Mock the data (yield fixed batches)
- Mock the callbacks (record calls, return flags)
- Assert on callback invocation order and `EpochResult` contents

**Reusable mock patterns from existing tests:**

| Pattern | Location | Reuse for |
|---|---|---|
| `MagicMock()` model + `return_value=torch.randn(2, 1)` | `tests/test_training.py:1000` | Mock model in loop tests |
| `MagicMock()` loader + `__iter__.return_value` | `tests/test_training.py:1002` | Mock data iteration |
| `MagicMock()` gauges dict | `tests/test_rnn.py` | Mock Ray gauges in `MetricsCallback` tests |
| `monkeypatch.setattr` on `_METRICS_DIR` | `tests/test_training.py:666` | Mock JSON persistence path |
| `patch("ray.train.report")` | `tests/test_rnn.py` | Mock Ray reporting |
| `patch("ray.train.get_context")` | `tests/test_rnn.py` | Mock rank-0 gating |

---

**Test Cases to Add**

**A. `_run_training_loop()` with mock callbacks (`tests/test_training.py`)**

```python
class TestRunTrainingLoop:
    """Test _run_training_loop with mock callbacks."""

    def test_calls_all_callbacks_every_epoch(self) -> None:
        """All callbacks should receive on_epoch_end every epoch."""
        mock_cb = MagicMock()
        mock_cb.on_epoch_end.return_value = False  # don't stop

        # Run 3 epochs with mock model/data
        _run_training_loop(
            model=mock_model,
            train_iter=[(torch.zeros(2, 10), torch.ones(2, 1))],
            val_iter=[(torch.zeros(2, 10), torch.ones(2, 1))],
            epochs=3,
            callbacks=[mock_cb],
            ...
        )

        assert mock_cb.on_epoch_end.call_count == 3
        # Verify EpochResult passed to each call
        for call in mock_cb.on_epoch_end.call_args_list:
            result = call.args[0]
            assert isinstance(result, EpochResult)
            assert result.epoch in {1, 2, 3}

    def test_stops_when_callback_returns_true(self) -> None:
        """Loop should break early when a callback signals stop."""

        def stop_after_2(result: EpochResult) -> bool:
            return result.epoch >= 2

        mock_cb = MagicMock()
        mock_cb.on_epoch_end.side_effect = stop_after_2

        state = _run_training_loop(
            model=mock_model,
            train_iter=[...],
            val_iter=[...],
            epochs=5,
            callbacks=[mock_cb],
            ...
        )

        assert state.latest_epoch == 2  # stopped early
        assert mock_cb.on_epoch_end.call_count == 2

    def test_callbacks_run_in_registration_order(self) -> None:
        """Callbacks should fire in the order they were registered."""
        calls: list[str] = []

        class OrderedCallback:
            def __init__(self, name: str) -> None:
                self.name = name
            def on_epoch_end(self, result: EpochResult) -> bool:
                calls.append(self.name)
                return False

        _run_training_loop(
            ...,
            callbacks=[
                OrderedCallback("first"),
                OrderedCallback("second"),
                OrderedCallback("third"),
            ],
        )

        assert calls == ["first", "second", "third"] * num_epochs
```

**B. `EarlyStoppingCallback` unit tests (`tests/test_training.py`)**

```python
class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback in isolation."""

    def test_stops_when_patience_exceeded(self) -> None:
        cb = EarlyStoppingCallback(patience=2)
        # Simulate 3 epochs with no improvement
        result = EpochResult(epoch=3, val_loss=0.5, ...)
        cb.on_epoch_end(result)  # epoch 1: best=0.5, patience=0
        cb.on_epoch_end(result)  # epoch 2: patience=1
        cb.on_epoch_end(result)  # epoch 3: patience=2 → should_stop
        assert cb.should_stop is True

    def test_resets_patience_on_improvement(self) -> None:
        cb = EarlyStoppingCallback(patience=2)
        cb.on_epoch_end(EpochResult(epoch=1, val_loss=0.5, ...))
        cb.on_epoch_end(EpochResult(epoch=2, val_loss=0.4, ...))  # improved
        cb.on_epoch_end(EpochResult(epoch=3, val_loss=0.4, ...))  # no change
        assert cb.patience_counter == 1  # reset, then one bad epoch
        assert cb.should_stop is False
```

**C. `CheckpointCallback` unit tests (`tests/test_training.py`)**

```python
class TestCheckpointCallback:
    """Test CheckpointCallback without real disk I/O."""

    def test_saves_periodic_checkpoint(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(
            checkpoint_dir=tmp_path,
            checkpoint_every=2,
            checkpoint_best=False,
        )
        cb.on_epoch_end(EpochResult(epoch=2, val_loss=0.5, ...))
        assert (tmp_path / "checkpoint_epoch_2.pth").exists()

    def test_saves_best_model_on_improvement(self, tmp_path: Path) -> None:
        cb = CheckpointCallback(
            checkpoint_dir=tmp_path,
            checkpoint_every=0,
            checkpoint_best=True,
        )
        cb.on_epoch_end(EpochResult(epoch=1, val_loss=0.5, ...))
        cb.on_epoch_end(EpochResult(epoch=2, val_loss=0.4, ...))
        assert (tmp_path / "best_model.pth").exists()

    def test_skips_save_when_no_checkpoint_dir(self) -> None:
        cb = CheckpointCallback(checkpoint_dir=None)
        cb.on_epoch_end(EpochResult(epoch=1, ...))
        # Should not raise; just no-op
```

**D. `MetricsCallback` unit tests (`tests/test_training.py`)**

```python
class TestMetricsCallback:
    """Test MetricsCallback publishes to all backends."""

    def test_publishes_to_prometheus_and_json(self, tmp_path: Path) -> None:
        with patch("sentimentizer.metrics_publisher._set_prometheus_gauges") as mock_prom:
            cb = MetricsCallback(model_type="test", rank=0)  # rank 0
            cb.on_epoch_end(EpochResult(
                epoch=1,
                train_loss=0.3,
                val_loss=0.4,
                metrics=ClassificationMetrics(accuracy=0.8, ...),
                lr=0.001,
            ))

            mock_prom.assert_called_once()
            # Verify JSON file written
            json_path = tmp_path / "test_metrics.json"
            assert json_path.exists()

    def test_skips_when_rank_not_zero(self) -> None:
        """MetricsCallback must be a no-op on non-zero ranks."""
        with patch("sentimentizer.metrics_publisher._set_prometheus_gauges") as mock_prom:
            cb = MetricsCallback(model_type="test", rank=1)  # rank 1
            cb.on_epoch_end(EpochResult(
                epoch=1,
                train_loss=0.3,
                val_loss=0.4,
                metrics=ClassificationMetrics(accuracy=0.8, ...),
                lr=0.001,
            ))

            mock_prom.assert_not_called()

    def test_skips_ray_gauges_when_not_provided(self) -> None:
        cb = MetricsCallback(model_type="test", ray_gauges=None, rank=0)
        # Should not raise; Ray gauge publishing is skipped
```

**E. `RayReportCallback` unit tests (`tests/test_rnn.py` or new `test_ray_callbacks.py`)**

```python
class TestRayReportCallback:
    """Test RayReportCallback checkpointing and reporting.

    IMPORTANT: train.report() MUST be called by ALL workers, not just rank 0.
    Ray Train aggregates metrics internally from all workers. Rank gating here
    would break distributed checkpoint aggregation.
    """

    def test_reports_on_all_ranks(self) -> None:
        """train.report() should fire regardless of rank."""
        with patch("ray.train.get_context") as mock_ctx:
            mock_ctx.return_value.get_world_rank.return_value = 1  # rank 1

            cb = RayReportCallback()
            with patch("ray.train.report") as mock_report:
                cb.on_epoch_end(EpochResult(epoch=1, ...))
                mock_report.assert_called_once()

    def test_includes_checkpoint_data(self) -> None:
        """Reported dict must contain all metrics and a checkpoint."""
        with (
            patch("ray.train.report") as mock_report,
            patch("tempfile.TemporaryDirectory"),
            patch("ray.train.Checkpoint.from_directory"),
        ):
            cb = RayReportCallback()
            cb.on_epoch_end(EpochResult(
                epoch=3,
                train_loss=0.3,
                val_loss=0.4,
                metrics=ClassificationMetrics(accuracy=0.8, ...),
                lr=0.001,
            ))

            args = mock_report.call_args[0][0]
            assert args["epoch"] == 3
            assert args["train_loss"] == 0.3
            assert args["accuracy"] == 0.8
            assert "checkpoint" in mock_report.call_args[1]
```

**F. `_iter_batches()` unit tests (`tests/test_training.py`)**

```python
class TestIterBatches:
    """Test _iter_batches normalizes both DataLoader and Ray shard iteration."""

    def test_dataloader_source(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.zeros(4, 10), torch.ones(4))
        loader = DataLoader(dataset, batch_size=2)

        batches = list(_iter_batches(loader, batch_size=2, device="cpu"))
        assert len(batches) == 2
        assert all(isinstance(b, tuple) and len(b) == 2 for b in batches)

    def test_ray_shard_source(self) -> None:
        mock_shard = MagicMock()
        mock_shard.iter_torch_batches.return_value = [
            {"data": torch.zeros(2, 10), "target": torch.ones(2)},
        ]
        batches = list(_iter_batches(mock_shard, batch_size=2, device="cpu"))
        assert len(batches) == 1
        data, target = batches[0]
        assert data.dtype == torch.long   # .long() applied
        assert target.dtype == torch.float  # .float() applied
```

---

**Where Mock Tests Are Missing (Pre-Unification)**

| Component | Current Test Coverage | Mock Test Gap |
|---|---|---|
| `Trainer._train_epoch()` | Indirect via `Trainer.fit()` | No isolated test with mock data |
| `_train_func()` | `test_rnn.py` has one integration-style mock test | No unit test for the loop itself; only end-to-end |
| `EarlyStoppingCallback` | N/A (doesn't exist yet) | Full mock suite needed |
| `CheckpointCallback` | N/A (doesn't exist yet) | Full mock suite needed |
| `MetricsCallback` | N/A (doesn't exist yet) | Full mock suite needed |
| `RayReportCallback` | N/A (doesn't exist yet) | Full mock suite needed |
| `_create_training_components()` | N/A (doesn't exist yet) | Mock test with different model types |
| `create_model_from_registry()` | N/A (doesn't exist yet) | Mock test verifying registry lookup |

**Recommended new test file: `tests/test_callbacks.py`**

Create `tests/test_callbacks.py` to house all callback unit tests in one place. This keeps `test_training.py` focused on training primitives and integration tests, while `test_callbacks.py` covers the new callback protocol and implementations.

```python
# tests/test_callbacks.py
"""Unit tests for TrainingCallback implementations.

All tests use MagicMock to avoid real model training, disk I/O,
and Ray dependencies. This ensures fast, deterministic test execution.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from sentimentizer.trainer import (
    CheckpointCallback,
    EarlyStoppingCallback,
    EpochResult,
    MetricsCallback,
    RayReportCallback,
    TrainingState,
)
```


## Benefits

- **~100 lines removed** (training loop, evaluation loop, model creation, optimizer/scheduler/loss setup duplicated 3×)
- **Single source of truth** for training logic — bug fixes apply everywhere
- **Consistent behavior** between single-node, distributed, and tuning paths
- **Better testability** — `_run_training_loop()` is pure PyTorch and testable with mock callbacks
- **Bounded memory** — The O(1) iterative running mean fixes the unbounded losses bug

---

## Detailed Risk Analysis

### RISK 1: Trainer.evaluate() Mutates Instance State

**Severity: HIGH**

`Trainer.evaluate()` mutates five instance attributes:
- `self.val_loss`, `self.latest_train_loss`, `self.latest_epoch`, `self.latest_metrics`, `self.losses`
- External consumers: `tuner.py`, `workflows/stages/train.py`, tests

**Mitigation Strategy**: `TrainerStateCallback` explicitly writes state back to the `Trainer` object during Phase 1-2 to preserve backward compatibility. Eventually, `Trainer.fit()` stores the returned `TrainingState` into `self`. The `Trainer` API surface (attributes) remains unchanged; only the internal implementation changes.

### RISK 2: tuner.py Calls Private Trainer Methods Directly

**Severity: MEDIUM**

`_trainable_wrapper()` calls `trainer._train_epoch()` (private!) and `trainer.evaluate()`, then reads `trainer.val_loss` and `trainer.losses[-1]`.

**Mitigation Strategy**: Solved organically in Phase 3 when `_trainable_wrapper()` is refactored to call the new `_run_training_loop()` directly with Tune-specific callbacks. This simplifies the tuner code.

### RISK 3: Ray-Specific Patterns Can't Be Extracted

**Severity: MEDIUM**

`_train_func()` uses `prepare_model()`, `train.report()`, `train.get_context().get_world_rank()`, `train.get_dataset_shard()`, `Checkpoint.from_directory()`.

**Mitigation Strategy**: Isolate `train.report()`, `get_dataset_shard()`, and `prepare_model()` entirely within `RayReportCallback` and `_train_func()`. `_run_training_loop()` is pure PyTorch.

### RISK 4: Scheduler Step Ordering

**Severity: LOW**

All three paths call `scheduler.step()` after validation. Already consistent.

**Mitigation Strategy**: Document in `_run_training_loop()` docstring.

### RISK 5: Early Stopping Must Break Epoch Loop

**Severity: MEDIUM**

`Trainer.fit()` interleaves checkpointing and early stopping. Callbacks typically can't break loops directly.

**Mitigation Strategy**: Handled by `EarlyStoppingCallback.on_epoch_end()` returning a `bool`. `_run_training_loop()` checks this return value and breaks the loop if true. Tested via `MagicMock` in Phase 4.

### RISK 6: CUDA Cleanup in finally Block

**Severity: MEDIUM**

`Trainer.fit()` has `try/finally` with `torch.cuda.synchronize()` + `torch.cuda.empty_cache()`.

**Mitigation Strategy**: `Trainer.fit()` wraps `_run_training_loop()` in a `try/finally` block that strictly enforces `torch.cuda.empty_cache()`.

### RISK 7: Unbounded self.losses List

**Severity: LOW**

`_train_epoch()` appends every step's loss to `self.losses`. For 200 epochs × 1000 steps, this is 200k floats (~1.6 MB). Worse for longer runs.

**Mitigation Strategy**: Addressed in Phase 1 by calculating an O(1) iterative running mean instead of storing the unbounded list.

### RISK 8: Test Dependencies on Trainer Instance State

**Severity: MEDIUM**

Tests access `trainer.val_loss`, `trainer.losses` after training.

**Mitigation Strategy**: Preserve `Trainer` instance attributes by copying `TrainingState` fields back into `Trainer` after `_run_training_loop()` returns. Tests see no behavioral change.

### RISK 9: Different Data Iteration Patterns

**Severity: MEDIUM**

`DataLoader` yields tuples; Ray shards yield dicts with `.long().to(device)` conversion needed.

**Mitigation Strategy**: `_iter_batches()` generator with a single `isinstance` check. No new types or wrapper classes needed.

### RISK 10: Loss Function Device Handling

**Severity: LOW**

`pos_weight` tensor is created on different devices in different paths.

**Mitigation Strategy**: `_create_training_components()` always moves `pos_weight` to the correct device.

### RISK 11: Rank Gating for Metrics and Logging Callbacks

**Severity: HIGH**

In the distributed `_train_func()`, `MetricsCallback` (writes to `metrics.json` and pushes to Prometheus) and `LoggingCallback` (writes stdout logs) could be executed simultaneously across all workers. This will cause file-write collisions, duplicated Prometheus metrics, and severe log spam.

**Mitigation Strategy**: In Phase 4, `_train_func()` must explicitly check the worker rank (`train.get_context().get_world_rank() == 0`) before appending `MetricsCallback` and `LoggingCallback` to the `LoopConfig`. 

### RISK 12: DDP Wrapping vs. Optimizer Instantiation Order

**Severity: HIGH**

In distributed mode (`_train_func`), PyTorch requires the `Optimizer` to be instantiated **after** the model is wrapped in `DistributedDataParallel` via `prepare_model()`. If `_create_training_components()` initializes the optimizer beforehand, gradient synchronization will fail.

**Mitigation Strategy**: Phase 4 must strictly enforce the instantiation sequence in `_train_func()`: 1) create the model, 2) call `prepare_model(model)` to wrap it, and 3) invoke `_create_training_components(model, ...)` so the optimizer tracks the correct DDP references.

## Risk Summary

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | `Trainer.evaluate()` mutates instance state | HIGH | `TrainerStateCallback` writes back to `Trainer` object in Phase 1-2 |
| 2 | `tuner.py` calls private `Trainer` methods | MEDIUM | Solved organically in Phase 3 refactor to use `_run_training_loop()` |
| 3 | Ray-specific patterns leak | MEDIUM | Isolate entirely within `RayReportCallback` and `_train_func()` |
| 4 | Scheduler step ordering | LOW | Already consistent; document in docstring |
| 5 | Early stopping must break epoch loop | MEDIUM | Handled by `EarlyStoppingCallback` returning `bool`. Tested via `MagicMock` in Phase 4 |
| 6 | CUDA cleanup missed | MEDIUM | `Trainer.fit()` wraps loop in `try/finally` enforcing `empty_cache()` |
| 7 | Unbounded `self.losses` list | LOW | Addressed in Phase 1 by calculating an O(1) iterative running mean |
| 8 | Tests depend on `Trainer` instance state | MEDIUM | Copy `TrainingState` back to `Trainer` attributes |
| 9 | Different data iteration patterns | MEDIUM | `_iter_batches()` generator |
| 10 | Loss function device handling | LOW | `_create_training_components()` factory |
| 11 | Rank Gating for Metrics and Logging Callbacks | HIGH | Explictly rank check in `_train_func()` before appending callbacks |
| 12 | DDP Wrapping vs. Optimizer Instantiation Order | HIGH | Enforce model creation -> `prepare_model()` -> optimizer instantiation order |

## Estimated Effort

| Phase | Hours | Description |
|-------|-------|-------------|
| Phase 1 | ~1.5 | Foundation (`_iter_batches`, model registry, training components factory, O(1) running mean) |
| Phase 2 | ~1 | Unified evaluation (`compute_metrics_from_model` for all paths) |
| Phase 3 | ~2.5 | Callback protocol + core loop |
| Phase 4 | ~1.5 | Refactor callers and test via `MagicMock` |
| Phase 5 | ~1 | Testing & validation |
| **Total** | **~7.5** | |

## Recommended Implementation Order

Each sub-phase should be a separate commit/PR with full test coverage:

1. **Phase 1a**: Add `_iter_batches()` + tests
2. **Phase 1b**: Add `create_model_from_registry()` + tests; replace inline blocks
3. **Phase 1c**: Add `_create_training_components()` + tests; replace duplicated setup
4. **Phase 1d**: Fix `self.losses` → O(1) running mean + tests
5. **Phase 2**: Refactor `Trainer.evaluate()` to use `compute_metrics_from_model()` + remove tuner duplicate
6. **Phase 3a**: Define `EpochResult`, `TrainingState`, `TrainingCallback` protocol
7. **Phase 3b**: Implement `MetricsCallback`, `CheckpointCallback`, `EarlyStoppingCallback`
8. **Phase 3c**: Implement `RayReportCallback`, `LoggingCallback`
9. **Phase 3d**: Create `_run_training_loop()` with callback hooks
10. **Phase 4a**: Refactor `Trainer.fit()` to use `_run_training_loop()`
11. **Phase 4b**: Refactor `_train_func()` to use `_run_training_loop()`
12. **Phase 4c**: Refactor `_trainable_wrapper()` to use `_run_training_loop()`
13. **Phase 4d**: Test callbacks using `MagicMock`
14. **Phase 5**: Full test suite + training loop validation