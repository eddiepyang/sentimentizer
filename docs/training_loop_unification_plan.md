# Training Loop Unification Plan

## Problem

The codebase has **three** implementations of essentially the same training logic:

| Path | Location | Lines | Key Difference |
|------|----------|-------|----------------|
| **Single-node** | `Trainer.fit()` + `Trainer.evaluate()` | ~130 lines | Uses `DataLoader`, `Trainer` state object |
| **Distributed** | `_train_func()` | ~100 lines | Ray dataset shards, `prepare_model()` for DDP, `train.report()` |
| **Tuning** | `_trainable_wrapper()` in `tuner.py` | ~80 lines | Ray Tune trials, hyperparameter search, `tune.report()` |

## What's Duplicated (Same Logic, 3×)

1. **Optimizer + scheduler setup** — AdamW, CosineAnnealing/WarmupCosine, BCEWithLogitsLoss with pos_weight
2. **Training loop** — `for epoch in range(1, epochs+1): train_step() × N batches`
3. **Validation loop** — `model.eval()`, `torch.no_grad()`, forward + sigmoid + collect probabilities/targets
4. **Metrics computation** — `compute_epoch_metrics()` (NaN replacement + torchmetrics)
5. **Metrics publishing** — `publish_epoch_metrics()` (Ray gauges, Prometheus, JSON, logger)
6. **Model creation** — `new_model()` with embeddings config (repeated in 4 places)

## What's Different (Path-Specific)

| Concern | Single-node | Distributed | Tuning |
|---------|-------------|-------------|--------|
| Data loading | `DataLoader` from `CorpusDataset` | `train_shard.iter_torch_batches()` | `train_shard.iter_torch_batches()` |
| Model setup | Created in `new_trainer()` | `new_model()` + `prepare_model()` | `new_model()` (no DDP) |
| Checkpointing | Periodic + best model saves | `train.report(checkpoint=...)` | `train.report(checkpoint=...)` |
| Metrics reporting | `publish_epoch_metrics()` | `publish_epoch_metrics()` from rank 0 | `publish_epoch_metrics()` from rank 0 |
| Loss function | `BCEWithLogitsLoss(pos_weight=...)` | Same, built inline | Same, built inline |
| Scheduler | Created in `new_trainer()` | Created inline with warmup | Created inline with warmup |
| Early stopping | Yes (patience counter) | No | No |
| Device management | `.to(device)` in `fit()` | `next(model.parameters()).device` | `model.to(device)` then `.device` |
| CUDA cleanup | `try/finally` with `cuda.synchronize()` + `cuda.empty_cache()` | None | None |

## Proposed Architecture

```
sentimentizer/trainer.py
  ├── train_step()            # (existing, shared)
  ├── val_step()              # (existing, shared)
  ├── compute_epoch_metrics() # (existing, shared)
  │
  ├── TrainingConfig          # Dataclass with ALL training hyperparams
  │   (epochs, lr, betas, weight_decay, pos_weight, scheduler params,
  │    early_stopping_patience, checkpoint_dir, device, ...)
  │
  ├── Trainer                 # High-level orchestrator (replaces current Trainer.fit())
  │   ├── fit()               # Single-node path (delegates to _run_training_loop)
  │   └── evaluate()          # Single-node validation
  │
  ├── _run_training_loop()    # Shared core loop
  │   Args: model, optimizer, scheduler, loss_fn,
  │         train_data, val_data, epochs, device, model_type,
  │         callbacks: list[TrainingCallback]
  │
  │   for epoch in range(1, epochs+1):
  │       train_losses = _train_epoch(model, train_iter, ...)
  │       val_losses, metrics = _evaluate(model, val_iter, ...)
  │       if scheduler: scheduler.step()
  │       for cb in callbacks: cb.on_epoch_end(epoch, metrics, ...)
  │
  ├── TrainingCallback        # Protocol/ABC for hooking into the loop
  │   ├── on_epoch_end(epoch, metrics, train_loss, val_loss, lr)
  │   ├── on_train_begin(config)
  │   └── on_train_end(metrics)
  │
  ├── MetricsCallback(TrainingCallback)     # publish_epoch_metrics
  ├── CheckpointCallback(TrainingCallback)  # periodic + best model saves
  ├── EarlyStoppingCallback(TrainingCallback)
  ├── RayReportCallback(TrainingCallback)   # train.report() for Ray Train/Tune
  └── LoggingCallback(TrainingCallback)     # structured logging
```

## Implementation Phases

### Phase 1: Extract Shared Functions (~2 hours)

1. **Extract `_evaluate()` function** from `Trainer.evaluate()`
   - Takes `(model, val_iter, loss_fn, device)` → returns `(val_loss, ClassificationMetrics)`
   - Works with both `DataLoader` and Ray shard iterators
   - Currently ~30 lines duplicated between `Trainer.evaluate()` and `_train_func()`

2. **Extract `TrainingConfig` dataclass** from scattered params in `Trainer`, `_train_func()`, `_trainable_wrapper()`
   - Move `pos_weight`, `scheduler_type`, `warmup_steps`, etc. into one place
   - Both `new_trainer()` and `_train_func()` build from the same config

3. **Unify model creation** — Replace 4× duplicated `if model_type == "rnn": ... elif ...` blocks with the `MODEL_REGISTRY` from `base.py`

4. **Unify optimizer/scheduler creation** — Extract `_create_optimizer()` and `_create_scheduler()` functions that work from `TrainingConfig`

### Phase 2: Add TrainingCallback Protocol (~3 hours)

1. **Create `TrainingCallback` protocol**
   ```python
   from typing import Protocol

   class TrainingCallback(Protocol):
       def on_epoch_end(self, epoch: int, metrics: EpochResult) -> None: ...
       def on_train_begin(self, config: TrainingConfig) -> None: ...
       def on_train_end(self, metrics: EpochResult) -> None: ...
   ```

2. **Create callback implementations**:
   - `MetricsCallback` — calls `publish_epoch_metrics()` (replaces both the inline gauge-setting in `_train_func()` and the `publish_epoch_metrics()` call in `Trainer.evaluate()`)
   - `CheckpointCallback` — periodic + best model saves (extracted from `Trainer.fit()`)
   - `EarlyStoppingCallback` — patience counter (extracted from `Trainer.fit()`)
   - `RayReportCallback` — calls `train.report()` + `Checkpoint.from_directory()` (extracted from `_train_func()`)
   - `LoggingCallback` — structured logger calls

### Phase 3: Refactor Callers (~2 hours)

1. **Create unified `_run_training_loop()`**
   - Shared epoch loop with callback hooks
   - Called by both `Trainer.fit()` and `_train_func()`
   - Each path passes different callbacks

2. **Refactor callers**:
   - `Trainer.fit()` → creates `_run_training_loop()` with `[MetricsCallback(), CheckpointCallback(), EarlyStoppingCallback(), LoggingCallback()]`
   - `_train_func()` → creates `_run_training_loop()` with `[MetricsCallback(ray_gauges=...), RayReportCallback()]`
   - `_trainable_wrapper()` → creates `_run_training_loop()` with `[MetricsCallback(), RayReportCallback(), TuningCallback()]`

### Phase 4: Testing & Validation (~1 hour)

- Run existing test suite (`pytest tests/ -v`)
- Run single-node training loop (`python -m workflows.driver --model rnn --device cpu --run-type new train`)
- Verify distributed training still works (`TestRayDistributed` tests)
- Verify tuning path still works (Ray Tune tests)

## Benefits

- **~200 lines removed** (validation loop, metrics publishing, model creation duplicated 3-4×)
- **Single source of truth** for training logic — bug fixes apply everywhere
- **Easier to add** new features (gradient accumulation, mixed precision) — add callback or modify one loop
- **Consistent behavior** between single-node, distributed, and tuning paths
- **Better testability** — can unit test `_run_training_loop()` with mock callbacks

---

## Detailed Risk Analysis

### RISK 1: Trainer.evaluate() Mutates Instance State

**Severity: HIGH**

`Trainer.evaluate()` (line ~450) mutates **four** instance attributes every call:

| Attribute | Set To | External Consumers |
|---|---|---|
| `self.val_loss` | `float(np.mean(losses))` | `tuner.py:467`, `workflows/stages/train.py`, tests |
| `self.latest_train_loss` | `float(np.mean(self.losses))` | `workflows/stages/train.py` |
| `self.latest_epoch` | `epoch` | Internal to `Trainer.fit()` |
| `self.latest_metrics` | `ClassificationMetrics` | `workflows/stages/train.py` |
| `self.losses` (appended in `_train_epoch`) | `list[float]` | `tuner.py:476` reads `trainer.losses[-1]` |

**Impact**: If we extract evaluation into a callback that doesn't mutate `Trainer` state, `tuner.py` and `workflows/stages/train.py` will break because they read `trainer.val_loss`, `trainer.losses`, etc.

**Mitigation**:
- Option A: Keep `Trainer` as a state container. `_run_training_loop()` updates `Trainer` instance fields via a `TrainerStateCallback` that writes back to the `Trainer` object. This preserves backward compatibility.
- Option B: Return an `EpochResult` dataclass from `_run_training_loop()`. `Trainer.fit()` stores it in `self.latest_result`. Update callers to use `trainer.latest_result.val_loss` instead of `trainer.val_loss`.
- **Recommended**: Option A for Phase 1-2 (minimal disruption), migrate to Option B in a later phase.

### RISK 2: tuner.py Calls Private Trainer Methods Directly

**Severity: MEDIUM**

`_trainable_wrapper()` in `tuner.py` (lines ~461-485) creates a `Trainer` object and calls:
- `trainer._train_epoch(model, train_loader, epoch)` (private method!)
- `trainer.evaluate(model, val_loader, epoch)` (public method)
- Then reads `trainer.val_loss` and `trainer.losses[-1]`

**Impact**: If we refactor `Trainer.fit()` to use `_run_training_loop()` with callbacks, `_trainable_wrapper()` needs to either:
1. Also use `_run_training_loop()` with appropriate callbacks, OR
2. Continue calling the decomposed methods directly

**Mitigation**: In Phase 3, refactor `_trainable_wrapper()` to use `_run_training_loop()` with `[MetricsCallback(), RayReportCallback()]`. This actually simplifies the tuner code since it won't need to create a `Trainer` object at all.

### RISK 3: Ray-Specific Patterns Can't Be Extracted

**Severity: MEDIUM**

`_train_func()` uses several Ray-specific patterns that can't live in shared code:

| Pattern | Location | Can Be Extracted? |
|---|---|---|
| `prepare_model(model)` | ~line 620 | No — must be called in Ray worker context |
| `train.report({...})` | ~line 680 | No — Ray Train API |
| `train.get_context().get_world_rank()` | ~line 665 | No — Ray Train API |
| `train.get_dataset_shard("train")` | ~line 630 | No — Ray Data API |
| `Checkpoint.from_directory()` | ~line 685 | No — Ray Train API |

**Impact**: The shared `_run_training_loop()` cannot contain any Ray imports or Ray-specific logic.

**Mitigation**: All Ray-specific code stays in callback implementations (`RayReportCallback`) and the `_train_func()` wrapper (which handles `prepare_model()` and dataset sharding). The core loop is pure PyTorch. This is already the plan — just needs careful review during implementation.

### RISK 4: Scheduler Step Ordering Differences

**Severity: LOW**

Both `Trainer.fit()` and `_train_func()` call `scheduler.step()` **after** `evaluate()` (i.e., after validation, at the end of each epoch). The ordering is consistent:
```
for epoch in range(1, epochs+1):
    train_epoch()
    evaluate()
    scheduler.step()  # <-- same position in both paths
```

**Impact**: No risk — ordering is already consistent. The `_trainable_wrapper()` in `tuner.py` creates its own scheduler and also steps after training, so all three paths are aligned.

**Mitigation**: Document the expected scheduler step position in `_run_training_loop()` docstring.

### RISK 5: Early Stopping Interleaves with Checkpointing

**Severity: MEDIUM**

`Trainer.fit()` has interleaved logic after each epoch:
```python
if checkpoint_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
    save_checkpoint(model, self.optimizer, epoch, ckpt_path)  # periodic checkpoint

if checkpoint_dir and checkpoint_best and self.val_loss < best_val_loss:
    save_checkpoint(model, self.optimizer, epoch, best_path)  # best model checkpoint

if self.cfg.early_stopping_patience > 0:
    if self.val_loss < best_val_loss:
        best_val_loss = self.val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= self.cfg.early_stopping_patience:
            break  # early stopping
```

**Impact**: Early stopping must be able to **break the epoch loop**. Callbacks typically can't break loops — they can only signal "should stop".

**Mitigation**: `EarlyStoppingCallback.on_epoch_end()` returns a `bool` (continue/break). `_run_training_loop()` checks the return value:
```python
for epoch in range(1, epochs + 1):
    ...
    for cb in callbacks:
        should_stop = cb.on_epoch_end(epoch, ...)
        if should_stop:
            break
    if should_stop:
        break
```
This is a common pattern (Keras does this). The `CheckpointCallback` doesn't need to break the loop, so it returns `False` (continue).

### RISK 6: CUDA Cleanup in finally Block

**Severity: MEDIUM**

`Trainer.fit()` has a `try/finally` block that calls:
```python
if self.cfg.device in ("cuda", "mps") and torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

**Impact**: This cleanup must not be lost during refactoring. If `_run_training_loop()` doesn't have a `finally` block, GPU memory could leak on Ctrl-C or exceptions.

**Mitigation**: Add a `try/finally` wrapper in `Trainer.fit()` that calls `_run_training_loop()` inside the try, and performs CUDA cleanup in the finally. The `_run_training_loop()` function itself should NOT contain device-specific cleanup — that's a caller responsibility. Alternatively, add an `on_train_end()` callback that handles cleanup.

### RISK 7: Unbounded `self.losses` List

**Severity: LOW**

`_train_epoch()` appends every step's loss to `self.losses`, which grows unboundedly. For long training runs (200+ epochs × 1000+ steps), this could consume significant memory.

**Impact**: This is a pre-existing bug, not introduced by the refactoring. But the refactoring is a good opportunity to fix it.

**Mitigation**: In `_run_training_loop()`, track `train_loss` as a rolling average instead of appending to a list. Or cap the list at the last N entries. The `MetricsCallback` can compute the average from the rolling data.

### RISK 8: Test Dependencies on Trainer Instance State

**Severity: MEDIUM**

Multiple tests access `Trainer` instance state after calling methods:

| Test File | State Accessed | Pattern |
|---|---|---|
| `tests/test_training.py` | `trainer.val_loss` | `assert trainer.val_loss < initial_val_loss` |
| `tests/test_training.py` | `trainer.losses` | Checks that losses decrease |
| `tests/test_rnn.py` | `trainer.val_loss`, mock gauges | Verifies gauge-setting via `evaluate()` |

**Impact**: If we change `Trainer.evaluate()` to not mutate instance state (or remove it entirely), these tests will break.

**Mitigation**: In Phase 1-2, preserve `Trainer` instance state mutation via the `TrainerStateCallback` pattern. Only migrate tests in Phase 3 when we have a stable `EpochResult` dataclass.

### RISK 9: Different Data Iteration Patterns

**Severity: MEDIUM**

The three paths iterate over training/validation data differently:

| Path | Training Iteration | Validation Iteration |
|---|---|---|
| Single-node | `for sent, target in train_loader:` | `for sent, target in val_loader:` |
| Distributed | `for _i, batch in train_shard.iter_torch_batches(batch_size=batch_size):` | Same for `val_shard` |
| Tuning | Same as distributed | Same as distributed |

Key differences:
- `DataLoader` yields `(sent_tensor, target_tensor)` tuples
- Ray shards yield `dict` with `"data"` and `"target"` keys, requiring `.long().to(device)` and `.float().to(device)` conversion

**Impact**: `_run_training_loop()` must accept either pattern or use adapter functions.

**Mitigation**: Create a `DataAdapter` protocol that normalizes iteration:
```python
class DataAdapter(Protocol):
    def iter_train(self) -> Iterator[tuple[Tensor, Tensor]]: ...
    def iter_val(self) -> Iterator[tuple[Tensor, Tensor]]: ...
```
- `DataLoaderAdapter` wraps a `DataLoader` (just returns its iterator)
- `RayShardAdapter` wraps Ray shards (converts `dict` → `(Tensor, Tensor)` tuples with device transfer)

### RISK 10: Loss Function Construction Differences

**Severity: LOW**

- `Trainer.fit()` path: `BCEWithLogitsLoss(pos_weight=...)` created in `new_trainer()`, stored as `self.loss_function`
- `_train_func()` path: `BCEWithLogitsLoss(pos_weight=torch.tensor([config.get("pos_weight", 1.0)]).to(device))` created inline
- `_trainable_wrapper()` path: Same as `_train_func()`, but uses `Trainer.loss_function` from `new_trainer()`

**Impact**: The pos_weight tensor is created on different devices in different paths. In `_train_func()`, it's explicitly `.to(device)`. In `new_trainer()`, it's `.to(cfg.device)`.

**Mitigation**: `_create_loss_function(config, device)` factory that always moves pos_weight to the correct device. Used by all three paths.

## Updated Risk Summary

| # | Risk | Severity | Mitigation |
|---|------|----------|-----------|
| 1 | `Trainer.evaluate()` mutates instance state read by external callers | HIGH | `TrainerStateCallback` writes back to `Trainer` object in Phase 1-2 |
| 2 | `tuner.py` calls private `Trainer` methods directly | MEDIUM | Refactor `_trainable_wrapper()` to use `_run_training_loop()` in Phase 3 |
| 3 | Ray-specific patterns can't be in shared code | MEDIUM | Isolate in `RayReportCallback` and `_train_func()` wrapper |
| 4 | Scheduler step ordering | LOW | Already consistent; document in docstring |
| 5 | Early stopping must break epoch loop | MEDIUM | `EarlyStoppingCallback.on_epoch_end()` returns `bool` to break |
| 6 | CUDA cleanup in `finally` block | MEDIUM | `Trainer.fit()` wraps `_run_training_loop()` in `try/finally` with cleanup |
| 7 | Unbounded `self.losses` list | LOW | Pre-existing; fix with rolling average during refactoring |
| 8 | Tests depend on `Trainer` instance state | MEDIUM | Preserve state mutation via callback in Phase 1-2; migrate tests in Phase 3 |
| 9 | Different data iteration patterns (DataLoader vs Ray shards) | MEDIUM | `DataAdapter` protocol normalizes iteration |
| 10 | Loss function device handling | LOW | `_create_loss_function(config, device)` factory |

## Estimated Effort

| Phase | Hours | Description |
|-------|-------|-------------|
| Phase 1 | ~2 | Extract shared functions (`_evaluate`, `TrainingConfig`, model creation) |
| Phase 2 | ~3 | Add `TrainingCallback` protocol + callback implementations |
| Phase 3 | ~2 | Refactor callers to use `_run_training_loop()` |
| Phase 4 | ~1 | Testing & validation |
| **Total** | **~8** | |

## Recommended Implementation Order

Based on risk analysis, the safest implementation order is:

1. **Phase 1a**: Extract `_evaluate()` as a standalone function (low risk, high value — removes 30 lines of duplication)
2. **Phase 1b**: Extract `_create_loss_function()` and `_create_optimizer()`/`_create_scheduler()` (low risk)
3. **Phase 1c**: Unify model creation via `MODEL_REGISTRY` (already partially done in `base.py`)
4. **Phase 2a**: Add `TrainingCallback` protocol + `MetricsCallback` (medium risk — must preserve `Trainer` state mutation)
5. **Phase 2b**: Add `CheckpointCallback`, `EarlyStoppingCallback`, `LoggingCallback`
6. **Phase 2c**: Add `RayReportCallback` (requires Ray import isolation)
7. **Phase 3a**: Create `_run_training_loop()` and refactor `Trainer.fit()` (high risk — must test thoroughly)
8. **Phase 3b**: Refactor `_train_func()` to use `_run_training_loop()` (medium risk)
9. **Phase 3c**: Refactor `_trainable_wrapper()` to use `_run_training_loop()` (medium risk)
10. **Phase 4**: Full test suite + training loop validation

Each sub-phase should be a separate PR with full test coverage before merging.