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
| Model setup | Created in `new_trainer()` | `new_model()` + `prepare_model()` | `new_model()` + `prepare_model()` |
| Checkpointing | Periodic + best model saves | `train.report(checkpoint=...)` | `train.report(checkpoint=...)` |
| Metrics reporting | `publish_epoch_metrics()` | `publish_epoch_metrics()` from rank 0 | `publish_epoch_metrics()` from rank 0 |
| Loss function | `BCEWithLogitsLoss(pos_weight=...)` | Same, built inline | Same, built inline |
| Scheduler | Created in `new_trainer()` | Created inline | Created inline |
| Early stopping | Yes (patience counter) | No | No |
| Device management | `.to(device)` in `fit()` | `next(model.parameters()).device` | `next(model.parameters()).device` |

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

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Ray-specific logic (DDP, `prepare_model`, `train.report`) leaking into shared code | Callback pattern isolates Ray-specific code into `RayReportCallback` |
| Different data iteration patterns (DataLoader vs shard) | `_run_training_loop()` takes any iterable; adapter functions handle conversion |
| Regression in distributed training | Existing `test_rnn.py::TestRayDistributed` tests cover this path |
| Large PR size | Split into 3 phases as described above |

## Estimated Effort

| Phase | Hours | Description |
|-------|-------|-------------|
| Phase 1 | ~2 | Extract shared functions (`_evaluate`, `TrainingConfig`, model creation) |
| Phase 2 | ~3 | Add `TrainingCallback` protocol + callback implementations |
| Phase 3 | ~2 | Refactor callers to use `_run_training_loop()` |
| Phase 4 | ~1 | Testing & validation |
| **Total** | **~8** | |