# Per-Batch Scheduler Stepping Migration Plan

> **Status: COMPLETE** — All models now use per-batch scheduler stepping with `warmup_ratio=0.06`. `CosineAnnealingLR` has been removed entirely.

## Problem

All models except ModernBERT step their learning rate scheduler **once per epoch** rather than once per optimizer step (per batch). The scheduler's `T_max` and `warmup_epochs` are in epoch units, so with 12 training epochs and `T_max=24`, the cosine curve only gets 12 discrete LR transitions across the entire training run.

This has two major consequences:

1. **Warmup is broken**: With `warmup_epochs=1` and per-epoch stepping, the "warmup" is a single step that starts at full LR (`(0+1)/1 = 1.0`). There is no gradual ramp from near-zero. The optimizer trains at full LR from the very first batch.

2. **LR is effectively flat for the first ~5 epochs**: The cosine schedule with `T_max=24` (designed so training only uses the gentle first half) produces near-constant LR for the first several epochs. At epoch 3, the encoder LR has decreased from 3e-4 to ~2.98e-4 (a 0.7% drop). This explains the observed loss plateau at ~0.75 around epoch 3 — the model is barely learning at a different rate than epoch 1.

Per-batch scheduler stepping is the standard practice across all major training frameworks (HuggingFace Trainer, PyTorch Lightning, fairseq, Megatron-LM). The scheduler should track optimizer steps, not epoch boundaries.

### Numerical comparison (encoder, `lr=3e-4`, 12 epochs, `T_max=24`, `warmup_epochs=1`)

| Step | Phase | LR multiplier | Effective LR | Notes |
|:-----|:------|:--------------|:-------------|:------|
| 0 | warmup | 1.0 | 3e-4 | No ramp — immediate full LR |
| 1 | cosine | 1.0 | 3e-4 | **No change at all** |
| 2 | cosine | ~0.998 | ~2.99e-4 | 0.3% decrease after 2 full epochs |
| 3 | cosine | ~0.993 | ~2.98e-4 | 0.7% decrease after 3 full epochs |
| 4 | cosine | ~0.981 | ~2.94e-4 | 2% decrease after 4 full epochs |
| 12 | cosine | ~0.686 | ~2.06e-4 | 31% decrease after 12 epochs |

For comparison, per-batch stepping with `warmup_ratio=0.06` and `total_steps=9000` (batch_size=8, ~6000 samples, 12 epochs):

| Milestone | LR | Notes |
|:----------|:---|:------|
| Step 0 | ~1e-6 | Start of warmup ramp |
| Step 540 | 3e-4 | End of warmup (6% of total) |
| Step 4500 | ~2.2e-4 | Mid-training, meaningful decay |
| Step 9000 | 1e-6 | End of cosine decay |

ModernBERT already uses per-batch stepping (`STEP_SCHEDULER_PER_BATCH = True`) and this is working correctly.

---

## Current Implementation Status

**All phases are complete.** The per-batch scheduler migration has been fully implemented across all four model types (RNN, Encoder, Decoder, ModernBERT).

**Done:**
- `STEP_SCHEDULER_PER_BATCH: ClassVar[bool] = True` on `BaseSentimentModel` in `sentimentizer/models/base.py` (flipped from `False`)
- `STEP_SCHEDULER_PER_BATCH = True` override removed from `HFTransformerModel` (now redundant, inherits from base)
- `ModernBERTSchedulerParams` replaced `T_max=6` + `warmup_epochs=1` with `warmup_ratio=0.06`
- `ModernBERTConfig.freeze_backbone_epochs` changed from `1` to `0`
- `RNNSchedulerParams` added with `warmup_ratio=0.06`; `warmup_ratio` added to `EncoderSchedulerParams` and `DecoderSchedulerParams`
- `_get_sched_params()` returns `RNNSchedulerParams()` for RNN; return type annotation updated
- `CosineAnnealingLR` removed from `_create_training_components()`, `_train_func()`, and `_rebuild_optimizer_after_unfreeze()` — all models use `_LinearWarmupCosineScheduler`
- Stale `warmup_sched`/`use_warmup`/`warmup_steps`/`total_steps` logic replaced with `warmup_ratio` in `new_ray_trainer()` and `_train_func()`
- `Trainer.fit()` rebuilds the scheduler with real step counts for any model where `STEP_SCHEDULER_PER_BATCH=True` (after `_new_loaders()` creates the DataLoader)
- `_train_epoch()` steps the scheduler per accumulated optimizer step when `STEP_SCHEDULER_PER_BATCH=True`, including the partial-flush path
- `_run_training_loop()` has the same per-batch step gates (used in tests)
- `_train_func()` (Ray distributed) rebuilds the scheduler before the epoch loop for per-batch models, using `train_dataset_count` from `train_loop_config`
- Per-epoch `scheduler.step()` calls guarded with `not getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False)` (always `True` — the guard is a safety net for future model types)
- `train_dataset_count` passed in `train_loop_config` for **all model types** via `train_ds.count()` in `new_ray_trainer()`
- `T_max` and `warmup_epochs` on `SchedulerParams` subclasses are now legacy fields — kept for checkpoint compat but not used by the scheduler
- Documentation updated in AGENTS.md, `docs/modernbert_plan.md`, `docs/troubleshooting.md`

---

## Plan

### Principle

Every model scheduler should step **per optimizer step** (per batch), not per epoch. The `STEP_SCHEDULER_PER_BATCH` flag on `BaseSentimentModel` should default to `True`, and per-epoch `.step()` calls should only exist as a no-op fallback.

### Phase 1: Switch all models to `STEP_SCHEDULER_PER_BATCH = True`

**Files changed**: `sentimentizer/models/base.py`, `sentimentizer/models/hf_base.py`

- Change `BaseSentimentModel.STEP_SCHEDULER_PER_BATCH` from `False` to `True`
- This flips the default for RNN, Encoder, and Decoder (ModernBERT already inherits `True` from `HFTransformerModel`)
- Remove the per-class override from `HFTransformerModel` since the base default is now `True` for everyone

### Phase 2: Add `warmup_ratio` to all scheduler params, switch RNN to warmup+cosine

**Files changed**: `sentimentizer/config.py`

#### `RNNSchedulerParams` (new dataclass)

The RNN currently uses `CosineAnnealingLR` (no warmup). We switch it to `_LinearWarmupCosineScheduler` with per-batch stepping, matching the other models.

Currently `_get_sched_params("rnn")` returns the base `SchedulerParams()`. Proposed:

```python
class RNNSchedulerParams(SchedulerParams):
    """Scheduler params for RNN model.

    Uses per-batch warmup+cosine decay. warmup_ratio controls what fraction
    of total optimizer steps are spent warming up. T_max and warmup_epochs
    are inherited but unused — the per-batch scheduler is rebuilt in
    Trainer.fit() / _train_func() once the DataLoader length is known.
    """
    warmup_ratio: float = 0.06
    eta_min: float = 1e-6
```

#### `EncoderSchedulerParams`

```python
class EncoderSchedulerParams(SchedulerParams):
    # ... existing docstring ...
    T_max: int = 24            # legacy — unused for per-batch stepping
    warmup_epochs: int = 1     # legacy — unused for per-batch stepping
    warmup_ratio: float = 0.06  # NEW: ~6% of total steps for warmup
    eta_min: float = 1e-6
```

#### `DecoderSchedulerParams`

```python
class DecoderSchedulerParams(SchedulerParams):
    # ... existing docstring ...
    T_max: int = 24            # legacy — unused for per-batch stepping
    warmup_epochs: int = 1     # legacy — unused for per-batch stepping
    warmup_ratio: float = 0.06  # NEW: ~6% of total steps for warmup
    eta_min: float = 1e-5
```

#### `ModernBERTSchedulerParams`

Already has `warmup_ratio`. No changes needed.

### Phase 3: Remove `CosineAnnealingLR` from all scheduler creation sites

**Files changed**: `sentimentizer/trainer.py`

There are three sites that still create `CosineAnnealingLR` and need updating:

#### 3a. `_create_training_components()` placeholder

Currently:

```python
is_transformer = isinstance(sched_params, (EncoderSchedulerParams, DecoderSchedulerParams, ModernBERTSchedulerParams))
if is_transformer and sched_params.warmup_epochs > 0:
    scheduler = _LinearWarmupCosineScheduler(...)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)
```

After: remove the `is_transformer` check entirely, all models use `_LinearWarmupCosineScheduler` as the placeholder. The placeholder uses epoch-based rough defaults (`warmup_steps=sched_params.warmup_epochs or 1`, `total_steps=sched_params.T_max`) and gets rebuilt with real step counts in `Trainer.fit()` / `_train_func()` before any training happens.

#### 3b. `_train_func()` initial scheduler creation

Currently (lines ~1637–1650) the RNN path falls into:

```python
if use_warmup and warmup_steps > 0:
    scheduler = _LinearWarmupCosineScheduler(...)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=epochs, ...)
```

For RNN, `use_warmup=False` is passed from `new_ray_trainer()`, so a `CosineAnnealingLR` is created and **immediately thrown away** when the per-batch rebuild fires a few lines later. Fix: always create `_LinearWarmupCosineScheduler` as the placeholder here too, or replace the if/else with a single unconditional `_LinearWarmupCosineScheduler(...)`. The rebuild will overwrite it anyway.

#### 3c. `new_ray_trainer()` stale `warmup_sched` logic

Currently (lines ~2159–2169) `new_ray_trainer()` computes `use_warmup` / `warmup_steps` / `total_steps` using an isinstance check for transformer sched params and `sched.warmup_epochs > 0`. These values are passed to `train_loop_config` and used only for the initial placeholder scheduler in `_train_func()` — they're overwritten before the epoch loop. With RNN gaining `RNNSchedulerParams` (no `warmup_epochs`), the isinstance check would exclude it.

Fix: drop the `warmup_sched`/`use_warmup` logic. Instead pass `warmup_ratio = getattr(sched, "warmup_ratio", 0.06)` and let `_train_func()` always create `_LinearWarmupCosineScheduler` as the placeholder.

#### 3d. `_rebuild_optimizer_after_unfreeze()` dead branch

Currently (lines ~890–908):

```python
if isinstance(scheduler, _LinearWarmupCosineScheduler):
    new_scheduler = _LinearWarmupCosineScheduler(...)
else:
    new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)  # dead after this change
```

Once all models use `_LinearWarmupCosineScheduler`, the `else` branch is unreachable. Remove it — the dead `CosineAnnealingLR` construction is confusing and references `.T_max` which `_LinearWarmupCosineScheduler` doesn't have.

### Phase 4: Update `_get_sched_params()` for RNN

**Files changed**: `sentimentizer/trainer.py`

Currently:
```python
def _get_sched_params(model_type):
    if model_type == "encoder": return EncoderSchedulerParams()
    if model_type == "decoder": return DecoderSchedulerParams()
    if model_type == "rnn": return SchedulerParams()
    if model_type == "modernbert": return ModernBERTSchedulerParams()
```

After:
- `rnn` returns `RNNSchedulerParams()` instead of `SchedulerParams()`
- Return type annotation updated to include `RNNSchedulerParams`

### Phase 5: Verify the rebuild logic applies to all models

The scheduler rebuild blocks already exist and work for any model where `STEP_SCHEDULER_PER_BATCH=True`:

1. **`Trainer.fit()`** (~line 1065): Rebuilds after `_new_loaders()` using `len(train_loader)`. Works for all models.

2. **`_train_func()`** (~line 1699): Rebuilds before the epoch loop using `train_dataset_count`. Works for all models. `train_dataset_count` is **already passed** in `train_loop_config` for all model types via `train_ds.count()` in `new_ray_trainer()` — no change needed here.

Both blocks use `warmup_ratio = getattr(sched_params, "warmup_ratio", 0.06)`. Once all `SchedulerParams` subclasses have `warmup_ratio`, the `getattr` fallback is a safety net only.

### Phase 6: Update the per-epoch `.step()` call sites

These already have the correct guard:

```python
if self.scheduler and not getattr(type(model), "STEP_SCHEDULER_PER_BATCH", False):
    self.scheduler.step()
```

Once `STEP_SCHEDULER_PER_BATCH` is `True` for all models, this branch is **always skipped**. No code changes needed. The guard can be removed in a follow-up cleanup PR once all models are confirmed retrained and working.

---

## Risks

### 1. Existing checkpoints are incompatible with the new schedule

**Severity: HIGH**

A scheduler saved with per-epoch stepping has `last_epoch=5` (meaning "after 5 epochs"). A per-batch scheduler at the same point in training would have `last_epoch=3750` (meaning "after 3750 optimizer steps"). Resuming from an old checkpoint would reset the scheduler to a completely wrong point in the LR curve.

**Mitigation**: This is a breaking change. User has confirmed retraining all models is acceptable, so no backwards-compatibility shim is needed. Document in AGENTS.md that checkpoints from before this change cannot be resumed. Add a version marker (`"scheduler_mode": "per_batch"`) to new checkpoint metadata so future resume logic can detect and reject old-format checkpoints with a clear error.

### 2. Different effective LR schedule — training dynamics change

**Severity: MEDIUM**

Per-batch stepping produces a fundamentally different LR trajectory than per-epoch stepping, even with the same hyperparameters:

- **Warmup**: Per-batch produces a proper gradual ramp from `eta_min` to `base_lr` over hundreds of steps. Per-epoch produces a single-step jump to full LR.
- **Decay**: Per-batch decays smoothly across thousands of steps. Per-epoch makes 12 discrete jumps.

This means:
- Loss curves will look different (no more plateau at epoch 3)
- Optimal hyperparameters may shift — previously-tuned `lr`, `weight_decay`, `eta_min` may need adjustment
- Model quality metrics (macro F1, balanced accuracy) may change — could be better or worse
- The `T_max=2*epochs` invariant from AGENTS.md is no longer applicable — with per-batch stepping, `total_steps` is computed dynamically from dataset size

**Mitigation**: Retrain all three GloVe models and compare metrics. Expect a different but likely better loss curve.

### 3. Dataset-size dependency

**Severity: MEDIUM**

With per-epoch stepping, `T_max=24` always produces the same LR schedule regardless of training set size. With per-batch stepping, `total_steps = epochs * steps_per_epoch`, which depends on the training set size. A training set of 1000 samples with `batch_size=8` has 125 steps/epoch, while 100,000 samples has 12,500 steps/epoch — the cosine curve spans 10× more optimizer steps.

**Mitigation**: The `warmup_ratio` approach (6% of total steps for warmup) is dataset-size-agnostic and standard practice. The total number of optimizer steps naturally scales with dataset size. This is the correct behavior — larger datasets need more steps and smoother schedules.

### 4. RNN warmup change

**Severity: LOW**

The RNN currently has `warmup_epochs=0` (no warmup). Switching to `warmup_ratio=0.06` means the RNN will now have a warmup phase for the first time. This changes the RNN's training dynamics — the LR ramps from `eta_min` to `base_lr` over ~6% of training instead of starting at full LR.

**Mitigation**: This is an improvement. Warmup is universally recommended for transformer models and is standard for RNN/LSTM fine-tuning as well. The LR at step 1 will be near-zero instead of full, which is gentler on the model. If RNN metrics regress, we can reduce `warmup_ratio` to 0 or add a per-model override.

### 5. Gradient accumulation interaction

**Severity: LOW (already handled)**

Per-batch scheduler stepping should fire once per accumulated optimizer step, not once per mini-batch. The existing code correctly gates the step with `(accum_step_idx % grad_accum_steps) == 0`. No changes needed.

### 6. `_rebuild_optimizer_after_unfreeze` dead branch and fast-forward correctness

**Severity: LOW**

`_rebuild_optimizer_after_unfreeze()` has two separate concerns after this change:

1. **Dead `else` branch**: The `else` block that creates `CosineAnnealingLR` (triggered when the current scheduler is not `_LinearWarmupCosineScheduler`) becomes unreachable once all models use `_LinearWarmupCosineScheduler`. Remove it — the dead code references `.T_max` which doesn't exist on `_LinearWarmupCosineScheduler` and would raise `AttributeError` if ever reached.

2. **Fast-forward correctness**: The rebuild fast-forwards the new scheduler by `old_last_epoch` steps. With per-epoch stepping, `old_last_epoch` was 1 after epoch 1. With per-batch stepping, `old_last_epoch` will be ~750 after epoch 1 (however many optimizer steps happened). The rebuild code reads `scheduler.last_epoch` from the existing `_LinearWarmupCosineScheduler` instance, so this works correctly — it fast-forwards the right number of steps regardless of unit.

**Mitigation**: Remove the dead `else` branch as part of Phase 3d.

### 7. Distributed training per-worker step counts may vary

**Severity: LOW**

In Ray distributed training, different workers may process slightly different numbers of batches per epoch if the dataset isn't evenly divisible. This means different workers could be at slightly different scheduler steps after each epoch. However, since the learning rate differences are tiny and the existing code already handles this for ModernBERT without issues, this is not a practical concern.

### 8. `default_epochs()` invariant with `T_max`

**Severity: LOW (documentation only)**

AGENTS.md states: `T_max >= epochs_trained`. With per-batch stepping, `T_max` in the config is unused — the actual `total_steps` is computed dynamically as `epochs * steps_per_epoch`. The invariant becomes `total_steps >= epochs_trained * steps_per_epoch`, which is trivially true. The AGENTS.md entry about `T_max >= default_epochs` should be updated to reflect that this now applies to the dynamically-computed `total_steps` rather than the config `T_max`.

### 9. Ray `_train_func` dataset count estimation

**Severity: LOW (already wired; estimation is approximate)**

`train_dataset_count` is **already passed for all model types** in `new_ray_trainer()` via `train_ds.count()` — this is not a gap. The estimation itself can be slightly off:

```python
_per_worker_rows = math.ceil(_train_dataset_count / max(1, _num_workers))
_steps_per_epoch = max(1, math.ceil(_per_worker_rows / batch_size) // _grad_accum)
_total_steps = epochs * _steps_per_epoch
```

- The dataset may not be evenly divisible across workers
- Gradient accumulation can make the effective step count slightly different
- The per-worker shard may have a different row count than estimated

An off-by-10% estimate means the cosine curve ends 10% early or late — the LR will either undershoot or overshoot `eta_min` slightly. This is acceptable (the LR will still decay substantially). Document in AGENTS.md that Ray step counts are approximate.

---

## Summary of Changes

| File | Change | Status |
|:-----|:-------|:-------|
| `sentimentizer/models/base.py` | `STEP_SCHEDULER_PER_BATCH: ClassVar[bool] = True` (was `False`) | pending |
| `sentimentizer/models/hf_base.py` | Remove `STEP_SCHEDULER_PER_BATCH = True` override (base now defaults `True`) | done |
| `sentimentizer/models/encoder.py` | No change needed (inherits from `BaseSentimentModel`) | — |
| `sentimentizer/models/decoder.py` | No change needed (inherits from `BaseSentimentModel`) | — |
| `sentimentizer/models/rnn.py` | No change needed (inherits from `BaseSentimentModel`) | — |
| `sentimentizer/config.py` | Add `RNNSchedulerParams` with `warmup_ratio=0.06`; add `warmup_ratio` to `EncoderSchedulerParams` and `DecoderSchedulerParams` | done |
| `sentimentizer/trainer.py` — `_get_sched_params()` | Return `RNNSchedulerParams()` for RNN; update return type annotation | done |
| `sentimentizer/trainer.py` — `_create_training_components()` | Remove `is_transformer` check; all models use `_LinearWarmupCosineScheduler` placeholder (Phase 3a) | done |
| `sentimentizer/trainer.py` — `_train_func()` initial scheduler | Replace `CosineAnnealingLR` else-branch with `_LinearWarmupCosineScheduler` for all models (Phase 3b) | done |
| `sentimentizer/trainer.py` — `new_ray_trainer()` | Drop stale `warmup_sched`/`use_warmup` isinstance logic; pass `warmup_ratio` instead (Phase 3c) | done |
| `sentimentizer/trainer.py` — `_rebuild_optimizer_after_unfreeze()` | Remove dead `CosineAnnealingLR` else-branch (Phase 3d) | done |
| `sentimentizer/trainer.py` — per-batch/epoch step gates | Already in place for all three training paths | done |
| `sentimentizer/trainer.py` — scheduler rebuild in `Trainer.fit()` + `_train_func()` | Already in place; fires for any model with `STEP_SCHEDULER_PER_BATCH=True` | done |
| `sentimentizer/config.py` — `ModernBERTSchedulerParams` | `warmup_ratio=0.06` already replaces `T_max`/`warmup_epochs` | done |
| `sentimentizer/config.py` — `ModernBERTConfig.freeze_backbone_epochs` | Already set to `0` | done |
| `AGENTS.md` | Update scheduler invariants; note per-batch stepping; update `T_max` section; document checkpoint incompatibility; note Ray step count is approximate | done |
| `docs/modernbert_plan.md` | Update LR annealing section to reflect per-batch stepping is live | done |

## Implementation Order

1. ~~**Trainer infrastructure**~~ — per-batch step gates, scheduler rebuild blocks, `STEP_SCHEDULER_PER_BATCH` flag on base/HF classes, ModernBERT config updates ✓ **done**
2. ~~**Config changes**~~ — `RNNSchedulerParams`, `warmup_ratio` on Encoder/Decoder sched params ✓ **done**
3. ~~**Base class flag flip**~~ — `STEP_SCHEDULER_PER_BATCH = True` on `BaseSentimentModel`, remove override on `HFTransformerModel` ✓ **done**
4. ~~**Trainer cleanup**~~ — Remove `CosineAnnealingLR` branches (Phases 3a–3d), update `_get_sched_params()` ✓ **done**
5. **Retrain** — All three GloVe models need retraining; compare metrics
6. ~~**Documentation**~~ — Update AGENTS.md, `docs/modernbert_plan.md` ✓ **done**

## Post-Implementation Validation

- Train all four models (RNN, Encoder, Decoder, ModernBERT) and verify:
  - Loss curves show proper warmup ramp (LR starts low, ramps up over ~6% of training)
  - Loss curves show smooth cosine decay (no plateau at epoch 3)
  - Per-batch LR values logged in progress bar match expected schedule
  - Final metrics (macro F1, balanced accuracy) are comparable or better than before
  - Resume-from-checkpoint works correctly (new-format scheduler state)
  - Distributed (Ray) training produces the same LR schedule as single-node
