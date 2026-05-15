# Agent Guide

## Project Overview

Sentimentizer is a PyTorch-based sentiment analysis pipeline with three model architectures (RNN/LSTM, Transformer Encoder, Transformer Encoder-Decoder). It supports single-node and distributed training (via Ray Train), Prometheus metrics, and Grafana dashboards.

## Recent Changes

### torchmetrics Migration (current)

**Change**: Replaced custom `_cohen_kappa()`, `_auc_roc()`, and `_auc_roc_manual()` implementations with `torchmetrics` library (`BinaryPrecision`, `BinaryRecall`, `BinaryF1Score`, `BinaryCohenKappa`, `BinaryAUROC`, `BinaryMatthewsCorrCoef`, `BinaryNegativePredictiveValue`, `BinaryAveragePrecision`). This eliminates ~200 lines of hand-rolled metric math and the `sklearn.metrics.roc_auc_score` conditional import.

**New metrics added**:
- **MCC** (Matthews Correlation Coefficient): Range -1 to 1, robust to class imbalance. Best single-number summary of confusion matrix quality.
- **NPV** (Negative Predictive Value): TN / (TN + FN). Complement to precision for the negative class.
- **Average Precision** (PR-AUC): Area under the Precision-Recall curve. More informative than AUC-ROC for imbalanced datasets.
- **Macro F1**: Mean of per-class F1 scores. Weights both classes equally — drops significantly when the model ignores the negative class, unlike positive-class-only F1.

**Edge-case handling**:
- **NaN probabilities**: `torchmetrics.BinaryAUROC` silently gives wrong results with NaN input. `_replace_nan_probs()` replaces NaN→0.5 *before* calling torchmetrics, preserving the existing warning log.
- **Single-class Cohen's kappa**: `torchmetrics.BinaryCohenKappa` returns `nan` for single-class targets. `_safe_item()` coerces `nan→0.0` to avoid Prometheus gauge issues. **Behavioral change**: single-class kappa is now `0.0` (was `1.0` in the custom implementation).
- **Empty arrays**: Guard clause returns `ClassificationMetrics(total=0)` before calling torchmetrics (which would crash on empty input).

**Key files changed**:
- `sentimentizer/metrics.py` — rewrote with torchmetrics `Binary*` classes; added `_to_long_tensor`, `_to_float_tensor`, `_replace_nan_probs`, `_safe_item` helpers; added `mcc`, `npv`, `avg_precision` fields to `ClassificationMetrics`
- `sentimentizer/trainer.py` — added MCC/NPV/avg_precision gauges, JSON persistence, and logger fields
- `sentimentizer/exporter.py` — added `TRAINING_VAL_MCC`, `TRAINING_VAL_NPV`, `TRAINING_VAL_AVG_PRECISION` gauges and JSON reading
- `workflows/stages/train.py` — updated `_reset_stale_metrics()` and `_persist_metrics_to_file()` for new metrics
- `tests/test_metrics.py` — updated for torchmetrics; added `TestHelperFunctions` class; updated `TestCohenKappa.test_single_class_returns_zero`
- `tests/test_rnn.py` — updated mock_gauges dicts with new keys
- `pyproject.toml` — added `torchmetrics>=1.9.0` dependency
- `docs/metrics.md` — updated NaN handling section; added new metric names

### Dictionary Tokenization Bug Fix

**Problem**: The dictionary stored tokens with wrapping quotes (e.g., `"'the'"` instead of `"the"`) because `str()` was called on numpy arrays from parquet columns instead of converting them to Python lists via `list()`. This caused a 99.9% GloVe vocabulary mismatch — nearly every word was mapped to the OOV embedding index (row `len(dictionary) + 1`), so the model trained on random embeddings and collapsed to predicting the majority class (zero negative accuracy, zero Cohen's kappa).

**Root cause**: `isinstance(doc_tokens, list)` returns `False` for numpy arrays. The fallback `regex_tokenize(str(numpy_array))` stringified the array representation `"['if' 'you' ...]"` and the regex `[a-z0-9'-]+` captured the wrapping quotes as part of each token.

**Fix**: Changed the fallback in `new_dictionary()` and `_count_vocab_batch()` to `list(doc_tokens)` with a `TypeError` catch for genuinely non-iterable types. Same fix in `workflows/stages/tokenize.py` for the resume path.

**Related fix — scheduler**: `EncoderSchedulerParams.T_max` was `4` but `default_epochs("encoder")` is `8`, causing the LR to decay to minimum after half the epochs. Changed `T_max` to `8`. Also fixed `_LinearWarmupCosineScheduler` which returned `0.0` at step 0 (zero LR for the entire first epoch) — changed from `step / warmup_steps` to `(step + 1) / warmup_steps`.

**Tests**: `TestDictionaryNumpyArrays` in `tests/test_loader.py`, `TestSchedulerCorrectness` in `tests/test_training.py`, `TestCountVocabBatch.test_numpy_array_tokens` and `TestCountVocabBatch.test_pandas_series_tokens` in `tests/test_dictionary_lifecycle.py`.

**Key files changed**:
- `sentimentizer/tokenizer.py` — numpy array handling in `new_dictionary()` and `_count_vocab_batch()`
- `workflows/stages/tokenize.py` — numpy array handling in resume path
- `sentimentizer/config.py` — `EncoderSchedulerParams.T_max` changed from 4 to 8
- `sentimentizer/trainer.py` — `_LinearWarmupCosineScheduler` warmup formula fix

### Stale Metrics Cleanup

**Problem**: When training the encoder model, the dashboard showed unexpected RNN metrics from a previous training run because a single shared JSON file accumulated entries across runs and `prometheus_client` gauges retained their last-set values. Concurrent training processes could also race on the shared file, overwriting each other.

**Solution**: Switched from a single shared JSON file (`/tmp/sentimentizer_training_metrics.json`) to per-model-type JSON files (`/tmp/sentimentizer_metrics/{model_type}_metrics.json`). Each model type writes to its own file, so concurrent training processes never race. The standalone exporter discovers all three files and zeroes gauges for any model type whose file is missing or stale. Added `_written_by` and `_written_at` trace fields to every write, and `_trace.reset_by` / `_trace.reset_at` to reset files, so debugging future issues is trivial: just `cat` the file to see which model_type wrote it and when.

**Important**: `_reset_stale_metrics(model_type)` zeroes **all three** model types (rnn, encoder, decoder), not just the current one, because starting a new training run makes all previous metrics stale — regardless of which model produced them. Otherwise old RNN metrics linger on the dashboard when training encoder.

**Tests**: `TestResetStaleMetrics` in `tests/test_training.py` (6 tests covering JSON file cleanup for all model types, stale cross-model-type data overwrite, missing/corrupt files, Prometheus gauge reset for all model types, and Ray gauge cache invalidation).

**Key files changed**:
- `workflows/stages/train.py` — added `_reset_stale_metrics()` function and call in `run_train()`
- `tests/test_training.py` — added `TestResetStaleMetrics` test class
- `docs/metrics.md` — added "Stale Metrics Reset" section
- `CLAUDE.md` — added stale metrics convention

## Architecture Quick Reference

```
sentimentizer/
  config.py        — Dataclass configs (DriverConfig, RNNConfig, EncoderConfig, etc.)
  trainer.py        — Trainer class, new_trainer(), _train_func(), Ray gauge logic
  exporter.py       — Standalone Prometheus exporter (port 8081), all gauge definitions
  tuner.py          — Ray Tune integration with TunePrometheusCallback
  metrics.py        — ClassificationMetrics dataclass + compute functions
  models/
    rnn.py          — Bidirectional LSTM
    encoder.py      — Transformer encoder with CLS token
    decoder.py      — Encoder-decoder transformer

workflows/
  driver.py         — CLI entry point
  stages/
    train.py        — run_train(), _run_fit_single(), _run_fit_distributed(), _reset_stale_metrics()
    tune.py          — Ray Tune orchestration
  lifecycle.py       — State, logger, Ray init/cleanup
  helpers.py         — Model loading, config utilities

tests/
  test_training.py          — Training primitives, Trainer.fit, checkpoints, scheduler correctness, stale metrics
  test_loader.py            — DataLoader, compute_pos_weight, dictionary numpy array handling
  test_dictionary_lifecycle.py — Dictionary save/load, _count_vocab_batch with numpy arrays
  test_rnn.py                — RNN/Encoder model integration + Ray distributed tests
  test_skill.py      — Agent/tuning config tests
```

## Metrics Pipeline

- **Training driver** writes metrics to `/tmp/sentimentizer_metrics/{model_type}_metrics.json` (one file per `model_type`)
- **Standalone exporter** (port 8081) reads JSON every 10s and serves `sentimentizer_training_*` gauges
- **Ray workers** (port 8080) emit `ray_sentimentizer_live_*` gauges from rank 0
- **Tune process** (port 8082) emits `sentimentizer_tune_*` gauges per trial
- **Grafana dashboards** use `sentimentizer_training_* or ray_sentimentizer_live_*` PromQL fallback

## Common Tasks

### Troubleshooting

See [`docs/troubleshooting.md`](docs/troubleshooting.md) for common issues and fixes, including:
- Zero negative-class accuracy / Cohen's kappa = 0
- Dictionary tokens with wrapping quotes
- GloVe match rate below 50%
- Scheduler T_max and warmup issues
- Class imbalance

### Running tests

```bash
uv run pytest tests/ -v --exitfirst --failed-first
# For Ray-specific tests:
uv run pytest tests/ -v -k "Ray"
```

### Linting and formatting

```bash
uv run ruff check .
uv run ruff check --fix .
uv run black --check .
uv run black .
```

### Regenerating dashboards

After modifying `scripts/generate_ray_dashboards.py` (or any code that affects dashboard JSON output), regenerate and reload Grafana:

```bash
make start-metrics    # Regenerates JSON + restarts Grafana + starts exporter
```

Grafana only reads provisioned dashboard files on **startup**, so a restart is required after any dashboard changes.

### Adding a new model type

1. Add config dataclass in `sentimentizer/config.py`
2. Add model class in `sentimentizer/models/`
3. Add optimization/scheduler params in `sentimentizer/config.py` (`_get_opt_params`, `_get_sched_params`)
4. Add model factory import in `sentimentizer/trainer.py` (`_train_func`, `new_trainer`)
5. Add model import in `workflows/helpers.py` (`_load_model`, `_get_model_config`)

### Adding a new training metric

1. Add gauge definition in `sentimentizer/exporter.py` (with `model_type` label)
2. Add gauge in `_get_ray_gauges()` in `sentimentizer/trainer.py`
3. Set gauge in `Trainer.evaluate()`, `_train_func()`, and `_publish_distributed_metrics()`
4. Add to `_reset_stale_metrics()` in `workflows/stages/train.py` (reset to 0)
5. Add to `_persist_metrics_to_file()` and `_update_training_metrics()` in exporter
6. Add to dashboard in `scripts/generate_ray_dashboards.py`

## Important Conventions

- `_reset_stale_metrics(model_type)` is called at training start to prevent stale cross-model-type metrics
- Ray 2.55.1 API: use `Checkpoint.from_directory()`, `train.get_dataset_shard()` (not `get_context().get_dataset_shard()`)
- `prometheus_client` gauges must NOT be created at module import time for Ray workers — use lazy init via `_get_ray_gauges()`
- `ray.init(ignore_reinit_error=True)` in tests; `ray.shutdown()` in cleanup
- All function signatures need type hints
- When iterating over DataFrame or batch columns containing token lists, use `list(doc_tokens)` with a `TypeError` catch — never `str(doc_tokens)`. Numpy arrays from parquet are iterable but not `isinstance(x, list)`, and `str()` produces array representations with wrapping quotes
- Scheduler `T_max` must match `default_epochs()` for the model type — otherwise LR decays to minimum before training finishes
- _LinearWarmupCosineScheduler warmup must use `(step + 1) / warmup_steps` to avoid zero LR at step 0
- PyTorch CUDA torch is installed by default via `make setup` (resolves from PyPI). For CI environments without GPU, use `make setup-ci` to install the CPU-only variant from the PyTorch wheel index. `resolve_device("auto")` warns when torch is CPU-only (`+cpu` suffix) but NVIDIA libraries are installed — this indicates a misconfigured environment where `torch.cuda.is_available()` returns `False` despite hardware being present.
- `resolve_device("auto")` warns when torch is CPU-only (`+cpu` suffix) but NVIDIA libraries are installed — this indicates a misconfigured environment where `torch.cuda.is_available()` returns `False` despite hardware being present.
- `_load_model()` is only called in the single-node path. Distributed training (`_run_fit_distributed`) does NOT load the model in the driver process — Ray workers create their own model via `_train_func`. Loading the model in the driver would waste GPU memory for `run_type="update"`.
- `torchmetrics.BinaryAUROC` silently gives wrong results with NaN input — always call `_replace_nan_probs()` before passing probabilities to it
- `torchmetrics.BinaryCohenKappa` returns `nan` for single-class targets — always wrap with `_safe_item()` to coerce `nan→0.0` for Prometheus gauge compatibility
- Single-class Cohen's kappa is `0.0` (not `1.0`) — this is a behavioral change from the previous custom implementation
- Empty arrays must be guarded before calling torchmetrics (it crashes on empty input) — use the `if total == 0` early return in `compute_classification_metrics()`
