# Agent Guide

## Project Overview

Sentimentizer is a PyTorch-based sentiment analysis pipeline with three model architectures (RNN/LSTM, Transformer Encoder, Transformer Encoder-Decoder). It supports single-node and distributed training (via Ray Train), Prometheus metrics, and Grafana dashboards.

## Recent Changes

### Stale Metrics Cleanup (current)

**Problem**: When training the encoder model, the dashboard showed unexpected RNN metrics from a previous training run because `/tmp/sentimentizer_training_metrics.json` accumulates entries across runs and `prometheus_client` gauges retain their last-set values.

**Solution**: Added `_reset_stale_metrics(model_type)` in `workflows/stages/train.py` that is called at the start of every `run_train()` invocation. It:
1. Writes a zeroed-out entry for the current `model_type` in `/tmp/sentimentizer_training_metrics.json` (sets all metrics to 0, epoch to 0). This is critical: deleting the entry would cause the standalone exporter to keep serving stale in-process gauge values, since it only updates labels found in the file.
2. Resets all 11 `sentimentizer_training_*` Prometheus gauges for that model type to 0
3. Invalidates the `_RAY_GAUGES` cache entry so a fresh gauge is lazily created

**Tests**: `TestResetStaleMetrics` in `tests/test_training.py` (6 tests covering JSON file cleanup, missing/corrupt files, Prometheus gauge reset, and Ray gauge cache invalidation).

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
  test_training.py  — Training primitives, Trainer.fit, checkpoints, stale metrics
  test_rnn.py        — RNN/Encoder model integration + Ray distributed tests
  test_skill.py      — Agent/tuning config tests
```

## Metrics Pipeline

- **Training driver** writes metrics to `/tmp/sentimentizer_training_metrics.json` (one key per `model_type`)
- **Standalone exporter** (port 8081) reads JSON every 10s and serves `sentimentizer_training_*` gauges
- **Ray workers** (port 8080) emit `ray_sentimentizer_live_*` gauges from rank 0
- **Tune process** (port 8082) emits `sentimentizer_tune_*` gauges per trial
- **Grafana dashboards** use `sentimentizer_training_* or ray_sentimentizer_live_*` PromQL fallback

## Common Tasks

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