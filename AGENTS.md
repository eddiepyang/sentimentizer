# Agent Guide

## Project Overview

Sentimentizer is a PyTorch-based sentiment analysis pipeline with three model architectures (RNN/LSTM, Transformer Encoder, Transformer Encoder-Decoder). It supports single-node and distributed training (via Ray Train), Prometheus metrics, and Grafana dashboards.

## Recent Changes

### 3-Class Classification Migration (current)

**Change**: Migrated from binary classification (negative/positive, BCEWithLogitsLoss) to 3-class classification (negative/neutral/positive, CrossEntropyLoss). All models now output logits of shape `(B, 3)` instead of `(B, 1)`. `predict_text()` returns `dict[str, float]` instead of `float`. Per-class metrics replace single-class metrics throughout the pipeline.

**Key changes**:
- **Models**: All three models (RNN, Encoder, Decoder) use `num_classes=3` in classifier head, output `(B, 3)` logits, and use `softmax` instead of `sigmoid`
- **Loss**: `CrossEntropyLoss` with `class_weights` and `label_smoothing=0.1` (or `FocalCrossEntropyLoss` with `loss_type="focal"`)
- **Config**: `LABEL_NAMES = ["negative", "neutral", "positive"]`, `NUM_CLASSES = 3`, `include_neutral=True`, `TrainerConfig` has `class_weights`, `loss_type`, `focal_gamma`, `label_smoothing`, `weight_smoothing`, `neutral_oversample_ratio`
- **Tokenizer**: `convert_rating()` maps 1-2★→0, 3★→1, 4-5★→2; `include_neutral=True` keeps 3-star reviews
- **Metrics**: `ClassificationMetrics` has per-class fields (e.g., `neutral_precision`, `negative_recall`), `balanced_accuracy`, `macro_f1`, `weighted_f1`, `confusion_matrix` (3×3); old binary fields (`tp`, `tn`, `fp`, `fn`, `precision`, `recall`, `f1`, `npv`, `positive_accuracy`, `negative_accuracy`, `auc_roc`, `avg_precision`) removed
- **Prometheus gauges**: Per-class names (e.g., `TRAINING_VAL_NEUTRAL_F1`) replace binary names (e.g., `TRAINING_VAL_F1`)
- **Serving**: `_predict_sentiment()` returns `{"model": ..., "label": ..., "scores": {...}}` instead of `{"sentiment_score": float, "label": ...}`
- **Training pipeline**: `compute_class_weights()` replaces `compute_pos_weight()`; `_balance_dataframe()` / `_balance_ray_dataset()` handle multi-class targets
- **ONNX**: Metadata JSON includes `num_classes` and `label_names`

**Key files changed**: `config.py`, `tokenizer.py`, `loader.py`, `metrics.py`, `metrics_publisher.py`, `exporter.py`, `trainer.py`, `models/{base,rnn,encoder,decoder}.py`, `losses.py`, `serve.py`, `tuner.py`, `agent/skill.py`, `hf.py`, `export_onnx.py`, `workflows/{cli,stages/train,stages/tune}.py`, `scripts/generate_ray_dashboards.py`

### torchmetrics Migration

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
- `sentimentizer/metrics.py` — rewrote with torchmetrics `Multiclass*` classes; added `_to_long_tensor`, `_to_float_tensor`, `_replace_nan_probs`, `_safe_item` helpers; added per-class precision/recall/F1, `balanced_accuracy`, `macro_f1`, `weighted_f1`, `mcc`, per-class `auc_roc`/`avg_precision`, confusion matrix 3×3
- `sentimentizer/trainer.py` — added per-class gauges, JSON persistence, and logger fields
- `sentimentizer/exporter.py` — added per-class `TRAINING_VAL_*` gauges (negative/neutral/positive precision, recall, F1), `TRAINING_VAL_BALANCED_ACCURACY`, `TRAINING_VAL_MACRO_F1`, `TRAINING_VAL_WEIGHTED_F1`, `TRAINING_VAL_MCC`, `TRAINING_VAL_NEUTRAL_AUC_ROC`, `TRAINING_VAL_NEUTRAL_AVG_PRECISION`, neutral diagnostic gauges
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

**Important**: `_reset_stale_metrics(model_type)` writes zeroed-out JSON files for **all three** model types (rnn, encoder, decoder) so the standalone exporter can clear stale values, but only resets **Prometheus gauges for the current model type**. Other model types' gauges retain their last real values — the exporter skips `_reset: true` files so untrained models don't show `epoch=0` on the dashboard. Each zeroed-out file includes `_reset: true` and `_trace.reset_by` to distinguish stale resets from real training data.

**Tests**: `TestResetStaleMetrics` in `tests/test_training.py` (6 tests covering JSON file cleanup for all model types, stale cross-model-type data overwrite, missing/corrupt files, Prometheus gauge reset for current model type only, and Ray gauge cache invalidation).

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
  export_onnx.py    — Unified ONNX export, quantization, validation (_RNNOnnxWrapper)
  models/
    base.py          — BaseSentimentModel with predict() and predict_text()
    rnn.py          — Bidirectional LSTM (with onnx_export flag)
    encoder.py      — Transformer encoder with CLS token
    decoder.py      — Encoder-decoder transformer
  router/           — SetFit router module (avoids shadowing setfit library)
    config.py        — SetFitConfig, RouteLabels, AugmentConfig dataclasses
    seeds.py         — Golden example utterances per category (10 per category)
    augment.py       — GLM 5.1 augmentation via Ollama API
    dataset.py       — JSONL dataset loader, train/test split
    train_router.py  — SetFit training script with compat shims
    evaluate.py      — Validation: similarity heatmap, threshold calibration

workflows/
  driver.py         — CLI entry point
  stages/
    train.py        — run_train(), _run_fit_single(), _run_fit_distributed(), _reset_stale_metrics()
    tune.py          — Ray Tune orchestration
    export.py         — ONNX export workflow stage
  lifecycle.py       — State, logger, Ray init/cleanup
  helpers.py         — Model loading, config utilities

tests/
  test_training.py          — Training primitives, Trainer.fit, checkpoints, scheduler correctness, stale metrics
  test_loader.py            — DataLoader, compute_pos_weight, dictionary numpy array handling
  test_dictionary_lifecycle.py — Dictionary save/load, _count_vocab_batch with numpy arrays
  test_rnn.py                — RNN/Encoder model integration + Ray distributed tests
  test_skill.py      — Agent/tuning config tests
  test_export_onnx.py — ONNX export, quantization, validation, _RNNOnnxWrapper
  test_router.py      — SetFit config, labels, seeds, dataset, augmentation
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
3. Add to `_METRIC_GAUGE_KEYS` in `sentimentizer/metrics_publisher.py`
4. Set gauge in `Trainer.evaluate()`, `_train_func()`, and via `publish_epoch_metrics()`
5. Add to `_reset_stale_metrics()` in `workflows/stages/train.py` (reset to 0)
6. Add to `_persist_metrics_to_file()` and `_update_training_metrics()` in exporter
7. Add to dashboard in `scripts/generate_ray_dashboards.py`

## Important Conventions

- **3-class classification**: Models output logits of shape `(B, 3)` with label mapping: 0=negative, 1=neutral, 2=positive. `LABEL_NAMES = ["negative", "neutral", "positive"]` is the single source of truth in `config.py` — import it, don't duplicate.
- **Loss function**: `CrossEntropyLoss` (not `BCEWithLogitsLoss`). Target dtype is `torch.long` (not `torch.float32`). `FocalCrossEntropyLoss` in `sentimentizer/losses.py` for hard-example mining.
- **`predict_text()` returns `dict[str, float]`**: e.g., `{"negative": 0.05, "neutral": 0.12, "positive": 0.83}`. Never call `.item()` on it — use `max(scores, key=scores.get)` for the predicted label.
- **`forward()` output shape**: Always `(B, num_classes)`, never squeeze. Batch-of-1 returns `(1, 3)`, not `(3,)`.
- **`predict()` output**: `torch.softmax(logits, dim=-1)` returns `(B, num_classes)` probability matrix.
- **`compute_class_weights()`** replaces `compute_pos_weight()`. The old function raises `ValueError` if `num_classes > 2`.
- **`_replace_nan_probs()`**: Accepts `(N, num_classes)` shape. NaN values replaced with `1/num_classes`. Rows with partial NaN are re-normalized to sum to 1.0.
- **`ClassificationMetrics`**: 3-class dataclass with per-class fields (`negative_precision`, `neutral_recall`, etc.), `balanced_accuracy`, `macro_f1`, `weighted_f1`, `confusion_matrix` (3×3), `neutral_to_positive_rate`, `neutral_to_negative_rate`, `pred_neutral_frac`. Old binary fields (`tp`, `tn`, `fp`, `fn`, `precision`, `recall`, `f1`, `npv`, `positive_accuracy`, `negative_accuracy`, `auc_roc`, `avg_precision`) no longer exist.
- **Prometheus gauges**: Per-class names (e.g., `TRAINING_VAL_NEUTRAL_F1`, `TRAINING_VAL_NEGATIVE_PRECISION`), not binary names. Ray gauge dict keys match `_METRIC_GAUGE_KEYS` in `metrics_publisher.py`.
- **`_reset_stale_metrics(model_type)`** is called at training start to prevent stale cross-model-type metrics. Writes `_reset: true` flag in zeroed-out JSON files.
- **Ray 2.55.1 API**: use `Checkpoint.from_directory()`, `train.get_dataset_shard()` (not `get_context().get_dataset_shard()`)
- **`prometheus_client` gauges** must NOT be created at module import time for Ray workers — use lazy init via `_get_ray_gauges()`
- **`ray.init(ignore_reinit_error=True)`** in tests; `ray.shutdown()` in cleanup
- All function signatures need type hints
- **Always run lint before tests after code changes**: `make test-lint` (runs `ruff check .` then `pytest`). Alternatively, run `make lint` first, fix any findings, then `make test`. Never skip lint — it catches issues that tests won't.
- When iterating over DataFrame or batch columns containing token lists, use `list(doc_tokens)` with a `TypeError` catch — never `str(doc_tokens)`. Numpy arrays from parquet are iterable but not `isinstance(x, list)`, and `str()` produces array representations with wrapping quotes
- Scheduler `T_max` must match `default_epochs()` for the model type — otherwise LR decays to minimum before training finishes
- _LinearWarmupCosineScheduler warmup must use `(step + 1) / warmup_steps` to avoid zero LR at step 0
- **Target dtype is `torch.long`** for `CrossEntropyLoss` — never `.float()` on targets. This applies to both DataLoader and Ray paths. The bug was that three `.float()` casts existed in the Ray distributed path — all are now `.long()`.
- **`_load_model()`** is only called in the single-node path. Distributed training (`_run_fit_distributed`) does NOT load the model in the driver process — Ray workers create their own model via `_train_func`.
- **`torchmetrics.MulticlassAUROC`** silently gives wrong results with NaN input — always call `_replace_nan_probs()` before passing probabilities to it
- **`torchmetrics.MulticlassCohenKappa`** returns `nan` for single-class targets — always wrap with `_safe_item()` to coerce `nan→0.0` for Prometheus gauge compatibility
- Single-class Cohen's kappa is `0.0` (not `1.0`) — this is a behavioral change from the previous custom implementation
- Empty arrays must be guarded before calling torchmetrics (it crashes on empty input) — use the `if total == 0` early return in `compute_classification_metrics()`
- **`_balance_dataframe()`** and **`_balance_ray_dataset()`** handle multi-class targets (0, 1, 2), not just binary (0.0, 1.0)
- **`weight_smoothing`** parameter (default `0.5`) controls class weight aggressiveness in `compute_class_weights()`: `1.0` = full inverse frequency, `0.0` = uniform weights
- **`label_smoothing`** default is `0.1` for 3-class `CrossEntropyLoss`
- **`neutral_oversample_ratio`** (default `0.0`) targets a moderate neutral class ratio; `0.20` = oversample neutral to 20% of training data

### ONNX Export

- **RNN `onnx_export` flag**: `RNN.forward(inputs, onnx_export=True)` bypasses `pack_padded_sequence` (ONNX-incompatible) with a masked fallback that extracts hidden states from `lstm_out`. The standard path (default `onnx_export=False`) is unchanged for training and inference.
- **`_RNNOnnxWrapper`**: Wraps RNN to call `forward(inputs, onnx_export=True)` during `torch.onnx.export()` tracing, since `torch.onnx.export()` calls `model(*args)` internally and cannot pass keyword arguments.
- **RNN ONNX tolerance**: Use `1e-2` for RNN ONNX validation (masked fallback has padding drift), `1e-4` for Encoder/Decoder.
- **ONNX opset version**: Use 17 (stable, well-tested). Opset 18+ requires `dynamo_export` which is still preview.
- **Quantization**: Use `onnxruntime.quantization.quantize_dynamic` with `QuantType.QInt8` for INT8 dynamic quantization (FP32 activations, INT8 weights — optimal for AVX-512).
- **`optimum-onnx[onnxruntime]`**: Use this package (not `optimum[onnxruntime]`) — `optimum` v2.0+ moved ONNX to a separate package.
- **ONNX artifacts** go in `onnx_artifacts/` (gitignored). Metadata JSON files are saved alongside each `.onnx` file with model_type, opset_version, input_shape, dictionary path, and validation results.
- **Export CLI**: `sentimentizer export --model rnn --quantize` (also `encoder`, `decoder`)

### SetFit Router

- **Router module**: `sentimentizer/router/` — named `router` to avoid shadowing the `setfit` library.
- **Base model**: Default is `BAAI/bge-base-en-v1.5` (109M params, 768-dim embeddings, strong MTEB scores). Switch to `mxbai-embed-large-v1` (335M params) only if evaluation thresholds are not met.
- **Categories**: Dietary (0), Service (1), General (2) — defined in `RouteLabels` dataclass.
- **Seed utterances**: 10 per category in `sentimentizer/router/seeds.py` — expanded via `augment.py` (GLM 5.1 via Ollama, default model `glm-5.1:cloud`).
- **Training**: `sentimentizer router augment` to generate data, `sentimentizer router train --data augmented_yelp.jsonl` to train — uses `setfit>=1.1.0` with `Trainer` (not deprecated `SetFitTrainer`).
- **Evaluation**: `sentimentizer router evaluate --model-path models/router` — similarity matrix (inter-class < 0.65, intra-class > 0.85) and tau threshold calibration.
- **Router ONNX export**: Deferred to v2 — router uses Python `setfit` inference for now.
- **Optional dependencies**: `pip install -e ".[router]"` for SetFit training, `pip install -e ".[onnx]"` for ONNX export, `pip install -e ".[router,onnx]"` for both.
- **setfit/transformers compatibility**: `setfit 1.1.x` imports `default_logdir` from `transformers.training_args`, which was removed in `transformers 5.x`. The `sentimentizer/compat.py` module includes a monkey-patch shim that injects `default_logdir` if missing. This must be imported BEFORE `import setfit`. The shim is applied automatically by `sentimentizer/router/__init__.py` and `sentimentizer/router/train_router.py`.
- **setfit/config_setfit.json 404**: Sentence-transformer models like `BAAI/bge-base-en-v1.5` don't have `config_setfit.json` on HuggingFace Hub. `huggingface_hub>=1.0` raises a hard 404 error. The `_load_setfit_model()` function in `train_router.py` catches this and falls back to loading via `SentenceTransformer(model_id)` then wrapping with `SetFitModel(model_body=...)`.
- **setfit model_head is None**: When loading a sentence-transformer model as a SetFit backbone (no `config_setfit.json`), `SetFitModel(model_body=...)` does NOT auto-create a classification head. `_load_setfit_model()` creates a `LogisticRegression(max_iter=1000, solver="lbfgs")` head explicitly and sets `model.labels` to the route category names (`["dietary", "service", "general"]`).
