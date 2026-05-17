# Troubleshooting

## 3-Class Migration

### Binary → 3-class changes

The pipeline was migrated from binary (negative/positive, `BCEWithLogitsLoss`) to 3-class (negative/neutral/positive, `CrossEntropyLoss`). Key changes:

| Aspect | Binary (old) | 3-class (current) |
|--------|-------------|-------------------|
| **Classes** | 2 (negative, positive) | 3 (negative, neutral, positive) |
| **Loss** | `BCEWithLogitsLoss` with `pos_weight` | `CrossEntropyLoss` with `class_weights` and `label_smoothing=0.1` |
| **Target dtype** | `torch.float32` | `torch.long` |
| **Model output** | `(B, 1)` logits | `(B, 3)` logits |
| **Final activation** | `sigmoid` | `softmax` |
| **Rating mapping** | 1-2★ → 0, 4-5★ → 1 | 1-2★ → 0, 3★ → 1, 4-5★ → 2 |
| **`predict_text()` return** | `float` (0-1 score) | `dict[str, float]` (e.g., `{"negative": 0.05, "neutral": 0.12, "positive": 0.83}`) |
| **Class balance** | `compute_pos_weight()` → single float | `compute_class_weights()` → 3-element tensor, `weight_smoothing` parameter |
| **NaN replacement** | `0.5` (binary random) | `1/num_classes` ≈ `0.333` (uniform over 3 classes) |
| **Configuration** | `NUM_CLASSES=2`, `pos_weight` | `NUM_CLASSES=3`, `LABEL_NAMES`, `class_weights`, `loss_type`, `focal_gamma`, `label_smoothing`, `weight_smoothing`, `neutral_oversample_ratio` |

### Common issues after migration

#### Old binary weights crash on load

**Symptoms**: `RuntimeError` about tensor shape mismatch (expecting `(3, hidden_dim)` but got `(1, hidden_dim)`).

**Fix**: Old binary weights are incompatible with the 3-class architecture. Delete them and retrain:

```bash
rm models/rnn_sentiment.model
rm models/encoder_sentiment.model
rm models/decoder_sentiment.model
make train MODEL=rnn
```

#### `compute_pos_weight()` raises `ValueError`

**Symptoms**: `ValueError: compute_pos_weight() is deprecated for 3-class classification. Use compute_class_weights() instead.`

**Fix**: Replace all `compute_pos_weight()` calls with `compute_class_weights()`. The old function raises an error for `num_classes > 2`.

#### Target dtype errors in training

**Symptoms**: `RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long, but got Float`.

**Fix**: Targets must be `torch.long` for `CrossEntropyLoss`, not `torch.float32`. This applies to both DataLoader and Ray distributed paths. Search for any residual `.float()` casts on targets and replace with `.long()`.

#### `torch.squeeze` removing class dimension

**Symptoms**: Shape `(3,)` instead of `(1, 3)` for batch-of-1 predictions.

**Fix**: `forward()` must always return `(B, num_classes)`, never squeeze. `predict()` uses `torch.softmax(logits, dim=-1)` returning `(B, 3)`.

#### Neutral class has low recall (< 0.60)

**Symptoms**: `neutral_recall` below 0.60, `pred_neutral_frac` below 0.05.

**Fix (escalate through these)**:
1. Default: `CrossEntropyLoss` with `class_weights` (smoothed) + `label_smoothing=0.1`
2. Focal loss: `loss_type="focal"`, `focal_gamma=2.0`
3. Oversampling: `neutral_oversample_ratio=0.20`
4. Reduce `weight_smoothing` toward `0.0` for more aggressive class weighting

## Model Training Issues

### Zero neutral-class recall / Cohen's kappa = 0

**Symptoms**: After training, `neutral_recall` is 0, `cohen_kappa` is 0, and the model predicts the majority class (positive) for every input.

**Root cause**: The model is effectively not learning word meanings. Nearly all tokens are mapped to the out-of-vocabulary (OOV) embedding, so the model trains on random vectors.

**Check**: Look at the GloVe match rate in the logs. If it's below ~50%, the dictionary is corrupted:

```
matched 23/20000 dictionary words to glove-wiki-gigaword-100 vectors (0.1%)
```

A healthy match rate should be 50–80%.

**Common causes and fixes**:

| Cause | Symptoms | Fix |
|-------|----------|-----|
| **Dictionary tokens have wrapping quotes** | `extract_embeddings()` raises `ValueError` for match rate < 50%; dictionary contains `"'the'"` instead of `"the"` | Since the fix, `new_dictionary()` and `_count_vocab_batch()` use `list(doc_tokens)` instead of `str(doc_tokens)` for numpy arrays. If you have a stale `.dictionary` file, delete it and re-run the pipeline with `--run-type new`. |
| **Wrong T_max for scheduler** | Loss decays to minimum halfway through training; model stops learning after ~50% of epochs | `EncoderSchedulerParams.T_max` should equal `default_epochs("encoder")` (= 8). Verify in `config.py`. |
| **Zero LR during warmup** | First epoch shows no learning progress | `_LinearWarmupCosineScheduler` uses `(step + 1) / warmup_steps` instead of `step / warmup_steps`. Verify the warmup formula is correct. |
| **Class imbalance not addressed** | Model always predicts positive; `balanced_accuracy` near 0.33 (random) | Use `class_weights` (default with `compute_class_weights()`), `label_smoothing=0.1`, or `loss_type="focal"` with `focal_gamma=2.0`. |

### Dictionary has wrapping quotes (`'the'` instead of `the`)

This occurs when parquet stores the `tokens` column as numpy arrays. The code `isinstance(doc_tokens, list)` returns `False` for numpy arrays, and the fallback `regex_tokenize(str(numpy_array))` stringifies the array representation (`"['if' 'you' ...]"`), causing the regex `[a-z0-9'-]+` to capture the quotes as part of each token.

**Affected files** (all fixed as of the current version):
- `sentimentizer/tokenizer.py` — `new_dictionary()` and `_count_vocab_batch()` now use `list(doc_tokens)` with `TypeError` catch
- `workflows/stages/tokenize.py` — resume path lambda now uses `hasattr(x, '__iter__')` instead of `isinstance(x, list)`

**Recovery**: Delete the dictionary file and re-run the pipeline:

```bash
rm sentimentizer/data/yelp.dictionary
make train   # or make train-distributed
```

### ValueError: GloVe vocabulary match rate is below 50%

`extract_embeddings()` in `sentimentizer/extractor.py` raises a `ValueError` if fewer than 50% of dictionary words match GloVe vectors. This is a guardrail against the wrapping-quotes bug and similar data corruption.

**Fix**: See "Dictionary has wrapping quotes" above. After fixing the dictionary, re-run the pipeline.

## Scheduler Issues

### Learning rate drops to minimum before training finishes

**Symptoms**: Training loss plateaus or increases after ~50% of epochs; final metrics are poor.

**Cause**: `T_max` (for `CosineAnnealingLR` or `_LinearWarmupCosineScheduler`) is set lower than the number of training epochs. The LR decays to `eta_min` by epoch `T_max` and stays there for the remaining epochs.

**Fix**: Ensure `EncoderSchedulerParams.T_max` equals `default_epochs("encoder")` (= 8). A test in `TestSchedulerCorrectness.test_scheduler_t_max_matches_default_epochs` catches regressions.

### Zero learning rate on first epoch (warmup)

**Symptoms**: First epoch shows no learning progress; loss barely changes.

**Cause**: The warmup formula `step / warmup_steps` returns 0.0 at `step=0`, making the LR zero for the entire first epoch.

**Fix**: `_LinearWarmupCosineScheduler` uses `(step + 1) / warmup_steps`, which gives `1/warmup_steps` at `step=0`. A test in `TestSchedulerCorrectness.test_warmup_cosine_no_zero_lr` catches regressions.

## Data Pipeline Issues

### Class imbalance in Yelp dataset

The Yelp reviews dataset has roughly 3.5:1 positive-to-negative ratio with neutral (~11%) being the smallest class. Without addressing this, the model learns to always predict positive.

**Solutions** (use one or more, in escalation order):
- **Smoothed class weights** (default): `compute_class_weights()` with `weight_smoothing=0.5` produces sqrt-scaled inverse frequency weights. Controlled by `--weight-smoothing` CLI flag.
- **Label smoothing** (default `0.1`): Regularizes the model from making overconfident predictions. Controlled by `--label-smoothing` CLI flag.
- **Focal loss**: `--loss-type focal --focal-gamma 2.0` applies higher loss to hard examples. Useful when class weights alone aren't enough.
- **Neutral oversampling**: `--neutral-oversample-ratio 0.20` targets 20% neutral reviews in training data via duplication.
- **Undersampling**: `--balance-strategy class_weights_only` (default) avoids aggressive undersampling that destroys 89% of data.

All options are available for `train`, `train-distributed`, and `run` commands.

### Stale metrics from previous training run on dashboard

See the [Stale Metrics Reset](#stale-metrics-reset) section in `metrics.md` for details. `_reset_stale_metrics(model_type)` is called automatically at the start of every training run and zeroes **all** model types (rnn, encoder, decoder), not just the current one.

## GPU / CUDA Issues

### Distributed training runs on CPU instead of GPU

**Symptoms**: `make train-distributed MODEL=encoder` trains on CPU; `torch.cuda.is_available()` returns `False`; `resolve_device("auto")` returns `"cpu"`; dashboard shows RNN metrics from a previous run instead of encoder metrics.

**Root cause**: If you previously ran `make setup-ci` or had the CPU-only torch source configured, `torch` resolves from the CPU-only PyTorch wheel index. This gives `torch 2.x+cpu` which has no CUDA support — even when NVIDIA libraries (`nvidia-cublas`, `nvidia-cuda-runtime`, etc.) are installed, `torch.cuda.is_available()` returns `False`.

**Fix**: Install CUDA-enabled torch (the default since `pyproject.toml` no longer pins CPU-only):

```bash
make setup
```

This runs `uv sync --extra ray` which resolves torch from PyPI with CUDA support.

This tells `uv` to ignore the `[tool.uv.sources]` override and resolve torch from PyPI, which includes CUDA support.

**Diagnostic**: `resolve_device("auto")` logs a warning when it detects CPU-only torch with NVIDIA libraries installed:

```
torch 2.11.0+cpu is CPU-only but NVIDIA CUDA libraries are installed.
Distributed and GPU training will use CPU.
Install the CUDA variant with: uv sync --no-sources-package torch
```

## Running Tests

```bash
# All tests
make test

# Skip Ray-dependent tests (faster, no Ray initialization)
uv run pytest tests/ -v -k "not Ray"

# Only run dictionary/tokenizer tests
uv run pytest tests/test_dictionary_lifecycle.py tests/test_loader.py -v

# Only run scheduler correctness tests
uv run pytest tests/test_training.py -v -k "TestSchedulerCorrectness"
```

## ONNX Export Issues

### RNN ONNX validation fails with high max_diff

**Symptoms**: `validate_onnx_export()` reports `max_diff > 1e-2` for RNN models.

**Cause**: The RNN's ONNX-compatible forward path (masked fallback) skips `pack_padded_sequence` and processes padding tokens. Short sequences have more padding, causing larger numeric drift.

**Fix**: This is expected. The tolerance for RNN is intentionally set to `1e-2` (vs `1e-4` for Encoder/Decoder). If validation still fails:
- Ensure the model was exported with `_RNNOnnxWrapper` (calls `forward(onnx_export=True)`)
- Check that `seq_len` in `export_model_to_onnx()` matches the training sequence length
- Verify the model is in `eval()` mode before export

### `onnxruntime` import error during export

**Symptoms**: `ModuleNotFoundError: No module named 'onnxruntime'` when running `sentimentizer export`.

**Fix**: Install the ONNX optional dependency group:

```bash
pip install -e ".[onnx]"
# or with uv:
uv sync --extra onnx
```

### `setfit` import error during router training

**Symptoms**: `ModuleNotFoundError: No module named 'setfit'` when running `sentimentizer router train`.

**Fix**: Install the router optional dependency group:

```bash
pip install -e ".[router]"
# or with uv:
uv sync --extra router
```

### optimum-onnx conflict with numpy

**Symptoms**: `pip install -e ".[onnx]"` fails with numpy version conflict involving `optimum-onnx`.

**Fix**: The `[onnx]` dependency group does not include `optimum-onnx` because it conflicts with `numpy>=2.4.0`. ONNX export and quantization work directly with `onnxruntime.quantization.quantize_dynamic`. If you need `optimum` for SetFit ONNX export (v2), pin `optimum[onnxruntime]<2.0.0`.

## SetFit Router Issues

### Ollama API connection failure during augmentation

**Symptoms**: `augment_seeds()` logs `Ollama API request failed` and returns only the original seeds.

**Fix**: Ensure Ollama is running locally on port 11434:

```bash
ollama serve  # Start the Ollama API server
ollama pull glm-5.1:cloud  # Pull the model (cloud variant)
```

The `augment_seeds()` function gracefully handles API failures — it returns the original seeds unchanged, so training can proceed with just the 30 golden examples if augmentation is unavailable.

### Low inter-class similarity (> 0.65) after training

**Symptoms**: `evaluate_router()` reports inter-class similarity above 0.65, meaning the model cannot distinguish between categories well.

**Fix**: This typically means:
1. Too few training examples — run augmentation to expand seeds
2. Hard negatives not represented — ensure `augment.py` generates category-confusing examples
3. Consider upgrading the base model from `BAAI/bge-base-en-v1.5` to `mxbai-embed-large-v1` in `SetFitConfig`

## Linting and Formatting

```bash
make check   # Runs ruff check --fix, ruff check, and black --check
make format  # Runs ruff check --fix and black .
```
