# Troubleshooting

## Model Training Issues

### Zero negative-class accuracy / Cohen's kappa = 0

**Symptoms**: After training, `negative_accuracy` is 0, `cohen_kappa` is 0, and `positive_accuracy` is high (often matching the fraction of positive reviews in the dataset). The model predicts the majority class for every input.

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
| **Class imbalance not addressed** | Model always predicts positive; `pos_weight ≈ 1.0` | Pass `--pos-weight 0.0` (auto-calculates `neg_count/pos_count`) or `--balance-classes` to undersample the majority class. |

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

The Yelp reviews dataset has roughly 3.5:1 positive-to-negative ratio (5-star and 4-star reviews dominate). Without addressing this, the model learns to always predict positive.

**Solutions** (use one or both):
- `--pos-weight 0.0`: Auto-calculates `neg_count/pos_count` and passes it to `BCEWithLogitsLoss`, reducing the loss weight for the majority class
- `--balance-classes`: Undersamples the majority class to create equal class counts

Both options are available for `train`, `train-distributed`, and `run` commands.

### Stale metrics from previous training run on dashboard

See the [Stale Metrics Reset](#stale-metrics-reset) section in `metrics.md` for details. `_reset_stale_metrics(model_type)` is called automatically at the start of every training run.

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

## Linting and Formatting

```bash
make check   # Runs ruff check --fix, ruff check, and black --check
make format  # Runs ruff check --fix and black .
```