# fastText Embeddings Integration Plan

> **Status: NOT YET IMPLEMENTED** — This is a future plan. The 3-class migration has been completed; this can be implemented independently or bundled with a future embedding upgrade.
> Current status: Using GloVe-100 embeddings with `NUM_CLASSES=3`, `CrossEntropyLoss`, and `compute_class_weights()`.

## Background

The pipeline currently uses GloVe Wiki-Gigaword-100 (100-dim, context-free, trained on Wikipedia/news). This has two problems for Yelp sentiment analysis:

1. **Domain mismatch** — Informal words common in reviews ("meh", "overpriced", "mouthwatering", "waitress") either map to OOV or have weak vectors because they rarely appear in Wikipedia/news
2. **100 dims is small** — Limited capacity to distinguish fine-grained sentiment (great vs good vs okay vs mediocre)
3. **No OOV handling** — Unknown words get a random embedding vector (line 157 in `extractor.py`), losing all morphological signal ("unflavored" → random, even though "flavor" has a good vector)

fastText solves all three: 300-dim embeddings trained on a larger corpus, with character n-gram subword generation that produces meaningful vectors for any word — even words never seen during training.

## Scope

Two tiers:

- **Tier 1** (5 min): Switch from GloVe-100 to fastText-300 via `gensim.downloader` — no OOV generation, just better pretrained vectors
- **Tier 2** (1-2 hours): Load the full fastText `.bin` model for OOV subword generation — this is the real win

Both tiers are backward-compatible: GloVe remains available via config.

---

## Tier 1: fastText via gensim.downloader (drop-in replacement)

### `sentimentizer/config.py`

Change `EmbeddingsConfig` defaults:

```python
@dataclass
class EmbeddingsConfig:
    model_name: str = "fasttext-wiki-news-subwords-300"  # was "glove-wiki-gigaword-100"
    emb_length: int = 300                                # was 100
```

That's it. `gensim_api.load("fasttext-wiki-news-subwords-300")` returns `KeyedVectors` — same API as GloVe. `extract_embeddings()` works unchanged. The gensim downloader auto-caches to `~/gensim-data/` on first use (~600MB download).

### Impact

- 3× embedding capacity (300 vs 100 dims)
- Better vocabulary coverage (trained on Wikipedia + news, 1M word vectors vs GloVe's 400K)
- **No OOV generation** — words not in the 1M vocabulary still hit the random fallback
- Embedding matrix is 3× larger: `(vocab_size + 2, 300)` vs `(vocab_size + 2, 100)` — negligible memory impact for a 20K vocab

### Files changed

Only `config.py` — update two default values. All model architectures accept `emb_length` from config, no hardcoded dimensions.

### Test

```bash
# Verify match rate is still high
uv run pytest tests/ -v -k "embedding"
# Retrain one model and compare metrics
sentimentizer train --model rnn
```

---

## Tier 2: Full fastText model with OOV subword generation

### Overview

The full fastText `.bin` model (~7.2GB for `crawl-300d-2M-subword.bin`) contains character n-gram tables that can synthesize vectors for any word. This means:
- "mouthwatering" → composed from n-grams of "mouth", "water", etc.
- "unflavorful" → composed from "un", "flavor", "ful"
- "waitresss" (typo) → still gets a reasonable vector from shared n-grams with "waitress"

### `sentimentizer/config.py`

Extend `EmbeddingsConfig`:

```python
@dataclass
class EmbeddingsConfig:
    model_name: str = "fasttext-wiki-news-subwords-300"
    emb_length: int = 300
    # Full fastText model for OOV subword generation.
    # Set to path of .bin file to enable. None = use gensim.downloader (no OOV generation).
    fasttext_bin_path: str | None = None
```

### `sentimentizer/extractor.py`

Modify `extract_embeddings()` to support the full fastText model:

```python
@time_decorator
def extract_embeddings(
    dictionary: corpora.Dictionary, cfg: EmbeddingsConfig
) -> dict[int, np.ndarray]:
    """Load pre-trained word vectors.

    If cfg.fasttext_bin_path is set, loads the full fastText model
    which can generate vectors for OOV words via character n-grams.
    Otherwise falls back to gensim.downloader (GloVe or fastText KeyedVectors).
    """
    if cfg.fasttext_bin_path is not None:
        return _extract_fasttext_full(dictionary, cfg)

    # Existing gensim.downloader path (GloVe or fastText KeyedVectors)
    logger.info(f"loading embeddings model: {cfg.model_name} ...")
    model = gensim_api.load(cfg.model_name)

    embeddings_dict: dict[int, np.ndarray] = {}
    for word, token_id in dictionary.token2id.items():
        if word in model:
            embeddings_dict[token_id + 1] = model[word].astype(EMBEDDING_DTYPE)

    _log_match_rate(embeddings_dict, dictionary, cfg.model_name)
    return embeddings_dict


def _extract_fasttext_full(
    dictionary: corpora.Dictionary, cfg: EmbeddingsConfig
) -> dict[int, np.ndarray]:
    """Load full fastText .bin model for OOV subword generation."""
    from gensim.models.fasttext import load_facebook_model

    logger.info(f"loading full fastText model: {cfg.fasttext_bin_path} ...")
    ft_model = load_facebook_model(cfg.fasttext_bin_path)
    wv = ft_model.wv

    embeddings_dict: dict[int, np.ndarray] = {}
    oov_generated = 0

    for word, token_id in dictionary.token2id.items():
        try:
            # fastText wv[word] works for ALL words — in-vocab uses stored
            # vectors, OOV uses character n-gram composition
            vec = wv[word].astype(EMBEDDING_DTYPE)
            embeddings_dict[token_id + 1] = vec

            if word not in wv.key_to_index:
                oov_generated += 1
        except KeyError:
            # Should never happen with fastText, but guard anyway
            continue

    in_vocab = len(embeddings_dict) - oov_generated
    logger.info(
        f"fastText: {in_vocab}/{len(dictionary)} in-vocab, "
        f"{oov_generated}/{len(dictionary)} OOV (subword-generated), "
        f"{len(dictionary) - len(embeddings_dict)}/{len(dictionary)} failed"
    )
    return embeddings_dict


def _log_match_rate(
    embeddings_dict: dict[int, np.ndarray],
    dictionary: corpora.Dictionary,
    model_name: str,
) -> None:
    """Log and validate embedding match rate."""
    match_rate = len(embeddings_dict) / max(len(dictionary), 1)
    logger.info(
        f"matched {len(embeddings_dict)}/{len(dictionary)} dictionary words "
        f"to {model_name} vectors ({match_rate:.1%})"
    )
    if match_rate < 0.5:
        raise ValueError(
            f"Vocabulary match rate is {match_rate:.1%} "
            f"({len(embeddings_dict)}/{len(dictionary)} words). "
            f"This usually means the dictionary tokens have wrapping quotes "
            f'(e.g. "\'the\'" instead of "the") caused by calling str() on '
            f"numpy arrays. Check that new_dictionary() and _count_vocab_batch() "
            f"use list() instead of str() on non-list iterables."
        )
```

### `new_embedding_weights()` change

When using full fastText, most/all dictionary words get vectors (even OOV ones via n-grams), so the random fallback in `new_embedding_weights()` is rarely triggered. No code change needed — the existing random fallback still applies for the rare case where fastText truly can't generate a vector.

### `workflows/cli.py`

Add CLI argument:

```
--fasttext-bin-path   Path to fastText .bin file for OOV subword generation (optional)
```

### Downloading the fastText model

The recommended model is `crawl-300d-2M-subword.bin` from Facebook Research:

```bash
# ~7.2GB download, one-time setup
mkdir -p data/embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip -O data/embeddings/crawl-300d-2M-subword.zip
unzip data/embeddings/crawl-300d-2M-subword.zip -d data/embeddings/
```

Add to `.gitignore`:
```
data/embeddings/*.bin
data/embeddings/*.zip
```

### Memory considerations

The full fastText model loads into ~8GB RAM (it needs the n-gram tables). This is fine for training (which already uses several GB for data), but too large for lightweight inference. The embedding matrix itself is the same size — once `new_embedding_weights()` has built the `(vocab_size + 2, 300)` matrix, the fastText model can be garbage collected. Only the training/tokenization pipeline loads the full model.

---

## Files changed summary

### Tier 1 (drop-in)

| File | Change |
|------|--------|
| `sentimentizer/config.py` | Update `model_name` and `emb_length` defaults |

### Tier 2 (OOV generation)

| File | Change |
|------|--------|
| `sentimentizer/config.py` | Add `fasttext_bin_path: str \| None = None` field |
| `sentimentizer/extractor.py` | Add `_extract_fasttext_full()`, refactor `_log_match_rate()` |
| `workflows/cli.py` | Add `--fasttext-bin-path` CLI argument |
| `.gitignore` | Add `data/embeddings/` exclusions |

### No changes needed

These files all consume `EmbeddingsConfig` but don't need modification — they pass the config through to `new_embedding_weights()` which handles the dispatch:

- `sentimentizer/models/rnn.py` — calls `new_embedding_weights(dict, cfg)`
- `sentimentizer/models/encoder.py` — calls `new_embedding_weights(dict, cfg)`
- `sentimentizer/models/decoder.py` — calls `new_embedding_weights(dict, cfg)`
- `sentimentizer/trainer.py` — passes `EmbeddingsConfig` to model constructors
- `sentimentizer/tuner.py` — constructs `EmbeddingsConfig` from tuning config
- `sentimentizer/agent/diagnose_model.py` — constructs `EmbeddingsConfig` for diagnostics

---

## Testing

### New tests

```python
class TestFastTextEmbeddings:
    """Tests for fastText embedding integration."""

    def test_gensim_downloader_fasttext(self) -> None:
        """fasttext-wiki-news-subwords-300 loads via gensim.downloader."""
        cfg = EmbeddingsConfig(model_name="fasttext-wiki-news-subwords-300", emb_length=300)
        # Just verify config is valid — actual download is slow
        assert cfg.emb_length == 300

    def test_extract_fasttext_full_generates_oov(self) -> None:
        """Full fastText model generates vectors for OOV words."""
        # Requires crawl-300d-2M-subword.bin — skip if not available
        cfg = EmbeddingsConfig(fasttext_bin_path="data/embeddings/crawl-300d-2M-subword.bin")
        # Create a dictionary with an OOV word
        from gensim import corpora
        d = corpora.Dictionary([["mouthwatering", "unflavored", "the"]])
        embeddings = _extract_fasttext_full(d, cfg)
        # All words should get vectors (including OOV ones)
        assert len(embeddings) == len(d)

    def test_embedding_matrix_shape_300d(self) -> None:
        """Embedding matrix has correct shape with 300d embeddings."""
        cfg = EmbeddingsConfig(emb_length=300)
        # ... verify (vocab + 2, 300) shape

    def test_glove_still_works(self) -> None:
        """GloVe embeddings still work with explicit config."""
        cfg = EmbeddingsConfig(model_name="glove-wiki-gigaword-100", emb_length=100)
        assert cfg.emb_length == 100
```

### Existing tests

- `tests/test_ray_compat.py` `TestEmbeddingsConfigAttributes` — add test for new `fasttext_bin_path` attribute
- `tests/test_training.py` embedding config tests — verify backward compat with GloVe config

---

## Expected impact on sentiment models

| Metric | GloVe-100 (current) | fastText-300 KeyedVectors (Tier 1) | fastText-300 + OOV (Tier 2) |
|--------|---------------------|-----------------------------------|---------------------------|
| OOV rate | ~8-12% | ~3-5% | **~0%** |
| Vocab match rate | ~85% | ~95% | **~100%** |
| Embedding dim | 100 | 300 | 300 |
| Neutral recall (est.) | ~0.55 | ~0.60 | **~0.65** |
| Macro F1 (est.) | ~0.70 | ~0.73 | **~0.75** |

The biggest gain is OOV elimination. Informal review language (slang, compound words, misspellings) is disproportionately concentrated in neutral reviews — people use hedging language that tends to be non-standard.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| **All existing weights become invalid** | High | Changing from 100-dim to 300-dim embeddings changes every model's input layer size. Existing `.pth` weights, HuggingFace Hub weights (`ryeyoo/sentimentizer-*`), and exported ONNX models all become incompatible. **Bundle this with the 3-class migration to avoid retraining twice.** |
| **`serve.py` hardcodes `"embedding_dim": 100`** | Low | Lines 65, 73, 82 in `MODEL_REGISTRY` have `"embedding_dim": 100` for all three models. This is metadata for the `/models` endpoint (not functional), but must be updated to `300` or read from config. |
| **Test fixtures hardcode `100` in ~8 places** | Medium | `test_rnn.py` (lines 321, 361, 486), `test_ray_compat.py` (lines 227, 355), and `test_training.py` (lines 595, 615, 646-658) all have `"embeddings_emb_length": 100`. Update to `300` or make config-driven. |
| **~30-50% slower training** | Medium | 300-dim embeddings increase LSTM input size 3× (LSTM params scale as `4 * hidden * (input + hidden)`). Encoder/decoder embedding-to-d_model projection goes from `(100, 256)` to `(300, 256)`. Acceptable tradeoff for better embeddings, but worth measuring. |
| **Full `.bin` model is 7.2GB (Tier 2 only)** | Medium | Too large for CI runners with limited disk. Must be downloaded manually. Loading takes ~30s and ~8GB RAM, but only during `new_embedding_weights()` — not during inference. Add a `pytest.mark.skipif` for tests that need the `.bin` file. |
| **No architecture code changes needed** | — | Model architectures (`rnn.py`, `encoder.py`, `decoder.py`) all read embedding dimensions from the weight matrix shape, not hardcoded values. The pipeline is cleanly encapsulated in `extractor.py`. |

### Coordination with 3-class migration

⚠️ **Strongly recommended: bundle Tier 1 with the 3-class migration.** Both changes invalidate existing weights and require full retraining. Doing them separately means retraining all 3 models twice and pushing two sets of weights to HuggingFace Hub. Combined rollout:

1. PR 1-4 of 3-class migration (code changes)
2. Tier 1 config change (two values in `config.py`, metadata in `serve.py`)
3. PR 5 of 3-class migration (retrain with 3-class labels + 300-dim fastText)
4. Push v2 weights to HuggingFace Hub (one set, not two)

Tier 2 (OOV generation) can be added later without retraining — it only improves the embedding matrix quality, which triggers a retrain by choice, not by necessity.

---

## Rollout order

1. **Tier 1 + 3-class migration** — change config defaults, implement 3-class changes, retrain all models once. Ship as a combined v2 release.
2. **Measure** — check OOV match rate and neutral recall after retraining.
3. **If OOV rate is still >3%** — implement Tier 2, download `.bin` file, retrain, measure. 1-2 hours of code, plus retraining time.
4. **Skip Tier 2** if Tier 1 already meets neutral recall targets (≥ 0.60) with smoothed class weights + focal loss.
