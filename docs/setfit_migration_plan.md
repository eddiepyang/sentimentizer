# SetFit Migration Plan: Remove `setfit` Dependency

**Date**: 2025-05-18
**Status**: Planning (not yet implemented)

---

## Goal

Remove the `setfit` package dependency from the router module while maintaining identical functionality (3-class text routing: Dietary, Service, General). Replace SetFit's thin wrapper with direct use of `sentence-transformers` + `sklearn`, which SetFit already wraps internally.

---

## Current State

### SetFit Usage Summary

SetFit is an **optional** dependency (`[router]` extra in `pyproject.toml`) used for a 3-class contrastive-learning text router. The core SetFit approach is:

1. **Contrastive fine-tuning**: Generate same-class/different-class sentence pairs → train sentence-transformer backbone with `CosineSimilarityLoss`
2. **Classification head**: Encode training data with fine-tuned backbone → fit `LogisticRegression` on embeddings
3. **Inference**: `model_encode()` → `predict()` / `predict_proba()` via the head

SetFit is essentially a thin wrapper around `sentence-transformers` + `sklearn`. The migration replaces it with direct use of those libraries.

### SetFit API Surface to Replace

| SetFit API | Used In | Replacement |
|---|---|---|
| `SetFitModel.from_pretrained(model_id)` | `train_router.py`, `predictor.py`, `cli.py` | `RouterModel.from_pretrained()` — loads backbone via `SentenceTransformer` + head via `joblib` |
| `SetFitModel(model_body=...)` | `train_router.py` | `RouterModel(backbone=..., head=..., labels=...)` |
| `model.model_head = LogisticRegression(...)` | `train_router.py` | Direct assignment on `RouterModel.head` |
| `model.labels = [...]` | `train_router.py` | Direct assignment on `RouterModel.labels` |
| `model.model_encode(texts)` | `evaluate.py` | `RouterModel.model_encode()` → `backbone.encode()` |
| `model.predict(texts)` | `evaluate.py`, `predictor.py` | `RouterModel.predict()` → `head.predict(embeddings)` |
| `model.predict_proba(texts)` | `evaluate.py`, `predictor.py` | `RouterModel.predict_proba()` → `head.predict_proba(embeddings)` |
| `model.save_pretrained(path)` | `train_router.py` | `RouterModel.save_pretrained()` — saves backbone + head + metadata |
| `model.push_to_hub(repo_id)` | `cli.py` | `RouterModel.push_to_hub()` — via `huggingface_hub` |
| `setfit.Trainer` + `TrainingArguments` | `train_router.py` | Custom `train_router()` with sentence-transformers `CosineSimilarityLoss` |
| `trainer.train()` | `train_router.py` | `backbone.fit(train_objectives=[...])` |
| `sentimentizer.compat` shim | `__init__.py`, `predictor.py`, `cli.py`, `app.py` | **Delete entirely** — only exists for setfit's `default_logdir` import |

### Files Using SetFit (12 files)

| File | SetFit Usage |
|---|---|
| `sentimentizer/router/train_router.py` | `from setfit import SetFitModel, Trainer, TrainingArguments` |
| `sentimentizer/router/evaluate.py` | `from setfit import SetFitModel` (TYPE_CHECKING only) |
| `sentimentizer/router/__init__.py` | `import sentimentizer.compat` (shim for setfit) |
| `sentimentizer/router/config.py` | `SetFitConfig` dataclass |
| `sentimentizer/router/config.yaml` | Comments referencing "SetFit" |
| `sentimentizer/predictor.py` | `from setfit import SetFitModel`; `import sentimentizer.compat` |
| `sentimentizer/serve/app.py` | `import sentimentizer.compat` |
| `workflows/cli.py` | `from setfit import SetFitModel` in `router_evaluate` and `router_push` commands |
| `sentimentizer/compat.py` | Monkey-patches `transformers.training_args.default_logdir` for setfit |
| `pyproject.toml` | `setfit>=1.1.0` in `[project.optional-dependencies] router` |
| `tests/test_router.py` | `import setfit` availability check; `SETHOOK_AVAILABLE` skip markers |
| `AGENTS.md` | Extensive SetFit/router documentation |

---

## Proposed Architecture

### New `RouterModel` class (`sentimentizer/router/model.py`)

Drop-in replacement for `SetFitModel` with identical external API:

```python
class RouterModel:
    """Sentence-transformer backbone + LogisticRegression head.
    
    Drop-in replacement for SetFitModel with equivalent API:
    - from_pretrained() / save_pretrained() / push_to_hub()
    - model_encode() / predict() / predict_proba()
    """
    
    def __init__(self, backbone: SentenceTransformer, head: LogisticRegression | None = None, 
                 labels: list[str] | None = None): ...
    
    @classmethod
    def from_pretrained(cls, model_id: str, num_classes: int = 3) -> RouterModel:
        """Load from a directory saved by save_pretrained(), or from a 
        sentence-transformer model ID on HuggingFace (creates fresh head)."""
        ...
    
    def model_encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings. Returns (N, D) ndarray."""
        ...
    
    def predict(self, texts: list[str]) -> list[int | str]:
        """Predict class labels for texts."""
        ...
    
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probabilities for texts. Returns (N, num_classes)."""
        ...
    
    def save_pretrained(self, path: str) -> None:
        """Save backbone (via sentence-transformers) + head (via joblib) + metadata (via JSON)."""
        ...
    
    def push_to_hub(self, repo_id: str) -> None:
        """Push model directory to HuggingFace Hub via huggingface_hub."""
        ...
    
    @classmethod
    def _migrate_legacy_setfit_model(cls, path: str) -> RouterModel | None:
        """Attempt to load a legacy SetFit model directory. 
        Returns None if directory doesn't contain config_setfit.json."""
        ...
```

### Save Format (directory structure)

```
models/router/
├── config.json              # sentence-transformer backbone config
├── model.safetensors        # backbone weights
├── tokenizer.json           # tokenizer config
├── tokenizer_config.json    # tokenizer metadata
├── special_tokens_map.json  # special tokens
├── 1_Pooling/               # pooling config
│   └── config.json
├── 2_Normalize/             # normalization config (if applicable)
├── router_head.joblib       # pickled LogisticRegression head
└── router_config.json       # {num_classes, label_names, model_type: "router"}
```

### Updated `train_router()` (`sentimentizer/router/train_router.py`)

Replace SetFit's `Trainer` with three explicit steps:

```python
def train_router(config, train_dataset, eval_dataset=None) -> RouterModel:
    # Step 1: Load backbone
    backbone = SentenceTransformer(config.base_model)
    
    # Step 2: Generate contrastive pairs
    pair_dataset = generate_contrastive_pairs(train_dataset, num_iterations=config.num_iterations)
    
    # Step 3: Fine-tune backbone with CosineSimilarityLoss
    train_dataloader = DataLoader(pair_dataset, batch_size=config.batch_size, shuffle=True)
    train_loss = CosineSimilarityLoss(model=backbone)
    backbone.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.num_epochs,
        output_path=str(config.output_dir),
    )
    
    # Step 4: Fit classification head (rank 0 only in distributed)
    train_texts = train_dataset["text"]
    train_labels = train_dataset["label"]
    embeddings = backbone.encode(train_texts)
    head = LogisticRegression(max_iter=1000, solver="lbfgs")
    head.fit(embeddings, train_labels)
    
    # Step 5: Build and save RouterModel
    labels = list(RouteLabels.label_names().values())[:config.num_classes]
    model = RouterModel(backbone=backbone, head=head, labels=labels)
    model.save_pretrained(str(config.output_dir))
    return model
```

### Contrastive Pair Generation (ported from SetFit)

```python
def generate_contrastive_pairs(
    dataset, num_iterations: int = 20, seed: int = 42
) -> list[dict]:
    """Generate same-class and different-class sentence pairs.
    
    Ported from SetFit's ContrastiveDataset logic:
    - For each example, sample num_iterations same-class pairs (label=1.0)
    - Sample num_iterations different-class pairs (label=0.0)
    - No self-pairing (text never paired with itself)
    """
    rng = np.random.default_rng(seed)
    pairs = []
    
    # Group texts by class
    texts_by_class = defaultdict(list)
    for text, label in zip(dataset["text"], dataset["label"]):
        texts_by_class[label].append(text)
    
    classes = list(texts_by_class.keys())
    
    for text, label in zip(dataset["text"], dataset["label"]):
        same_class_texts = [t for t in texts_by_class[label] if t != text]
        if not same_class_texts:
            continue
        
        # Same-class pairs (label=1.0)
        for _ in range(num_iterations):
            partner = rng.choice(same_class_texts)
            pairs.append({"sentence_a": text, "sentence_b": partner, "label": 1.0})
        
        # Different-class pairs (label=0.0)
        other_classes = [c for c in classes if c != label]
        for _ in range(num_iterations):
            other_class = rng.choice(other_classes)
            partner = rng.choice(texts_by_class[other_class])
            pairs.append({"sentence_a": text, "sentence_b": partner, "label": 0.0})
    
    rng.shuffle(pairs)
    return pairs
```

---

## Files Changed

| File | Change Type | Description |
|---|---|---|
| `sentimentizer/router/model.py` | **NEW** | `RouterModel` class with full API surface |
| `sentimentizer/router/train_router.py` | **Rewrite** | Remove setfit imports; use `SentenceTransformer` + `CosineSimilarityLoss`; add `generate_contrastive_pairs()`; return `RouterModel` |
| `sentimentizer/router/evaluate.py` | **Minor** | Change type hints from `SetFitModel` to `RouterModel` (no logic change) |
| `sentimentizer/router/__init__.py` | **Minor** | Remove `import sentimentizer.compat`; update docstring |
| `sentimentizer/router/config.py` | **Minor** | Rename `SetFitConfig` → `RouterConfig` (keep `SetFitConfig` as backward-compat alias); update docstring |
| `sentimentizer/router/config.yaml` | **Minor** | Update comments (remove "SetFit" references) |
| `sentimentizer/compat.py` | **DELETE** | No longer needed without setfit |
| `sentimentizer/predictor.py` | **Moderate** | Replace setfit imports with `RouterModel`; remove compat import; update `_load_router_model()` |
| `sentimentizer/serve/app.py` | **Minor** | Remove `import sentimentizer.compat` |
| `workflows/cli.py` | **Moderate** | Replace setfit imports in `router_evaluate` and `router_push`; remove compat imports; update docstrings |
| `pyproject.toml` | **Minor** | Replace `setfit>=1.1.0` with `sentence-transformers>=5.0.0` and explicitly add `joblib` in `[router]` optional deps |
| `tests/test_router.py` | **Major** | Replace `SETHOOK_AVAILABLE` → `ROUTER_AVAILABLE` checks; add comprehensive `RouterModel` tests |
| `AGENTS.md` | **Moderate** | Update all SetFit references to new RouterModel architecture |

---

## Implementation Steps (Ordered)

### Phase 1: Core Replacement

1. **Create `RouterModel`** in `sentimentizer/router/model.py` — the core replacement class with `from_pretrained()`, `save_pretrained()`, `push_to_hub()`, `model_encode()`, `predict()`, `predict_proba()`, and `_migrate_legacy_setfit_model()`

2. **Rewrite `train_router.py`** — Remove setfit imports; implement `generate_contrastive_pairs()` (ported from SetFit's `ContrastiveDataset`); use `SentenceTransformer` + `CosineSimilarityLoss` for fine-tuning; fit `LogisticRegression` head on embeddings; return `RouterModel`

3. **Update `evaluate.py`** — Change type hints from `SetFitModel` to `RouterModel` (no logic change — same `model_encode`/`predict`/`predict_proba` API)

4. **Update `__init__.py`** — Remove `import sentimentizer.compat`; update docstring

5. **Update `config.py`** — Rename `SetFitConfig` → `RouterConfig`; add `SetFitConfig = RouterConfig` alias for backward compat; update docstring

6. **Update `predictor.py`** — Replace setfit imports with `RouterModel`; remove compat import; update `_load_router_model()` to use `RouterModel.from_pretrained()`

7. **Update `cli.py`** — Replace setfit imports in `router_evaluate` and `router_push` with `RouterModel`; remove compat imports; update docstrings

8. **Update `serve/app.py`** — Remove `import sentimentizer.compat`

### Phase 2: Cleanup

9. **Delete `compat.py`** — No longer needed without setfit

10. **Update `pyproject.toml`** — Replace `setfit>=1.1.0` with `sentence-transformers>=5.0.0` in `[project.optional-dependencies] router`; explicitly add `joblib` (for head serialization) to avoid relying on sklearn's transitive dependencies.

11. **Update `config.yaml`** — Replace "SetFit" references in comments with "Router"

### Phase 3: Tests

12. **Write tests** (see detailed test plan below)

### Phase 4: Documentation

13. **Update `AGENTS.md`** — Replace SetFit references with RouterModel; update architecture quick reference; update router module description; remove compat shim documentation; update dependency info

---

## Detailed Test Plan

### P0 — Must-Pass Before Merge (13 tests)

#### `RouterModel` Core API (6 tests)

| Test | What it proves |
|---|---|
| `test_router_model_from_pretrained` | `RouterModel.from_pretrained("BAAI/bge-base-en-v1.5")` loads backbone + creates head |
| `test_router_model_model_encode` | `model.model_encode(["hello", "world"])` returns `(2, 768)` ndarray |
| `test_router_model_predict` | `model.predict(["text"])` returns list of int/string labels |
| `test_router_model_predict_proba` | `model.predict_proba(["text"])` returns `(1, 3)` ndarray with rows summing to 1.0 |
| `test_router_model_predict_returns_valid_category` | Predictions are in `{0, 1, 2}` or `{"dietary", "service", "general"}` |
| `test_router_model_predict_proba_shape` | Output shape is `(N, 3)` for N texts, values in `[0, 1]`, rows sum to ~1.0 |

#### Save/Load Round-Trip (4 tests)

| Test | What it proves |
|---|---|
| `test_save_load_roundtrip` | `model.save_pretrained(dir)` → `RouterModel.from_pretrained(dir)` → `predict()` gives identical results |
| `test_save_creates_expected_files` | After `save_pretrained()`, directory contains `router_head.joblib`, `router_config.json`, and backbone files |
| `test_router_config_json_contents` | `router_config.json` has `num_classes`, `label_names`, `model_type` fields |
| `test_load_without_head` | If `router_head.joblib` is missing, `from_pretrained()` raises clear error |

#### Contrastive Pair Generation (6 tests)

| Test | What it proves |
|---|---|
| `test_generate_pairs_correct_count` | For N examples with `num_iterations=K`, produces ~`N * K * 2` pairs |
| `test_pairs_same_class_label_1` | All same-class pairs have `label=1.0` |
| `test_pairs_different_class_label_0` | All different-class pairs have `label=0.0` |
| `test_pairs_no_self_pairing` | A text is never paired with itself |
| `test_pairs_balanced_across_classes` | Each class gets roughly equal pair representation |
| `test_pairs_deterministic_with_seed` | Same seed produces identical pair datasets |

#### Graceful Degradation (1 test)

| Test | What it proves |
|---|---|
| `test_router_module_imports_without_sentence_transformers` | `import sentimentizer.router` succeeds even if `sentence-transformers` is missing |

### P1 — Should-Pass Before Merge (7 tests)

#### Training Pipeline (4 tests)

| Test | What it proves |
|---|---|
| `test_train_router_returns_router_model` | `train_router(config, train_ds)` returns a `RouterModel` instance |
| `test_train_router_saves_to_disk` | After training, `config.output_dir` contains model files |
| `test_trained_model_improves_over_random` | Trained model beats 33% random baseline on training data |
| `test_train_router_with_eval_dataset` | Training with `eval_dataset` completes without error |

#### Predictor Integration (3 tests)

| Test | What it proves |
|---|---|
| `test_predictor_classify_batch` | `SentimentPredictor.classify_batch()` returns correct format |
| `test_predictor_classify_batch_score_range` | All scores are in `[0, 1]` range |
| `test_predictor_load_router_from_disk` | `SentimentPredictor(router_model_path=...)` loads `RouterModel` from disk |

### P2 — Nice-to-Have / CI-Only (6 tests)

#### Legacy SetFit Migration (3 tests)

| Test | What it proves |
|---|---|
| `test_migrate_legacy_setfit_model` | `_migrate_legacy_setfit_model()` loads backbone + head from SetFit directory |
| `test_migrate_missing_setfit_config` | Directory without `config_setfit.json` → returns `None` |
| `test_migrated_model_predict_matches` | Migrated model's `predict()` matches original SetFit output |

#### CLI Integration (3 tests)

| Test | What it proves |
|---|---|
| `test_cli_router_train` | `sentimentizer router train --data ...` exits 0 |
| `test_cli_router_evaluate` | `sentimentizer router evaluate --model-path ... --data ...` exits 0 |
| `test_cli_router_push` | `sentimentizer router push` calls `push_to_hub` (mocked) |

#### Evaluate Module Compatibility (3 tests)

| Test | What it proves |
|---|---|
| `test_evaluate_router_with_new_model` | `evaluate_router(RouterModel, eval_ds)` returns expected dict |
| `test_compute_similarity_matrix` | Returns `(3, 3)` ndarray, diagonal near 1.0 |
| `test_calibrate_threshold` | Returns float in `(0, 1)` range |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Contrastive pair generation differs from SetFit | Model quality regression | Port SetFit's exact `ContrastiveDataset` logic (~50 lines of straightforward Python) |
| Saved model format incompatibility | Existing `models/router/` can't be loaded | Add `_migrate_legacy_setfit_model()` + document "retrain after upgrade" |
| Deprecation Warnings in `sentence-transformers` 5.x | Imports like `sentence_transformers.losses` throw warnings | Use the non-deprecated import path `sentence_transformers.sentence_transformer.losses` |
| `scikit-learn` version mismatch during deserialization | `joblib.load` can crash if `scikit-learn` version changes between training and inference | Consider pinning `scikit-learn` tighter or adding a version check in `RouterModel.from_pretrained()` |
| `sentence-transformers` `fit()` doesn't support Ray Train callbacks | Can't do distributed training | Use lower-level training loop for Ray path; `fit()` for standalone |
| `joblib` missing from dependencies | Head serialization crashes with `ModuleNotFoundError` if sklearn removes it as transitive dep | Explicitly add `joblib` to `[router]` extras in `pyproject.toml` |
| `push_to_hub` metadata differs from SetFit's | Downstream services expecting `config_setfit.json` break | Update downstream consumers to expect `router_config.json` and the `RouterModel` format |

---

## Why `sentence-transformers` Won't Break Things

1. **Already installed**: `sentence-transformers` is currently in the environment as a transitive dependency of `setfit`. The migration makes an implicit dependency explicit — no new packages enter the environment, total install size stays the same or shrinks.

2. **`transformers` version is irrelevant**: No code in the project imports `transformers` directly. The `compat.py` shim only existed for setfit's benefit. Removing setfit + compat means `transformers` 5.x compatibility is no longer a concern.

3. **Could we avoid `sentence-transformers` entirely?**: Yes, by reimplementing ~2K lines of well-tested code (model loading with pooling, smart batching, contrastive training loop, encode with normalization). Not worth it — the whole point is removing the wrapper (`setfit`) while keeping the underlying library.

---

## Ray Train Integration (Future — Blocked by This Migration)

**Current state**: SetFit's `Trainer` is a black box that owns the entire training loop. Can't insert Ray Train's distributed coordination. Router only trains standalone.

**After migration**: Training becomes three explicit, composable steps that map directly to Ray Train's `_train_func` pattern:

```python
def _router_train_func(ray_config):
    from ray import train
    
    backbone = SentenceTransformer(ray_config["base_model"])
    train_shard = train.get_dataset_shard("train")
    
    # Distributed fine-tuning with DDP
    backbone.fit(train_objectives=[(train_shard, CosineSimilarityLoss())], ...)
    
    # Rank 0: fit head, save, report metrics
    if train.get_context().get_world_rank() == 0:
        embeddings = backbone.encode(all_train_texts)
        head = LogisticRegression().fit(embeddings, labels)
        model = RouterModel(backbone=backbone, head=head, labels=labels)
        model.save_pretrained(ray_config["output_dir"])
        train.report(metrics={"accuracy": ...}, checkpoint=Checkpoint.from_directory(...))
```

**Key benefits**:
- Full training loop control — Can wrap in `_train_func()`, add `session.report()`, checkpointing
- Data-parallel backbone fine-tuning — The GPU-intensive step can scale across workers
- Unified pattern — Same `_train_func` → `Trainer` → `Checkpoint` flow as sentiment models
- Pipeline stage — Can add `stages/router_train.py` alongside `stages/train.py`

This should be a **follow-up task** after the migration is complete and stable.

---

## Naming Changes

| Old Name | New Name | Notes |
|---|---|---|
| `SetFitConfig` | `RouterConfig` | `SetFitConfig` kept as alias for backward compat |
| `SetFitModel` | `RouterModel` | Complete replacement, new file |
| `SetFitConfig` in tests | `RouterConfig` | Update test references |
| `SETHOOK_AVAILABLE` | `ROUTER_AVAILABLE` | Update availability check |
| `skip_without_setfit` | `skip_without_router` | Update pytest marker |
| `sentimentizer/compat.py` | (deleted) | No replacement needed |
| "SetFit router" in docs | "Router" or "Embedding router" | Throughout AGENTS.md, docstrings, comments |
</task_progress>