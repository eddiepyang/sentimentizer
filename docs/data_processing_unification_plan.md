# Data Processing Unification Plan

> **Status: NOT YET IMPLEMENTED** — This is a future plan. The codebase currently has `data_source.py` with `PandasDataSource` and `RayDataSource` classes, but the full `DataSource` protocol described here has not been implemented. Both `_balance_dataframe()` and `_balance_ray_dataset()` now support 3-class targets (0, 1, 2).

## Problem

The codebase has **five** categories of duplicated data processing logic, following the same single-node vs distributed pattern as the training loops:

| Category | Single-node (pandas) | Distributed (Ray) | Location |
|----------|----------------------|-------------------|----------|
| Dictionary building | `new_dictionary(df, cfg)` | `_build_dictionary_distributed(ds, cfg)` | `tokenizer.py` |
| Dataset transformation | `transform_dataframe(df)` | `transform_dataset(ds)` | `tokenizer.py` |
| Class balancing | `_balance_dataframe(df, ...)` | `_balance_ray_dataset(ds, ...)` | `loader.py` |
| Data loading | `load_train_val_corpus_datasets(...)` | `load_train_val_ray_datasets(...)` | `loader.py` |
| Tokenization stage | `run_tokenize` (no-Ray path) | `run_tokenize` (Ray path) | `workflows/stages/tokenize.py` |

The `run_tokenize()` function alone has **four** code paths:
- `new` + Ray available
- `new` + no Ray
- `update/resume` + Ray available
- `update/resume` + no Ray

All of these pairs implement the same logic using different APIs (pandas vs Ray Data).

## What's Already Shared

| Asset | Location | Status |
|---|---|---|
| `text_sequencer()` | `tokenizer.py` | ✅ Shared primitive |
| `vectorized_convert_ratings()` | `tokenizer.py` | ✅ Shared primitive |
| `regex_tokenize()` | `tokenizer.py` | ✅ Shared primitive |
| `compute_class_weights()` | `loader.py` | ✅ Shared utility (3-class class weights with smoothing) |
| `_count_vocab_batch()` | `tokenizer.py` | ✅ Shared primitive (used by both paths) |
| `CorpusDataset` | `loader.py` | ✅ Single-node dataset |

## What's Actually Duplicated

1. **Dictionary building logic** — word frequency counting, doc frequency counting, ID assignment, `filter_extremes()`, `compactify()` — 2×
2. **Dataset transformation** — drop neutral reviews, `text_sequencer()`, `vectorized_convert_ratings()`, write parquet — 2×
3. **Class balancing** — count classes, undersample majority, shuffle — 2×
4. **Train/val split** — `train_test_split` vs `ds.train_test_split()` — 2×
5. **Parquet I/O** — `pd.read_parquet`/`df.to_parquet` vs `ray.data.read_parquet`/`ds.write_parquet` — 2×
6. **Tokenization stage orchestration** — 4× (due to new/update × Ray/no-Ray)

## What's Different (Path-Specific)

| Concern | Single-node | Distributed |
|---------|-------------|-------------|
| Data representation | `pd.DataFrame` | `ray.data.Dataset` |
| Dictionary building | In-memory loop over rows | Map-Reduce via `_count_vocab_batch()` |
| Filtering rows | `df[df[col] != 3]` | `ds.filter(lambda row: ...)` |
| Transforming rows | `df[col].map(lambda ...)` | `ds.map_batches(fn, batch_format="numpy")` |
| Balancing | `sample(n=...)` + `pd.concat` | `random_sample()` + `union()` |
| Train/val split | `sklearn.train_test_split` | `ds.train_test_split()` |
| Parquet read | `pd.read_parquet` | `ray.data.read_parquet` |
| Parquet write | `df.to_parquet` | `ds.write_parquet` |
| Batch iteration | `for row in df.itertuples()` | `for batch in ds.iter_batches()` |

## Proposed Architecture

```
sentimentizer/data_source.py (NEW)
  ├── DataSource              # Protocol abstracting pandas vs Ray
  │     filter(predicate) → DataSource
  │     map_batches(fn, batch_size) → DataSource
  │     train_test_split(test_size) → tuple[DataSource, DataSource]
  │     balance(target_col) → DataSource
  │     read_parquet(path) → DataSource      (classmethod)
  │     write_parquet(path) → None
  │     to_pandas() → pd.DataFrame
  │     to_ray() → ray.data.Dataset
  │
  ├── PandasDataSource        # wraps pd.DataFrame
  └── RayDataSource           # wraps ray.data.Dataset

sentimentizer/tokenizer.py
  ├── Tokenizer
  │   ├── build_dictionary(data_source) → corpora.Dictionary  # replaces new_dictionary + _build_dictionary_distributed
  │   ├── transform(data_source) → DataSource                 # replaces transform_dataframe + transform_dataset
  │   └── update_dictionary(data_source) → None               # replaces update_from_dataset + _update_dictionary_distributed
  │
  ├── (remove) new_dictionary()              → inlined into build_dictionary()
  ├── (remove) _build_dictionary_distributed() → inlined into build_dictionary()
  ├── (remove) transform_dataframe()           → inlined into transform()
  └── (remove) transform_dataset()            → inlined into transform()

sentimentizer/loader.py
  ├── load_train_val_datasets(data_source, ...) → tuple[DataSource, DataSource]  # replaces both corpus/ray variants
  ├── (remove) load_train_val_corpus_datasets()  → inlined into load_train_val_datasets()
  ├── (remove) load_train_val_ray_datasets()       → inlined into load_train_val_datasets()
  ├── (remove) _balance_dataframe()              → inlined into DataSource.balance()
  └── (remove) _balance_ray_dataset()            → inlined into DataSource.balance()

workflows/stages/tokenize.py
  └── run_tokenize()
      └── single code path: data_source = DataSource.read_parquet(path)
                            tokenizer = Tokenizer.build_dictionary(data_source)
                            processed = tokenizer.transform(data_source)
                            processed.write_parquet(out_path)
```

## New / Revised Components

### `DataSource` Protocol

```python
from typing import Protocol, Self, runtime_checkable

@runtime_checkable
class DataSource(Protocol):
    """Unified interface for pandas DataFrame and Ray Dataset operations."""

    @classmethod
    def read_parquet(cls, path: str) -> Self: ...

    def filter(self, predicate: Callable[[dict], bool]) -> Self: ...

    def map_batches(
        self,
        fn: Callable[[dict], dict],
        batch_size: int = 4096,
        batch_format: str = "numpy",
    ) -> Self: ...

    def train_test_split(
        self,
        test_size: float,
        random_state: int = 42,
    ) -> tuple[Self, Self]: ...

    def balance(self, target_col: str, random_state: int = 42) -> Self: ...

    def write_parquet(self, path: str) -> None: ...

    def iter_batches(self, batch_size: int, batch_format: str = "numpy"): ...

    def count(self) -> int: ...

    def to_pandas(self) -> pd.DataFrame: ...

    def to_ray(self) -> ray.data.Dataset: ...

    @property
    def is_ray(self) -> bool: ...
```

### `PandasDataSource`

```python
class PandasDataSource:
    """DataSource backed by a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @classmethod
    def read_parquet(cls, path: str) -> Self:
        return cls(pd.read_parquet(path))

    def filter(self, predicate: Callable[[dict], bool]) -> Self:
        # Convert row-oriented predicate to pandas mask
        mask = self._df.apply(lambda row: predicate(row.to_dict()), axis=1)
        return PandasDataSource(self._df[mask].copy())

    def map_batches(
        self, 
        fn: Callable[[dict], dict], 
        batch_size: int = 4096, 
        batch_format: str = "numpy"
    ) -> Self:
        # Process in batches to avoid OOM for large DataFrames
        results = []
        for i in range(0, len(self._df), batch_size):
            batch_df = self._df.iloc[i : i + batch_size]
            result_batch = fn(batch_df.to_dict("list"))
            results.append(pd.DataFrame(result_batch))
        if not results:
            return PandasDataSource(pd.DataFrame())
        return PandasDataSource(pd.concat(results, ignore_index=True))

    def train_test_split(self, test_size: float, random_state: int = 42) -> tuple[Self, Self]:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            self._df, test_size=test_size, random_state=random_state
        )
        return PandasDataSource(train_df), PandasDataSource(val_df)

    def balance(self, target_col: str, random_state: int = 42) -> Self:
        """Balance classes via undersampling the majority class(es).

        Works with both binary (2-class) and multi-class (3-class) targets.
        Undersamples to the smallest class size.
        """
        class_counts = self._df[target_col].value_counts()
        min_count = class_counts.min()
        sampled = [
            self._df[self._df[target_col] == cls].sample(n=min_count, random_state=random_state)
            for cls in class_counts.index
        ]
        balanced = pd.concat(sampled).sample(frac=1, random_state=random_state)
        return PandasDataSource(balanced)

    def write_parquet(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(path, engine="pyarrow")

    def iter_batches(self, batch_size: int, batch_format: str = "numpy"):
        for i in range(0, len(self._df), batch_size):
            yield self._df.iloc[i : i + batch_size].to_dict("list")

    def count(self) -> int:
        return len(self._df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def to_ray(self) -> ray.data.Dataset:
        import ray
        return ray.data.from_pandas(self._df)

    @property
    def is_ray(self) -> bool:
        return False
```

### `RayDataSource`

```python
class RayDataSource:
    """DataSource backed by a Ray Dataset."""

    def __init__(self, ds: ray.data.Dataset) -> None:
        self._ds = ds

    @classmethod
    def read_parquet(cls, path: str) -> Self:
        return cls(ray.data.read_parquet(path))

    def filter(self, predicate: Callable[[dict], bool]) -> Self:
        return RayDataSource(self._ds.filter(predicate))

    def map_batches(
        self, 
        fn: Callable[[dict], dict], 
        batch_size: int = 4096, 
        batch_format: str = "numpy"
    ) -> Self:
        return RayDataSource(
            self._ds.map_batches(fn, batch_size=batch_size, batch_format=batch_format)
        )

    def train_test_split(self, test_size: float, random_state: int = 42) -> tuple[Self, Self]:
        train_ds, val_ds = self._ds.train_test_split(
            test_size=test_size, shuffle=True, seed=random_state
        )
        return RayDataSource(train_ds), RayDataSource(val_ds)

    def balance(self, target_col: str, random_state: int = 42) -> Self:
        """Balance classes via undersampling the majority class(es).

        Works with both binary (2-class) and multi-class (3-class) targets.
        Undersamples to the smallest class size.
        """
        class_labels = self._ds.unique(column=target_col)
        class_counts = {label: self._ds.filter(lambda row: row[target_col] == label).count() for label in class_labels}
        min_count = min(class_counts.values())
        if min_count == 0:
            return self
        sampled = [
            self._ds.filter(lambda row: row[target_col] == label).random_sample(min_count / class_counts[label], seed=random_state)
            for label in class_labels
        ]
        balanced = sampled[0]
        for ds in sampled[1:]:
            balanced = balanced.union(ds)
        return RayDataSource(balanced.random_shuffle(seed=random_state))

    def write_parquet(self, path: str) -> None:
        self._ds.write_parquet(path)

    def iter_batches(self, batch_size: int, batch_format: str = "numpy"):
        return self._ds.iter_batches(batch_size=batch_size, batch_format=batch_format)

    def count(self) -> int:
        return self._ds.count()

    def to_pandas(self) -> pd.DataFrame:
        return self._ds.to_pandas()

    def to_ray(self) -> ray.data.Dataset:
        return self._ds

    @property
    def is_ray(self) -> bool:
        return True
```

### Unified `Tokenizer.build_dictionary()`

```python
class Tokenizer:
    ...

    @classmethod
    def build_dictionary(
        cls,
        data_source: DataSource,
        cfg: TokenizerConfig | None = None,
    ) -> Tokenizer:
        """Build dictionary from any DataSource (pandas or Ray)."""
        if cfg is None:
            cfg = TokenizerConfig(save_dictionary=True)

        # Unified vocab counting via DataSource.iter_batches()
        total_word_freq: Counter = Counter()
        total_doc_freq: Counter = Counter()
        total_num_docs = 0

        for batch in data_source.iter_batches(batch_size=10000, batch_format="numpy"):
            wf, df, nd = _count_vocab_batch(batch, cfg.text_col)
            total_word_freq += wf
            total_doc_freq += df
            total_num_docs += nd

        # Build Gensim dictionary (same reduce phase for both paths)
        dictionary = corpora.Dictionary()
        dictionary.num_docs = total_num_docs
        for idx, word in enumerate(
            sorted(total_word_freq.keys(), key=lambda w: (-total_word_freq[w], w))
        ):
            dictionary.token2id[word] = idx
            dictionary.dfs[idx] = total_doc_freq[word]
        dictionary.num_pos = sum(total_word_freq.values())
        dictionary.num_nnz = sum(total_doc_freq.values())
        dictionary.filter_extremes(
            no_below=cfg.dict_min,
            no_above=cfg.no_above,
            keep_n=cfg.dict_keep,
        )
        dictionary.compactify()

        if cfg.save_dictionary:
            dictionary.save(str(FileConfig.dictionary_file_path))

        return cls(dictionary=dictionary, cfg=cfg)
```

### Unified `Tokenizer.transform()`

```python
    def transform(self, data_source: DataSource) -> DataSource:
        """Transform any DataSource: drop neutral, tokenize, convert ratings."""
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")

        dictionary = self.dictionary
        cfg = self.cfg

        # Step 1: Drop neutral (3-star) reviews
        filtered = data_source.filter(lambda row: row[cfg.label_col] != 3)

        # Step 2: Map batches to convert text to sequences
        def transform_batch(batch: dict) -> dict:
            inputs = []
            for text in batch[cfg.text_col]:
                inputs.append(text_sequencer(dictionary, text, cfg.max_len))
            batch[cfg.inputs] = np.array(inputs)
            if cfg.label_col in batch:
                batch[cfg.labels] = vectorized_convert_ratings(
                    np.asarray(batch[cfg.label_col])
                )
            # Drop variable-length columns
            cols_to_keep = {cfg.inputs, cfg.labels}
            for col in list(batch.keys()):
                if col not in cols_to_keep:
                    del batch[col]
            return batch

        return filtered.map_batches(transform_batch, batch_size=10000, batch_format="numpy")
```

### Unified `load_train_val_datasets()`

```python
def load_train_val_datasets(
    data_path: str,
    use_ray: bool = False,
    test_size: float = 0.2,
    balance_classes: bool = False,
    random_state: int = 42,
) -> tuple[DataSource, DataSource]:
    """Load processed parquet as DataSource, split into train/val.

    Args:
        data_path: Path to processed parquet file.
        use_ray: If True, returns RayDataSource; otherwise PandasDataSource.
        test_size: Fraction for validation set.
        balance_classes: Whether to undersample majority class in training set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_data_source, val_data_source).
    """
    data_source = (
        RayDataSource.read_parquet(data_path)
        if use_ray
        else PandasDataSource.read_parquet(data_path)
    )
    train_ds, val_ds = data_source.train_test_split(test_size=test_size, random_state=random_state)

    if balance_classes:
        train_ds = train_ds.balance(target_col="target", random_state=random_state)

    return train_ds, val_ds
```

## Implementation Phases

### Phase 1: DataSource Protocol (~1.5 hours)

1. **Create `sentimentizer/data_source.py`** with `DataSource` protocol + `PandasDataSource` + `RayDataSource`
2. **Add tests** for both implementations (read/write parquet, filter, map_batches, train_test_split, balance)
3. **Add factory** `DataSource.from_path(path, use_ray=False)` for convenience

### Phase 2: Unified Dictionary Building (~1 hour)

1. **Replace `new_dictionary()` and `_build_dictionary_distributed()`** with `Tokenizer.build_dictionary(data_source)`
2. **Update `Tokenizer.from_data()`** and `Tokenizer.from_dataset()` to call `build_dictionary()`
3. **Update `_train_func()`** to use `create_model_from_registry()` (already done in training loop unification)

### Phase 3: Unified Dataset Transformation (~1 hour)

1. **Replace `transform_dataframe()` and `transform_dataset()`** with single `Tokenizer.transform(data_source)`
2. **Update tests** in `test_dictionary_lifecycle.py`
3. **Update `run_tokenize()`** to use `transform()` instead of branch logic

### Phase 4: Unified Data Loading (~1 hour)

1. **Replace `load_train_val_corpus_datasets()` and `load_train_val_ray_datasets()`** with `load_train_val_datasets()`
2. **Unwrap the DataSources**: Update callers (`Trainer.fit()`, `_train_func()`, `tuner.py`, `agent/diagnose_model.py`, `workflows/stages/train.py`) to convert the returned `DataSource` to the required format. Call `.to_pandas()` and wrap in `CorpusDataset(...)` for single-node paths, or call `.to_ray()` for distributed paths.

### Phase 5: Collapse `run_tokenize()` (~1 hour)

1. **Remove 4-path branching** in `run_tokenize()`
2. **Single path**: `data_source = DataSource.read_parquet(raw_path)` → `tokenizer = Tokenizer.build_dictionary(data_source)` → `processed = tokenizer.transform(data_source)` → `processed.write_parquet(out_path)`
3. **Handle resume/update logic** uniformly via `Tokenizer.update_dictionary(data_source)`

### Phase 6: Testing (~1.5 hours)

1. **Run full test suite**: `pytest tests/ -v`
2. **Integration test**: End-to-end tokenize → train pipeline with both pandas and Ray
3. **Verify no regressions**: Check that existing parquet files are byte-identical

## Benefits

- **~200 lines removed** (4 duplicate function pairs + 2 branching paths in `run_tokenize()`)
- **Single source of truth** for data processing — bug fixes apply everywhere
- **Easier testing** — mock `DataSource` for unit tests, no need for real Ray clusters
- **Simpler onboarding** — one `Tokenizer` API regardless of execution mode
- **Natural pairing** with `_iter_batches()` from training loop unification — data flows through a unified `DataSource` from loading through training

## Detailed Risk Analysis

### RISK 1: `map_batches` API Mismatch

**Severity: MEDIUM**

Ray `map_batches` receives dict-of-arrays; pandas `map_batches` receives dict-of-lists. The `transform_batch` function must work with both.

**Mitigation**: Both implementations convert to dict-of-lists before calling `fn`. Test with `batch_format="numpy"` for Ray and `"list"` for pandas.

### RISK 2: `filter` Predicate Signature

**Severity: LOW**

Ray `filter` receives `dict` (row-oriented); pandas `apply` receives `Series`. The predicate must be row-dict compatible.

**Mitigation**: Both implementations convert to `dict` before calling predicate. Document that predicates must accept `dict` and return `bool`.

### RISK 3: Dictionary Update Preserves IDs

**Severity: HIGH**

`update_from_dataset()` skips `filter_extremes` and `compactify` to preserve existing IDs. The unified `update_dictionary()` must maintain this behavior.

**Mitigation**: Separate `build_dictionary()` (full pipeline with filter/compactify) from `update_dictionary()` (add-only, no ID reassignment). Document the distinction clearly.

### RISK 4: Parquet Column Types

**Severity: MEDIUM**

Ray and pandas may write parquet with different column types (e.g., list columns). The `processed` parquet must be readable by both.

**Mitigation**: The `transform_batch` function already drops variable-length columns and keeps only numeric arrays. This should be consistent.

### RISK 5: Ray Dataset Laziness

**Severity: LOW**

Ray Datasets are lazy — `count()`, `write_parquet()` trigger execution. Pandas is eager. Callers must be aware.

**Mitigation**: `DataSource` abstracts this — `count()` and `write_parquet()` are always eager from the caller's perspective.

### RISK 6: `is_ray_available` Checks

**Severity: LOW**

`workflows/lifecycle.py` has `is_ray_available()` which checks `ray.is_initialized()`. The unified pipeline must still support environments without Ray.

**Mitigation**: `DataSource` factory uses `use_ray` parameter; callers check `is_ray_available()` once and pass the flag. No Ray imports at module level.

### RISK 7: `PandasDataSource.map_batches` Memory Bloat

**Severity: HIGH**

The original plan executed the map function on the entire DataFrame at once. For large datasets, converting the whole DataFrame to a dict-of-lists and processing it entirely in memory causes massive OOM crashes.

**Mitigation**: `PandasDataSource.map_batches` must accept a `batch_size` parameter and process the DataFrame iteratively in chunks via `iloc[i:i+batch_size]`. Collect results as dicts (not intermediate DataFrames) and build a single final DataFrame to avoid `pd.concat` overhead.

### RISK 8: Return Type Mismatch in `load_train_val_datasets`

**Severity: HIGH**

`load_train_val_datasets()` returns `DataSource` wrappers, but callers strictly expect PyTorch `CorpusDataset` or `ray.data.Dataset` objects to create their DataLoaders and shards.

**Mitigation**: Explicitly unwrap the data sources in the callers (`.to_pandas()` then `CorpusDataset(...)`, or `.to_ray()`) during Phase 4 implementation.

### RISK 9: Pandas Row-Wise Filter Performance

**Severity: HIGH**

Using `df.apply(lambda row: predicate(row.to_dict()), axis=1)` is catastrophic compared to the existing vectorized filtering (`df[df[col] != 3]`). `apply(axis=1)` iterates in pure Python, converts every row to a `dict`, and calls the predicate function per row. For a 1M-row Yelp dataset this is easily **100–1000× slower** than the vectorized C-level operation it replaces (milliseconds → minutes).

This affects the tokenization stage's neutral-review drop, which is the first operation on the full raw dataset.

**Mitigation**: 
- Add an **expression-based filter overload** to `DataSource` for common cases: `filter(self, col: str, op: str, value)` which uses vectorized pandas operations (`df[df[col] != value]`) internally for `PandasDataSource` while falling back to row-dict predicates for `RayDataSource`.
- Keep the row-dict `filter(predicate)` for complex/uniform logic, but document that it is **only suitable for small datasets (<100K rows)** on the pandas path.
- In `Tokenizer.transform()`, use the expression-based filter for the neutral-drop step instead of the generic predicate.

### RISK 10: Gensim Dictionary Serialization in Ray Closure

**Severity: MEDIUM**

In `Tokenizer.transform()`, the inner `transform_batch` function captures `self.dictionary` in its closure. Ray will serialize and broadcast the entire Gensim dictionary to all workers.

**Mitigation**: This is a known risk with map-reduce style data processing. If the Gensim dictionary grows too large and exceeds Ray's default task size limits (10MB-100MB depending on config), it will need to be explicitly placed in the Ray Object Store via `ray.put()` and passed by reference.

### RISK 11: `balance()` Eager Execution Cost on Ray

**Severity: MEDIUM**

Ray `balance()` performs `filter()` → `count()` → `random_sample()` → `union()` → `random_shuffle()`. Each `count()` triggers a full dataset scan. Two counts (one per class) plus the final shuffle means **three complete passes** over the data. On large datasets this is expensive.

On the pandas path, `balance()` is eager and performs a single pass over the DataFrame.

**Mitigation**: 
- Document that `balance()` is expensive on Ray and should be avoided for very large datasets.
- Consider adding a `balance_fraction` parameter to sample only a fraction of the data for quick class-balance checks during development.
- For production, consider using `compute_class_weights()` (loss weighting) instead of undersampling to avoid the data-pass cost entirely.

### RISK 12: `DataSource.iter_batches` Memory Overhead

**Severity: HIGH**

`PandasDataSource.iter_batches()` converts each batch to a dict-of-lists via `to_dict("list")`. For a 10K-row batch with a 200-token sequence column (int32), the DataFrame stores ~8MB of contiguous data, but the dict-of-lists representation scatters this into 10,000 Python `list` objects with per-element `int` objects (~28 bytes each). This is a **50–100× memory expansion** during batch iteration.

`RayDataSource.iter_batches()` with `batch_format="numpy"` returns NumPy arrays, which are memory-efficient.

**Mitigation**: 
- Add a `batch_format` parameter to `iter_batches()`: `"list"` yields dict-of-lists (matching pandas `to_dict("list")`), `"numpy"` yields dict-of-arrays.
- Use `"list"` for dictionary building to avoid numpy→list conversion overhead in `_count_vocab_batch()`.
- For numeric columns in `"numpy"` mode, convert via `np.asarray()` but keep ragged token columns as Python lists (detected via `isinstance(v[0], list)`).
- Ensure `_count_vocab_batch()` works with NumPy arrays (it already does via `list()` conversion) but prefer `"list"` format for the text column.

### RISK 13: `map_batches` Returning Non-Uniform Schemas

**Severity: HIGH**

Ray `map_batches` requires that every batch returned by the function has the **exact same schema** (same columns, same types). If `transform_batch` conditionally includes `cfg.labels` based on `cfg.label_col in batch`, a batch missing the label column will cause Ray to crash with a schema-mismatch error.

Currently this is safe because `cfg.label_col` is either present in all batches or absent, but the unified `DataSource` could receive data where some batches have the label and some don't (e.g., inference data mixed with training data).

**Mitigation**: 
- Always include the label column in the output schema, even if absent from the input batch, by filling with a default value (`np.zeros(len(batch[cfg.text_col]))`).
- Or, split `Tokenizer.transform()` into `transform_for_training()` (always includes labels) and `transform_for_inference()` (never includes labels).

### RISK 14: Ray Dataset Lazy `train_test_split` and `balance` Interaction

**Severity: MEDIUM**

Ray `train_test_split()` returns lazy Datasets, but `balance()` calls `count()` which triggers execution. If the caller does `train_ds, val_ds = ds.train_test_split(...)` then `train_ds.balance(...)`, the split is materialized twice: once during `train_test_split()`'s internal shuffle/count, and again during `balance()`'s class counts.

**Mitigation**: 
- Document the lazy/eager behavior in `DataSource` docstrings.
- Consider adding `.materialize()` calls after `train_test_split()` if the Ray path shows performance issues.
- Profile the tokenization stage with large datasets to quantify the materialization cost.

### RISK 15: `to_ray()` DataFrame Block Sizing

**Severity: LOW**

`PandasDataSource.to_ray()` calls `ray.data.from_pandas(self._df)`, which creates a Ray Dataset with a **single block** (the entire DataFrame). This defeats Ray's parallelism for downstream operations like `map_batches` and `filter`, because all computation runs on one worker.

**Mitigation**: 
- Use `ray.data.from_pandas_refs([ray.put(self._df)])` or partition the DataFrame before conversion.
- Document that `to_ray()` is intended for small-to-medium datasets; for large datasets, use `RayDataSource.read_parquet()` directly.
- Add a `num_blocks` parameter to `to_ray()` that splits the DataFrame before conversion.

### RISK 16: `CorpusDataset` Memory Retention After Unwrapping

**Severity: MEDIUM**

`load_train_val_datasets()` returns `DataSource` objects. Callers unwrap them via `.to_pandas()` then wrap in `CorpusDataset`. `CorpusDataset.__init__` converts the entire DataFrame columns to lists of tensors. The original DataFrame is not freed until garbage collection, so memory usage temporarily **doubles** (DataFrame + tensor lists).

**Mitigation**: 
- In `CorpusDataset.__init__`, accept the `DataSource` directly and call `.to_pandas()` internally, then `del df` after tensor conversion.
- Or, modify `CorpusDataset` to accept a `DataSource` and extract batches iteratively to avoid full materialization.

### RISK 17: Pandas `map_batches` Result Schema Drift

**Severity: MEDIUM**

`PandasDataSource.map_batches()` does `pd.DataFrame(result_batch)` for each batch, then `pd.concat()`. If `fn` returns dicts with different keys per batch (e.g., some batches missing a column), `pd.DataFrame()` will produce different column orders, and `pd.concat()` will introduce NaNs. This silently corrupts the dataset.

Ray `map_batches` would crash with a schema error, which is actually preferable.

**Mitigation**: 
- Add schema validation in `PandasDataSource.map_batches()`: assert that all result batches have the same keys before concatenating.
- Document that `fn` must return uniform keys across all batches.

### RISK 18: `pd.concat()` Quadratic Cost in `PandasDataSource.map_batches`

**Severity: MEDIUM**

The reference implementation shows `results.append(pd.DataFrame(result_batch))` inside a loop, followed by `pd.concat(results)`. For large datasets with many batches, repeatedly creating DataFrames and the final concat has **O(n²)** complexity due to repeated index allocation and data copying. A 1M-row dataset with `batch_size=1000` produces 1000 intermediate DataFrames; the final `pd.concat()` copies all data one more time. In practice this adds **2–5× wall-time overhead** versus processing the same data in a single pass.

**Mitigation**: Collect results as dict-of-lists (or list-of-dicts) during batch processing, then build a single DataFrame at the end with `pd.DataFrame(all_results)`. This eliminates intermediate DataFrame allocations and the expensive concat step. If result schemas are uniform, pre-allocate a list of the correct length and assign by slice.

### RISK 19: Ray `to_pandas()` Materialization OOM

**Severity: HIGH**

When `RayDataSource.to_pandas()` is called (e.g., in Phase 4 when unwrapping for `CorpusDataset`), it materializes the entire distributed dataset into a single pandas DataFrame on the driver node. For datasets larger than the driver's available RAM, this causes an immediate OOM kill. This is especially dangerous because it appears to work during development on small samples but fails in production.

**Mitigation**: 
- For distributed training paths, avoid `to_pandas()` entirely; keep data as `RayDataSource` and pass directly to Ray Train.
- For single-node paths that must use `CorpusDataset`, add a `max_rows` safety check or use chunked materialization via `ds.iter_batches()` + incremental DataFrame building.
- Document that `to_pandas()` is only safe for datasets that fit in driver memory.

### RISK 20: `_count_vocab_batch()` Numpy-to-List Conversion Overhead

**Severity: MEDIUM**

When `build_dictionary()` uses `iter_batches(batch_format="numpy")`, the text column arrives as a numpy `ndarray` of objects (each element is a Python list of tokens). `_count_vocab_batch()` calls `list(doc_tokens)` on each document to convert from a numpy scalar to a Python list. For a 10K-document batch with 50 tokens each, this is **500K Python object allocations per batch**—pure Python overhead that can dominate dictionary building time, especially when multiplied across hundreds of batches.

**Mitigation**: 
- Use `batch_format="list"` in `build_dictionary()` so the text column arrives as native Python lists, eliminating the numpy scalar wrapping entirely.
- Alternatively, optimize `_count_vocab_batch()` to iterate numpy object arrays directly without per-document `list()` calls (e.g., via `batch[text_col].tolist()` to convert the entire column at once).

## Risk Summary

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | `map_batches` API mismatch | MEDIUM | Both impls convert to dict-of-lists |
| 2 | `filter` predicate signature | LOW | Both impls convert to dict |
| 3 | Dictionary update preserves IDs | HIGH | Separate `build` vs `update` methods |
| 4 | Parquet column types | MEDIUM | Only keep numeric columns |
| 5 | Ray Dataset laziness | LOW | `DataSource` makes operations eager |
| 6 | Ray availability checks | LOW | Factory uses `use_ray` flag |
| 7 | `PandasDataSource.map_batches` memory bloat | HIGH | Batch the pandas processing internally |
| 8 | Return type mismatch | HIGH | Callers unwrap `DataSource` before use |
| 9 | Pandas row-wise filter performance | HIGH | Expression-based filter overload + document limits |
| 10 | Gensim dict serialization in Ray closure | MEDIUM | Use `ray.put()` if dict exceeds limits |
| 11 | `balance()` eager execution cost on Ray | MEDIUM | Document cost; use `compute_class_weights()` instead |
| 12 | `DataSource.iter_batches` memory overhead | HIGH | Yield dict-of-arrays; add `batch_format` param |
| 13 | `map_batches` non-uniform schemas | HIGH | Always output uniform schema; split train/infer paths |
| 14 | Ray lazy `train_test_split` + `balance` interaction | MEDIUM | Document lazy/eager; consider `.materialize()` |
 | 15 | `to_ray()` DataFrame block sizing | LOW | Partition before conversion; add `num_blocks` param |
 | 16 | `CorpusDataset` memory retention after unwrapping | MEDIUM | Accept `DataSource` directly; `del df` after conversion |
 | 17 | Pandas `map_batches` result schema drift | MEDIUM | Validate schema consistency before `pd.concat` |
 | 18 | `pd.concat()` quadratic cost in `map_batches` | MEDIUM | Collect dicts, build one DataFrame at end |
 | 19 | Ray `to_pandas()` materialization OOM | HIGH | Avoid `to_pandas()` in distributed paths; add safety checks |
 | 20 | `_count_vocab_batch()` numpy→list overhead | MEDIUM | Use `batch_format="list"` for dictionary building |

## Estimated Effort

| Phase | Hours | Description |
|-------|-------|-------------|
| Phase 1 | ~1.5 | DataSource protocol + implementations + tests |
| Phase 2 | ~1 | Unified dictionary building |
| Phase 3 | ~1 | Unified dataset transformation |
| Phase 4 | ~1 | Unified data loading and caller unwrapping |
| Phase 5 | ~1 | Collapse `run_tokenize()` |
| Phase 6 | ~1.5 | Testing & validation |
| **Total** | **~7.0** | |

## Recommended Implementation Order

Each sub-phase should be a separate commit/PR:

1. **Phase 1a**: Add `DataSource` protocol + `PandasDataSource` + tests
2. **Phase 1b**: Add `RayDataSource` + tests
3. **Phase 2**: Refactor `Tokenizer.build_dictionary()` + update `from_data`/`from_dataset`
4. **Phase 3**: Refactor `Tokenizer.transform()` + remove old transform methods
5. **Phase 4**: Refactor `load_train_val_datasets()` + update callers
6. **Phase 5**: Collapse `run_tokenize()` to single path
7. **Phase 6**: Full test suite + integration validation