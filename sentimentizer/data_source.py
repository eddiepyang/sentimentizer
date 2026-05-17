"""Unified data source abstraction for pandas DataFrame and Ray Dataset operations.

This module provides a protocol-based abstraction layer that lets the rest of the
codebase work with a single ``DataSource`` interface regardless of whether the
underlying data lives in a pandas DataFrame (single-node) or a Ray Dataset
(distributed).

Key design decisions that address plan risks:

- **Risk 9 mitigation**: ``filter`` has two signatures:
  - ``filter(predicate)`` – row-dict predicate (slow on pandas, use only for
    small datasets or uniform logic)
  - ``filter(col, op, value)`` – expression-based filter that uses vectorized
    pandas operations on the PandasDataSource path

- **Risk 12 mitigation**: ``iter_batches`` accepts a ``batch_format`` parameter.
  ``"numpy"`` (the default) yields dict-of-arrays and avoids the ~8× memory
  expansion caused by dict-of-lists.

- **Risk 15 mitigation**: ``to_ray()`` accepts ``override_num_blocks`` so that
  large DataFrames can be repartitioned instead of creating a single-block
  Ray Dataset.

- **Risk 17 mitigation**: ``PandasDataSource.map_batches`` validates that every
  result batch has the same keys before concatenating, raising a clear error
  instead of silently producing NaNs.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

if RAY_AVAILABLE:
    import ray.data


# ---------------------------------------------------------------------------
# Operator mapping for expression-based filters (Risk 9 mitigation)
# ---------------------------------------------------------------------------

_COMP_OPS: dict[str, Callable[[object, object], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DataSource(Protocol):
    """Unified interface for pandas DataFrame and Ray Dataset operations."""

    @classmethod
    def read_parquet(cls, path: str) -> Self:
        """Read a parquet file into a DataSource."""
        ...

    def filter(
        self,
        predicate: Callable[[dict], bool] | None = None,
        *,
        col: str | None = None,
        op: str | None = None,
        value: object = None,
    ) -> Self:
        """Filter rows.

        Two signatures are supported:

        1. ``filter(predicate)`` – row-dict predicate (uniform across both
           backends). On the pandas path this uses ``df.apply(axis=1)`` and is
           slow for large DataFrames (>100K rows).

        2. ``filter(col=..., op=..., value=...)`` – expression-based filter.
           On the pandas path this uses vectorized C-level operations
           (e.g. ``df[df[col] != value]``). On the Ray path it falls back to
           a lambda predicate.

        Args:
            predicate: Row-dict predicate ``row -> bool``.
            col: Column name for expression-based filter.
            op: Operator for expression-based filter
                (``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``, or symbolic
                forms like ``==``, ``!=``).
            value: Value to compare against.

        Returns:
            A new DataSource containing only rows that match the filter.
        """
        ...

    def map_batches(
        self,
        fn: Callable[[dict], dict],
        batch_size: int = 4096,
        batch_format: str = "numpy",
    ) -> Self:
        """Apply a function to batches of data.

        Args:
            fn: Function that receives a dict (keys are column names, values
                are arrays or lists depending on ``batch_format``) and returns
                a dict with the same keys for every batch.
            batch_size: Number of rows per batch.
            batch_format: ``"numpy"`` yields dict-of-arrays; ``"list"`` yields
                dict-of-lists. Default is ``"numpy"`` to avoid memory bloat.

        Returns:
            A new DataSource with the transformed batches.
        """
        ...

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[Self, Self]:
        """Split into train and validation sets."""
        ...

    def balance(self, target_col: str, random_state: int = 42) -> Self:
        """Undersample the majority class to match the minority class."""
        ...

    def write_parquet(self, path: str) -> None:
        """Write data to a parquet file."""
        ...

    def iter_batches(
        self,
        batch_size: int = 4096,
        batch_format: str = "numpy",
    ) -> Iterator[dict]:
        """Yield batches as dicts.

        Args:
            batch_size: Number of rows per batch.
            batch_format: ``"numpy"`` (default) yields dict-of-arrays.
                ``"list"`` yields dict-of-lists.
        """
        ...

    def count(self) -> int:
        """Return the number of rows."""
        ...

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        ...

    def to_ray(self, override_num_blocks: int | None = None) -> ray.data.Dataset:
        """Convert to Ray Dataset.

        Args:
            override_num_blocks: Number of blocks for the resulting Ray Dataset.
                If None, Ray chooses automatically. Use a higher value for
                large DataFrames to ensure parallelism.
        """
        ...

    @property
    def is_ray(self) -> bool:
        """True if this is a Ray-backed DataSource."""
        ...

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        ...


# ---------------------------------------------------------------------------
# PandasDataSource
# ---------------------------------------------------------------------------


class PandasDataSource:
    """DataSource backed by a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    # --- Factory -----------------------------------------------------------

    @classmethod
    def read_parquet(cls, path: str) -> Self:
        return cls(pd.read_parquet(path))

    # --- Filtering ---------------------------------------------------------

    def filter(
        self,
        predicate: Callable[[dict], bool] | None = None,
        *,
        col: str | None = None,
        op: str | None = None,
        value: object = None,
    ) -> Self:
        if predicate is not None and (col is not None or op is not None):
            raise ValueError(
                "Cannot specify both predicate and col/op/value. Use one filtering style at a time."
            )

        # Expression-based filter (Risk 9 mitigation: vectorized path)
        if col is not None and op is not None:
            comp_fn = _COMP_OPS.get(op)
            if comp_fn is None:
                raise ValueError(f"Unsupported operator: {op!r}")
            mask = comp_fn(self._df[col], value)
            return PandasDataSource(self._df[mask].copy())

        # Row-dict predicate (uniform but slow on pandas)
        if predicate is not None:
            mask = self._df.apply(lambda row: predicate(row.to_dict()), axis=1)
            return PandasDataSource(self._df[mask].copy())

        raise ValueError("Must provide either predicate or col/op/value.")

    # --- Map batches -------------------------------------------------------

    def _to_batch_dict(self, batch_df: pd.DataFrame, batch_format: str) -> dict:
        """Convert a DataFrame slice to a batch dict matching the requested format."""
        raw = batch_df.to_dict("list")
        if batch_format == "numpy":
            batch_dict: dict = {}
            for k, v in raw.items():
                # Skip ragged/nested data (e.g. token lists) — keep as Python lists
                if v and isinstance(v[0], list):
                    batch_dict[k] = v
                    continue
                try:
                    batch_dict[k] = np.asarray(v)
                except ValueError:
                    batch_dict[k] = v
            return batch_dict
        return raw

    def map_batches(
        self,
        fn: Callable[[dict], dict],
        batch_size: int = 4096,
        batch_format: str = "numpy",
    ) -> Self:
        # Risk 7 mitigation: process in chunks to avoid OOM
        # Risk 12 mitigation: use "numpy" format by default
        results: list[pd.DataFrame] = []
        expected_keys: set[str] | None = None

        for i in range(0, len(self._df), batch_size):
            batch_df = self._df.iloc[i : i + batch_size]
            batch_dict = self._to_batch_dict(batch_df, batch_format)

            result_batch = fn(batch_dict)

            # Risk 17 mitigation: schema validation
            if expected_keys is None:
                expected_keys = set(result_batch.keys())
            elif set(result_batch.keys()) != expected_keys:
                raise ValueError(
                    f"Schema mismatch in map_batches at batch {i // batch_size}: "
                    f"expected keys {expected_keys}, got {set(result_batch.keys())}"
                )

            # Convert arrays back to lists for DataFrame construction
            # (pd.DataFrame cannot ingest ragged ndarrays)
            df_data = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result_batch.items()
            }
            results.append(pd.DataFrame(df_data))

        if not results:
            return PandasDataSource(pd.DataFrame())
        return PandasDataSource(pd.concat(results, ignore_index=True))

    # --- Train / val split -------------------------------------------------

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[Self, Self]:
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            self._df,
            test_size=test_size,
            random_state=random_state,
        )
        return PandasDataSource(train_df), PandasDataSource(val_df)

    # --- Balancing ---------------------------------------------------------

    def balance(self, target_col: str, random_state: int = 42) -> Self:
        """Undersample majority classes to match the minority class count.

        Works with both binary (2-class) and multi-class (3-class) targets.
        """
        class_counts = self._df[target_col].value_counts()
        min_count = int(class_counts.min())

        if (class_counts == min_count).all():
            return self  # already balanced

        dfs = []
        for cls_val in class_counts.index:
            cls_df = self._df[self._df[target_col] == cls_val]
            if len(cls_df) > min_count:
                cls_df = cls_df.sample(n=min_count, random_state=random_state)
            dfs.append(cls_df)

        balanced = pd.concat(dfs).sample(frac=1, random_state=random_state)
        return PandasDataSource(balanced)

    # --- I/O ---------------------------------------------------------------

    def write_parquet(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(path, engine="pyarrow")

    # --- Iteration ---------------------------------------------------------

    def iter_batches(
        self,
        batch_size: int = 4096,
        batch_format: str = "numpy",
    ) -> Iterator[dict]:
        for i in range(0, len(self._df), batch_size):
            batch_df = self._df.iloc[i : i + batch_size]
            yield self._to_batch_dict(batch_df, batch_format)

    # --- Properties / conversion -------------------------------------------

    def count(self) -> int:
        return len(self._df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def to_ray(self, override_num_blocks: int | None = None) -> ray.data.Dataset:
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed.")
        # Risk 15 mitigation: allow override_num_blocks for large DataFrames
        return ray.data.from_pandas(
            self._df,
            override_num_blocks=override_num_blocks,
        )

    @property
    def is_ray(self) -> bool:
        return False

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)


# ---------------------------------------------------------------------------
# RayDataSource
# ---------------------------------------------------------------------------


class RayDataSource:
    """DataSource backed by a Ray Dataset."""

    def __init__(self, ds: ray.data.Dataset) -> None:
        self._ds = ds

    # --- Factory -----------------------------------------------------------

    @classmethod
    def read_parquet(cls, path: str) -> Self:
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed.")
        return cls(ray.data.read_parquet(path))

    # --- Filtering ---------------------------------------------------------

    def filter(
        self,
        predicate: Callable[[dict], bool] | None = None,
        *,
        col: str | None = None,
        op: str | None = None,
        value: object = None,
    ) -> Self:
        if predicate is not None and (col is not None or op is not None):
            raise ValueError(
                "Cannot specify both predicate and col/op/value. Use one filtering style at a time."
            )

        # Expression-based filter: compile to a row predicate for Ray
        if col is not None and op is not None:
            comp_fn = _COMP_OPS.get(op)
            if comp_fn is None:
                raise ValueError(f"Unsupported operator: {op!r}")

            def _predicate(row: dict) -> bool:
                return bool(comp_fn(row[col], value))

            return RayDataSource(self._ds.filter(_predicate))

        # Row-dict predicate
        if predicate is not None:
            return RayDataSource(self._ds.filter(predicate))

        raise ValueError("Must provide either predicate or col/op/value.")

    # --- Map batches -------------------------------------------------------

    def map_batches(
        self,
        fn: Callable[[dict], dict],
        batch_size: int = 4096,
        batch_format: str = "numpy",
    ) -> Self:
        # Ray's map_batches receives dict-of-arrays when batch_format="numpy"
        return RayDataSource(
            self._ds.map_batches(
                fn,
                batch_size=batch_size,
                batch_format=batch_format,
            )
        )

    # --- Train / val split -------------------------------------------------

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[Self, Self]:
        train_ds, val_ds = self._ds.train_test_split(
            test_size=test_size,
            shuffle=True,
            seed=random_state,
        )
        return RayDataSource(train_ds), RayDataSource(val_ds)

    # --- Balancing ---------------------------------------------------------

    def balance(self, target_col: str, random_state: int = 42) -> Self:
        """Undersample majority classes to match the minority class count.

        Works with both binary (2-class) and multi-class (3-class) targets.
        """
        # Get all unique class values
        unique_vals = self._ds.to_pandas()[target_col].unique()
        if len(unique_vals) <= 1:
            return self

        # Split by class and find minimum count
        class_dses = {}
        class_counts = {}
        for val in unique_vals:
            cls_ds = self._ds.filter(lambda row, v=val: row[target_col] == v)
            class_dses[val] = cls_ds
            class_counts[val] = cls_ds.count()

        min_count = min(class_counts.values())

        if all(c == min_count for c in class_counts.values()):
            return self  # already balanced

        # Undersample each class to min_count
        undersampled = []
        for val, cls_ds in class_dses.items():
            if class_counts[val] > min_count:
                ratio = min_count / class_counts[val]
                undersampled.append(cls_ds.random_sample(ratio, seed=random_state))
            else:
                undersampled.append(cls_ds)

        result = undersampled[0]
        for ds in undersampled[1:]:
            result = result.union(ds)

        return RayDataSource(result.random_shuffle(seed=random_state))

    # --- I/O ---------------------------------------------------------------

    def write_parquet(self, path: str) -> None:
        self._ds.write_parquet(path)

    # --- Iteration ---------------------------------------------------------

    def iter_batches(
        self,
        batch_size: int = 4096,
        batch_format: str = "numpy",
    ) -> Iterator[dict]:
        return self._ds.iter_batches(
            batch_size=batch_size,
            batch_format=batch_format,
        )

    # --- Properties / conversion -------------------------------------------

    def count(self) -> int:
        return self._ds.count()

    def to_pandas(self) -> pd.DataFrame:
        return self._ds.to_pandas()

    def to_ray(self, override_num_blocks: int | None = None) -> ray.data.Dataset:
        return self._ds

    @property
    def is_ray(self) -> bool:
        return True

    @property
    def columns(self) -> list[str]:
        # Ray Datasets don't expose columns directly; materialise a sample
        sample = self._ds.limit(1).to_pandas()
        return list(sample.columns)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def read_parquet(path: str, use_ray: bool = False) -> DataSource:
    """Read a parquet file as a DataSource.

    Args:
        path: Path to the parquet file.
        use_ray: If True, returns a RayDataSource; otherwise PandasDataSource.

    Returns:
        A DataSource wrapping the parquet data.
    """
    if use_ray:
        return RayDataSource.read_parquet(path)
    return PandasDataSource.read_parquet(path)
