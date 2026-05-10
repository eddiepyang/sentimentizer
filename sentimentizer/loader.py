from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    import ray
    from ray.data import DataContext

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)

# Enable rich progress bars for Ray Data and suppress the "new progress UI"
# info message.  Must be set before any Dataset operations are executed.
# See https://docs.ray.io/en/2.55.1/data/api/doc/ray.data.DataContext.html
if RAY_AVAILABLE:
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False


def compute_pos_weight(df: pd.DataFrame, target_col: str = "target") -> float:
    """Compute pos_weight for BCEWithLogitsLoss from class distribution.

    Returns neg_count / pos_count, which downweights the majority class
    (typically positive in Yelp reviews) so the loss contribution is
    balanced without throwing away any data.

    Args:
        df: DataFrame with a binary target column (0.0 and 1.0).
        target_col: Name of the target column.

    Returns:
        pos_weight = neg_count / pos_count.  Falls back to 1.0 if
        either class is empty.
    """
    neg_count = int((df[target_col] == 0.0).sum())
    pos_count = int((df[target_col] == 1.0).sum())

    if pos_count == 0 or neg_count == 0:
        logger.warning(
            "cannot compute pos_weight: one class is empty, using 1.0",
            neg_count=neg_count,
            pos_count=pos_count,
        )
        return 1.0

    weight = neg_count / pos_count
    logger.info(
        "computed pos_weight from class distribution",
        neg_count=neg_count,
        pos_count=pos_count,
        pos_weight=round(weight, 4),
    )
    return weight


class CorpusDataset(Dataset):
    """Dataset class required for pytorch to output items by index.

    Pre-converts DataFrame columns to tensors at init time for faster
    iteration during training. Avoids per-sample numpy/tensor conversion.
    """

    def __init__(
        self, data: pd.DataFrame, x_labels: str = "data", y_labels: str = "target"
    ) -> None:
        super().__init__()
        # Pre-convert columns to lists of tensors at init time
        # This avoids per-__getitem__ numpy/tensor conversion overhead
        self._x_data = [
            torch.tensor(np.asarray(val), dtype=torch.long) for val in data[x_labels].values
        ]
        self._y_data = [
            torch.tensor(np.asarray(val), dtype=torch.float32) for val in data[y_labels].values
        ]

    def __len__(self) -> int:
        return len(self._x_data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x_data[i], self._y_data[i]


def _balance_dataframe(
    df: pd.DataFrame, target_col: str = "target", random_state: int = 42
) -> pd.DataFrame:
    """Undersample the majority class to match the minority class count.

    Only balances training data — validation data should reflect real-world
    distribution for meaningful metrics.

    Args:
        df: DataFrame with a binary target column (0.0 and 1.0).
        target_col: Name of the target column.
        random_state: Seed for reproducible undersampling.

    Returns:
        Balanced DataFrame with equal class counts, shuffled.
    """
    neg_count = (df[target_col] == 0.0).sum()
    pos_count = (df[target_col] == 1.0).sum()

    logger.info(f"class balance before undersampling: negative={neg_count}, positive={pos_count}")

    if pos_count > neg_count:
        pos_df = df[df[target_col] == 1.0].sample(n=neg_count, random_state=random_state)
        neg_df = df[df[target_col] == 0.0]
    elif neg_count > pos_count:
        neg_df = df[df[target_col] == 0.0].sample(n=pos_count, random_state=random_state)
        pos_df = df[df[target_col] == 1.0]
    else:
        # Already balanced
        return df

    balanced = pd.concat([neg_df, pos_df]).sample(frac=1, random_state=random_state)
    balanced_neg = (balanced[target_col] == 0.0).sum()
    balanced_pos = (balanced[target_col] == 1.0).sum()
    logger.info(
        f"class balance after undersampling: negative={balanced_neg}, positive={balanced_pos}, "
        f"total={len(balanced)}"
    )
    return balanced


def load_train_val_corpus_datasets(
    data_path: str,
    test_size: float = 0.2,
    balance_classes: bool = False,
    random_state: int = 42,
) -> tuple[CorpusDataset, CorpusDataset]:
    df = pd.read_parquet(data_path)
    train_df, val_df = train_test_split(df, test_size=test_size)
    del df

    if balance_classes:
        train_df = _balance_dataframe(train_df, target_col="target", random_state=random_state)

    return CorpusDataset(data=train_df), CorpusDataset(val_df)


def _balance_ray_dataset(
    ds: ray.data.Dataset, target_col: str = "target", random_state: int = 42
) -> ray.data.Dataset:
    """Undersample the majority class in a Ray Dataset to match the minority class.

    Only balances training data — validation data should reflect real-world
    distribution for meaningful metrics.

    Uses ``random_sample`` to keep a fraction of the majority class rows.
    See https://docs.ray.io/en/2.55.1/data/api/doc/ray.data.Dataset.random_sample.html

    Args:
        ds: Ray Dataset with a binary target column (0.0 and 1.0).
        target_col: Name of the target column.
        random_state: Seed for reproducible undersampling.

    Returns:
        Balanced Ray Dataset with equal class counts, shuffled.
    """
    neg_ds = ds.filter(lambda row: row[target_col] == 0.0)
    pos_ds = ds.filter(lambda row: row[target_col] == 1.0)
    neg_count = neg_ds.count()
    pos_count = pos_ds.count()

    logger.info(f"class balance before undersampling: negative={neg_count}, positive={pos_count}")

    if pos_count > neg_count and neg_count > 0:
        keep_ratio = neg_count / pos_count
        pos_keep = pos_ds.random_sample(keep_ratio, seed=random_state)
        balanced = pos_keep.union(neg_ds)
    elif neg_count > pos_count and pos_count > 0:
        keep_ratio = pos_count / neg_count
        neg_keep = neg_ds.random_sample(keep_ratio, seed=random_state)
        balanced = neg_keep.union(pos_ds)
    else:
        # Already balanced or one class is empty
        return ds

    balanced_neg = balanced.filter(lambda row: row[target_col] == 0.0).count()
    balanced_pos = balanced.filter(lambda row: row[target_col] == 1.0).count()
    logger.info(
        f"class balance after undersampling: negative={balanced_neg}, positive={balanced_pos}, "
        f"total={balanced.count()}"
    )
    return balanced.random_shuffle(seed=random_state)  # shuffle


def load_train_val_ray_datasets(
    data_path: str,
    test_size: float = 0.2,
    balance_classes: bool = False,
    random_state: int = 42,
) -> tuple[ray.data.Dataset, ray.data.Dataset]:
    """Load processed parquet data as Ray Datasets for distributed training.

    Splits into train and validation datasets using ``train_test_split``.
    See https://docs.ray.io/en/2.55.1/data/api/doc/ray.data.Dataset.train_test_split.html

    Optionally balances classes in the training set by undersampling
    the majority class.
    """
    ds = ray.data.read_parquet(data_path)
    train_ds, val_ds = ds.train_test_split(test_size=test_size, shuffle=True, seed=random_state)

    if balance_classes:
        train_ds = _balance_ray_dataset(train_ds, target_col="target", random_state=random_state)

    logger.info(f"loaded ray datasets: train={train_ds.count()}, val={val_ds.count()}")
    return train_ds, val_ds
