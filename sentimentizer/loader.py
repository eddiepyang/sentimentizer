import numpy as np
import pandas as pd
import ray
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


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


def load_train_val_corpus_datasets(
    data_path: str, test_size: float = 0.2
) -> tuple[CorpusDataset, CorpusDataset]:
    df = pd.read_parquet(data_path)
    train_df, val_df = train_test_split(df, test_size=test_size)
    del df
    return CorpusDataset(data=train_df), CorpusDataset(val_df)


def load_train_val_ray_datasets(
    data_path: str, test_size: float = 0.2
) -> tuple[ray.data.Dataset, ray.data.Dataset]:
    """Load processed parquet data as Ray Datasets for distributed training.

    Splits into train and validation datasets using random_split.
    """
    ds = ray.data.read_parquet(data_path)
    train_ds, val_ds = ds.random_split([1 - test_size, test_size])
    logger.info(f"loaded ray datasets: train={train_ds.count()}, val={val_ds.count()}")
    return train_ds, val_ds
