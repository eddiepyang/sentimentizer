import numpy as np
import pandas as pd
import ray
import torch
from attr import define
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


@define
class CorpusDataset(Dataset):
    """Dataset class required for pytorch to output items by index"""

    data: pd.DataFrame
    x_labels: str = "data"
    y_labels: str = "target"

    def __attr_pre__init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[self.x_labels].iat[i]
        y = self.data[self.y_labels].iat[i]
        # Convert via numpy to handle Arrow TensorArrayElement types
        # that Ray Data writes to parquet
        return torch.tensor(np.asarray(x), dtype=torch.long), torch.tensor(
            np.asarray(y), dtype=torch.float32
        )


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
