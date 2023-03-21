from attr import define
import pandas as pd

import torch
from typing import Tuple
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sentimentizer import new_logger
from sentimentizer.config import DEFAULT_LOG_LEVEL

logger = new_logger(DEFAULT_LOG_LEVEL)


@define
class CorpusDataset(Dataset):
    """Dataset class required for pytorch to output items by index"""

    data: pd.DataFrame
    x_labels: str = "data"
    y_labels: str = "target"

    def __attr_pre__init__(self):

        super().__init__()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, i):
        return torch.tensor(self.data[self.x_labels].iat[i]), torch.tensor(
            self.data[self.y_labels].iat[i]
        )


def load_train_val_corpus_datasets(
    data_path: str, test_size=0.2
) -> Tuple[CorpusDataset, CorpusDataset]:

    df = pd.read_parquet(data_path)
    train_df, val_df = train_test_split(df, test_size=test_size)
    del df
    return CorpusDataset(data=train_df), CorpusDataset(val_df)
