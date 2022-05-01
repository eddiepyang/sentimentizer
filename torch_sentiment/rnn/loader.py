from dataclasses import dataclass, field
import enum
from attr import define
from typing import List, Tuple
import numpy as np
import pandas as pd
import jsonlines as jsonl

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch_sentiment.logging_utils import new_logger
from torch_sentiment.rnn.config import LogLevels

logger = new_logger(LogLevels.debug.value)


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
    return CorpusDataset(data=train_df), CorpusDataset(val_df)
