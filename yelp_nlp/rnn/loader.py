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
from yelp_nlp.logging_utils import new_logger
from yelp_nlp.rnn.config import LogLevels

logger = new_logger(LogLevels.debug.value)


def split_df(
    df: pd.DataFrame, test_size: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    tr_idx, val_idx = train_test_split(df.index.values, test_size=test_size)
    return df.iloc[tr_idx], df.iloc[val_idx]  # type: ignore


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


def new_train_val_corpus_datasets(
    data_path: str, test_size=0.2
) -> Tuple[CorpusDataset, CorpusDataset]:

    df = pd.read_parquet(data_path)
    train_df, val_df = split_df(df, test_size=test_size)
    return CorpusDataset(data=train_df), CorpusDataset(val_df)
