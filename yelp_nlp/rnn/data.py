from dataclasses import dataclass, field
import enum
from attr import define
from typing import List, Tuple
import numpy as np
import pandas as pd
import jsonlines as jsonl
import zipfile
import logging

from gensim import corpora
import os
import re
from yelp_nlp import root

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from yelp_nlp.rnn.config import ParserConfig, FitModes
from yelp_nlp.logging_utils import new_logger

logger = new_logger(logging.INFO)


def load_embeddings(
    emb_path: str,
    emb_file: str = "glove.6B.zip",
    emb_subfile: str = "glove.6B.100d.txt",
) -> dict:

    """load glove vectors"""

    embeddings_index = {}

    with zipfile.ZipFile(
        os.path.join(os.path.expanduser("~"), emb_path, emb_file), "r"
    ) as f:
        with f.open(emb_subfile, "r") as z:
            for line in z:
                values = line.split()
                embeddings_index[values[0].decode()] = np.asarray(
                    values[1:], dtype=np.float32
                )  # noqa: E501

    return embeddings_index


def id_to_glove(
    dictionary: corpora.Dictionary, emb_path: str, emb_n: int = 100
) -> np.ndarray:

    """converts local dictionary to embeddings from glove"""

    embeddings_index = load_embeddings(emb_path)
    conversion_table = {}

    for word in dictionary.values():
        if word in embeddings_index:
            conversion_table[dictionary.token2id[word] + 1] = embeddings_index[word]
        else:
            conversion_table[dictionary.token2id[word] + 1] = np.random.normal(
                0, 0.32, emb_n
            )
    return np.vstack(
        (np.zeros(emb_n), list(conversion_table.values()), np.random.randn(emb_n))
    )


def convert_rating(rating: int) -> float:
    """scaling ratings from 0 to 1"""
    if rating in [4, 5]:
        return 1.0
    elif rating in [1, 2]:
        return 0.0
    else:
        return -1.0


def convert_rating_linear(rating: int, max_rating: int) -> float:

    """scaling ratings from 0 to 1 linearly"""
    return rating / max_rating


def text_sequencer(
    dictionary: corpora.Dictionary, text: list, max_len: int = 200
) -> np.ndarray:

    """converts tokens to numeric representation by dictionary"""

    processed = np.zeros(max_len, dtype=int)
    # in case the word is not in the dictionary because it was
    # filtered out use this number to represent an out of set id
    dict_final = len(dictionary.keys()) + 1

    for i, word in enumerate(text):
        if i >= max_len:
            return processed
        if word in dictionary.token2id.keys():
            # the ids have an offset of 1 for this because
            # 0 represents a padded value
            processed[i] = dictionary.token2id[word] + 1
        else:
            processed[i] = dict_final

    return processed


def tokenize(x: str) -> List[str]:
    """regex tokenize, less accurate than spacy"""
    return re.findall(r"\w+", x.lower())


def load_data(path: str, fname: str, stop: int = 0) -> pd.DataFrame:
    "reads from zipped yelp data file"
    ls = []

    with zipfile.ZipFile(os.path.join(os.path.expanduser("~"), path)) as zfile:
        logger.info(f"archive contains the following: {zfile.namelist()}")
        inf = zfile.open(fname)

        with jsonl.Reader(inf) as file:
            for i, line in enumerate(file):
                line["text"] = tokenize(line.get("text"))
                ls.append(line)
                if i == stop - 1:
                    break

    return pd.DataFrame(ls)


def split_df(
    df: pd.DataFrame, test_size: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    tr_idx, val_idx = train_test_split(df.index.values, test_size=test_size)
    return df.iloc[tr_idx], df.iloc[val_idx]  # type: ignore


def _get_data(df, columns) -> pd.DataFrame:
    return df.loc[:, columns].reset_index(drop=True)


@dataclass
class DataParser:
    """wrapper class for handling datasets"""

    df: pd.DataFrame
    cfg: ParserConfig
    dictionary: corpora.Dictionary = field(default=None)  # type: ignore

    def __post_init__(self):

        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(self.df[self.cfg.text_col])
            self.dictionary.filter_extremes(
                no_below=self.cfg.dict_min,
                no_above=self.cfg.no_above,
                keep_n=self.cfg.dict_keep,
            )
            self.dictionary.save(f"{os.path.expanduser('~')}/{self.cfg.save_path}.dict")
            logger.info("dictionary created...")

    def convert_sentences(self):
        self.df[self.cfg.x_labels] = self.df[self.cfg.text_col].map(
            lambda x: text_sequencer(self.dictionary, x, self.cfg.max_len)
        )
        self.df[self.cfg.x_labels] = self.df[self.cfg.label_col].map(convert_rating)
        logger.info("converted tokens to numbers...")
        return self.df

    def save(self):
        _get_data(self.df, [self.cfg.x_labels] + [self.cfg.y_labels]).to_parquet(
            f"{os.path.expanduser('~')}/{self.cfg.save_path}.parquet", index=False
        )
        logger.info(
            f"file saved to {os.path.expanduser('~')}/{self.cfg.save_path}.parquet"
        )  # noqa: E501


@define
class CorpusDataset(Dataset):
    """Dataset class required for pytorch to output items by index"""

    data: pd.DataFrame
    x_labels: str = field(default="data")
    y_labels: str = field(default="target")

    def _pre__init__(self):

        super().__init__()

    def set_mode(self, mode: FitModes):  # todo: may not need this

        if mode not in FitModes:
            raise ValueError("not an available mode")
        self.mode = mode

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, i):
        return self.data[self.x_labels].iat[i], self.data[self.y_labels].iat[i]

    # @property
    # def train(self):
    #     return self._train

    # @train.setter
    # def train(self, value):
    #     self._train = self.data.df.iloc[self.tr_idx]

    # @property
    # def val(self):
    #     return self._val

    # @val.setter
    # def val(self, value):
    #     self._val = self.data.df.iloc[self.val_idx]
