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

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename="data.log")


class FitModes(enum.Enum):
    fitting = 0
    training = 1
    evaluation = 2


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
                    values[1:], dtype="float32"
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


def tokenize(x: str) -> list:
    """regex tokenize, less accurate than spacy"""
    return re.findall(r"\w+", x.lower())


def load_data(path: str, fname: str, stop: int = 0) -> pd.DataFrame:
    "reads from zipped yelp data file"
    ls = []

    with zipfile.ZipFile(os.path.join(os.path.expanduser("~"), path)) as zfile:
        logging.info(f"archive contains the following: {zfile.namelist()}")
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


def _get_data(df, x_labels) -> pd.DataFrame:
    return df.loc[:, x_labels].reset_index(drop=True)


@dataclass
class DataParser:
    """wrapper class for handling datasets"""

    df: pd.DataFrame
    stop: int
    max_len: int
    text_col: str = "text"
    label_col: str = "stars"
    x_labels: str = "data"
    y_labels: str = "target"
    spath: str = field(default="projects/yelp_nlp/data/yelp_data", init=True)
    dictionary: corpora.Dictionary = field(default=None)

    def __post_init__(self):

        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(self.df[self.text_col])
            self.dictionary.filter_extremes(no_below=10, no_above=0.97, keep_n=5000)
            self.dictionary.save(f"{os.path.expanduser('~')}/{self.spath}.dict")
            logging.info("dictionary created...")

    def convert_sentences(self):
        self.df[self.df] = self.df[self.text_col].map(
            lambda x: text_sequencer(self.dictionary, x, self.max_len)
        )
        self.df[self.x_labels] = self.df[self.label_col].apply(convert_rating)
        logging.info("converted tokens to numbers...")

    def save(self):
        _get_data(self.df, self.x_labels).to_parquet(
            f"{os.path.expanduser('~')}/{self.spath}.parquet", index=False
        )
        logging.info(
            f"file saved to {os.path.expanduser('~')}/{self.spath}.parquet"
        )  # noqa: E501


@define
class CorpusDataset(Dataset):
    """Dataset class required for pytorch to output items by index"""

    max_len: int
    dictionary: corpora.Dictionary
    data: DataParser
    x_labels: str = field(default="data")
    y_labels: str = field(default="target")
    mode: FitModes = FitModes.training
    test_size: float = 0.20

    def _pre__init__(self):

        super().__init__()

    def set_mode(self, mode: FitModes):

        if mode not in FitModes:
            raise ValueError("not an available mode")
        self.mode = mode

    def __len__(self):
        return self.data.df.__len__()

    def __getitem__(self, i):
        return self.data.df[self.x_labels].iat[i], self.data.df[self.y_labels][i]

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
