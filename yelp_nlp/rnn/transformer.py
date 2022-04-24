from typing import List, Tuple
from dataclasses import dataclass, field

import zipfile
import re
import numpy as np
import pandas as pd
import jsonlines as jsonl

from gensim import corpora

from yelp_nlp.rnn.config import ParserConfig
from yelp_nlp.logging_utils import new_logger, time_decorator
import logging

logger = new_logger(logging.INFO)


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


def _get_data(df, columns) -> pd.DataFrame:
    return df.loc[:, columns].reset_index(drop=True)


@dataclass
class DataParser:
    """wrapper class for handling datasets"""

    df: pd.DataFrame = field(repr=False)
    cfg: ParserConfig = field(default=ParserConfig())
    dictionary: corpora.Dictionary = field(default=None)  # type: ignore

    @time_decorator
    def __post_init__(self):

        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(self.df[self.cfg.text_col])
            self.dictionary.filter_extremes(
                no_below=self.cfg.dict_min,
                no_above=self.cfg.no_above,
                keep_n=self.cfg.dict_keep,
            )
            self.dictionary.save(f"{self.cfg.dictionary_save_path}")
            logger.info("dictionary created...")

    @time_decorator
    def convert_sentences(self):
        self.df[self.cfg.x_labels] = self.df[self.cfg.text_col].map(
            lambda x: text_sequencer(self.dictionary, x, self.cfg.max_len)
        )
        self.df[self.cfg.y_labels] = self.df[self.cfg.label_col].map(convert_rating)
        logger.info("converted tokens to numbers...")
        return self

    def save(self):
        _get_data(self.df, [self.cfg.x_labels] + [self.cfg.y_labels]).to_parquet(
            f"{self.cfg.data_save_path}", index=False
        )
        logger.info(f"file saved to {self.cfg.data_save_path}")  # noqa: E501
