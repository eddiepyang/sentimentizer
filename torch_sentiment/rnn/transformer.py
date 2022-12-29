from typing import List, Tuple
from dataclasses import dataclass, field

import re
import numpy as np
import pandas as pd
from gensim import corpora

from torch_sentiment.rnn.config import FileConfig, TransformerConfig
from torch_sentiment.logging_utils import new_logger, time_decorator, time_decorator_factory
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


def _get_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.loc[:, columns].reset_index(drop=True)


@dataclass
class DataTransformer:
    """wrapper class for handling tokenization of datasets"""

    data: pd.DataFrame = field(default=None, repr=False)
    cfg: TransformerConfig = field(default_factory=TransformerConfig()) # type: ignore
    dictionary: corpora.Dictionary = None # type: ignore

    @time_decorator_factory(optional_text="DataTransformer save_dictionary")
    def __post_init__(self):

        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(self.data[self.cfg.text_col])
            self.dictionary.filter_extremes(
                no_below=self.cfg.dict_min,
                no_above=self.cfg.no_above,
                keep_n=self.cfg.dict_keep,
            )
            logger.info("dictionary created...")

            if self.cfg.save_dictionary:
                self.dictionary.save(f"{FileConfig.dictionary_file_path}")
                logger.info(f"dictionary saved to {FileConfig.dictionary_file_path}...")

    @time_decorator
    def transform_sentences(self):
        self.data[self.cfg.inputs] = self.data[self.cfg.text_col].map(
            lambda text: text_sequencer(self.dictionary, text, self.cfg.max_len)
        )
        self.data[self.cfg.labels] = self.data[self.cfg.label_col].map(convert_rating)
        logger.info("converted tokens to numbers...")
        return self

    def save(self):
        _get_data(self.data, [self.cfg.inputs] + [self.cfg.labels]).to_parquet(
            f"{FileConfig.reviews_file_path}", index=False
        )
        logger.info(f"file saved to {FileConfig.reviews_file_path}")  # noqa: E501


def get_trained_tokenizer(path: str) -> DataTransformer:
    corp_dict = corpora.Dictionary()
    corp_dict.load(path)
    return DataTransformer(dictionary=corp_dict)