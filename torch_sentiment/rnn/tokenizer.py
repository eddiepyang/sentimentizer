from typing import List, TypeVar, Optional
from dataclasses import dataclass, field
from importlib.resources import files

import re
import numpy as np
import pandas as pd
from gensim import corpora

from torch_sentiment.rnn.config import FileConfig, TokenizerConfig, DEFAULT_LOG_LEVEL
from torch_sentiment.logging_utils import new_logger


logger = new_logger(DEFAULT_LOG_LEVEL)

TokenizerAny = TypeVar("TokenizerAny", bound="Tokenizer")

pattern = re.compile(r"[a-z0-9'-]+")


def convert_rating(rating: int) -> float:
    """scaling ratings from 0 to 1"""
    if rating in [4, 5]:
        return 1.0
    elif rating in [1, 2]:
        return 0.0
    else:
        return 0.5


def convert_rating_linear(rating: int, max_rating: int) -> float:

    """scaling ratings from 0 to 1 linearly"""
    return rating / max_rating


def text_sequencer(
    dictionary: corpora.Dictionary, text: list, max_len: int = 200
) -> np.ndarray:

    """
    converts tokens to numeric representation by dictionary;
    zero is considered padding
    """

    processed = np.zeros(max_len, dtype=int)
    # in case the word is not in the dictionary because it was
    # filtered out use this number to represent an out of set id
    dict_final = len(dictionary.keys()) + 1

    for i, word in enumerate(text):
        if i >= max_len:
            return processed
        if word in dictionary.token2id.keys():
            # the ids have an offset of 1 for this because
            # 0 represents a padded value in pytorch
            processed[i] = dictionary.token2id[word] + 1
        else:
            processed[i] = dict_final

    return processed


def tokenize(x: str) -> List[str]:
    """regex tokenize, less accurate than spacy"""
    return pattern.findall(x.lower())


def _get_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.loc[:, columns].reset_index(drop=True)


def _new_dictionary(data: pd.DataFrame, cfg: TokenizerConfig) -> corpora.Dictionary:
    dictionary = corpora.Dictionary(data[cfg.text_col])
    dictionary.filter_extremes(
        no_below=cfg.dict_min,
        no_above=cfg.no_above,
        keep_n=cfg.dict_keep,
    )
    logger.info("dictionary created...")

    if cfg.save_dictionary:
        dictionary.save(f"{FileConfig.dictionary_file_path}")
        logger.info(f"dictionary saved to {FileConfig.dictionary_file_path}...")

    return dictionary


@dataclass
class Tokenizer:
    """wrapper class for handling tokenization of datasets"""

    cfg: TokenizerConfig = field(default_factory=TokenizerConfig)
    dictionary: Optional[corpora.Dictionary] = None

    @classmethod
    def from_data(cls: type[TokenizerAny], data: pd.DataFrame) -> TokenizerAny:
        """creates tokenizer from dataframe"""
        return cls(
            dictionary=_new_dictionary(data, TokenizerConfig(save_dictionary=False))
        )

    # @time_decorator
    def transform_dataframe(self, data: pd.DataFrame) -> TokenizerAny:
        """transforms dataframe with text and target"""

        data[self.cfg.inputs] = data[self.cfg.text_col].map(
            lambda text: text_sequencer(self.dictionary, text, self.cfg.max_len)
        )

        data[self.cfg.labels] = data[self.cfg.label_col].map(convert_rating)
        logger.info("converted tokens to numbers...")
        return self

    def tokenize_text(self, text: str) -> np.ndarray:
        """converts string phrase to numpy array"""
        tokens = tokenize(text)
        return text_sequencer(self.dictionary, tokens, self.cfg.max_len).reshape(
            1, self.cfg.max_len
        )

    def save(self, data: pd.DataFrame) -> None:
        _get_data(data, [self.cfg.inputs] + [self.cfg.labels]).to_parquet(
            f"{FileConfig.reviews_file_path}", index=False
        )
        logger.info(f"file saved to {FileConfig.reviews_file_path}")  # noqa: E501


def get_trained_tokenizer() -> Tokenizer:
    corp_dict = corpora.Dictionary.load(
        str(files("torch_sentiment.data").joinpath("yelp.dictionary"))
    )
    return Tokenizer(dictionary=corp_dict)
