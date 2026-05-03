import re
from collections.abc import Generator
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TypeVar

import numpy as np
import pandas as pd
import ray
from gensim import corpora

from sentimentizer import new_logger, time_decorator
from sentimentizer.config import DEFAULT_LOG_LEVEL, FileConfig, TokenizerConfig

logger = new_logger(DEFAULT_LOG_LEVEL)

TokenizerType = TypeVar("TokenizerType", bound="Tokenizer")

pattern = re.compile(r"[a-z0-9'-]+")


def convert_rating(rating: int) -> float:
    """scaling ratings from 0 to 1"""
    if rating in [4, 5]:
        return 1.0
    elif rating in [1, 2]:
        return 0.0
    else:
        return 0.5


def text_sequencer(
    dictionary: corpora.Dictionary, text: list[str], max_len: int = 200
) -> np.ndarray:
    """
    converts tokens to numeric representation by dictionary;
    zero is considered padding
    """

    processed = np.zeros(max_len, dtype=int)
    # OOV token index: after compactify(), IDs are 0..N-1 so the
    # embedding matrix has rows [pad, word_0, word_1, ..., word_{N-1}, OOV].
    # Row N+1 = len(dictionary) + 1 is the dedicated OOV row.
    dict_final = len(dictionary) + 1

    for i, word in enumerate(text):
        if i >= max_len:
            return processed
        if word in dictionary.token2id:
            # the ids have an offset of 1 for this because
            # 0 represents a padded value in pytorch
            processed[i] = dictionary.token2id[word] + 1
        else:
            processed[i] = dict_final

    return processed


def regex_tokenize(x: str) -> list[str]:
    """regex tokenize, less accurate than spacy"""
    return pattern.findall(x.lower())


def _get_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.loc[:, columns].reset_index(drop=True)


def _new_dictionary(data: pd.DataFrame, cfg: TokenizerConfig) -> corpora.Dictionary:
    dictionary = corpora.Dictionary(data[cfg.text_col])
    dictionary.filter_extremes(
        no_below=cfg.dict_min,
        no_above=cfg.no_above,
        keep_n=cfg.dict_keep,
    )
    # Remap IDs to contiguous 0..N-1 after filter_extremes leaves gaps.
    # Required for embedding matrix row alignment (row k = token ID k-1).
    dictionary.compactify()
    logger.info("dictionary created...")

    if cfg.save_dictionary:
        dictionary.save(f"{FileConfig.dictionary_file_path}")
        logger.info(f"dictionary saved to {FileConfig.dictionary_file_path}...")

    return dictionary


@dataclass
class Tokenizer:
    """wrapper class for handling tokenization of datasets"""

    dictionary: corpora.Dictionary
    cfg: TokenizerConfig = field(default_factory=TokenizerConfig)

    @classmethod
    def from_data(cls: type[TokenizerType], data: pd.DataFrame) -> TokenizerType:
        """creates tokenizer from dataframe"""
        return cls(dictionary=_new_dictionary(data, TokenizerConfig(save_dictionary=False)))

    @time_decorator
    def transform_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """transforms dataframe with text and target"""
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")

        # Work on a copy to avoid mutating the input DataFrame
        data = data.copy()
        
        # Drop neutral (3-star) reviews for strict binary classification
        data = data[data[self.cfg.label_col] != 3].copy()
        data[self.cfg.inputs] = data[self.cfg.text_col].map(
            lambda text: text_sequencer(self.dictionary, text, self.cfg.max_len)  # type: ignore[arg-type]
        )

        data[self.cfg.labels] = data[self.cfg.label_col].map(convert_rating)
        logger.info("converted tokens to numbers...")
        return data

    def tokenize_text(self, text: str) -> np.ndarray:
        """converts string phrase to numpy array"""
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")
        tokens = regex_tokenize(text)
        return text_sequencer(self.dictionary, tokens, self.cfg.max_len).reshape(
            1, self.cfg.max_len
        )

    @classmethod
    def from_dataset(cls: type[TokenizerType], ds: ray.data.Dataset) -> TokenizerType:
        """creates tokenizer from ray dataset"""
        cfg = TokenizerConfig(save_dictionary=False)
        # Create dictionary from dataset tokens column
        # ds.iter_rows() yields items. We want the text_col.

        def gen_docs() -> Generator:
            for row in ds.iter_rows():
                yield row[cfg.text_col]

        dictionary = corpora.Dictionary(gen_docs())
        dictionary.filter_extremes(
            no_below=cfg.dict_min,
            no_above=cfg.no_above,
            keep_n=cfg.dict_keep,
        )
        # Remap IDs to contiguous 0..N-1 after filter_extremes leaves gaps.
        dictionary.compactify()
        logger.info("dictionary created...")

        if cfg.save_dictionary:
            dictionary.save(f"{FileConfig.dictionary_file_path}")
            logger.info(f"dictionary saved to {FileConfig.dictionary_file_path}...")

        return cls(dictionary=dictionary)

    @time_decorator
    def transform_dataset(self, ds: ray.data.Dataset) -> ray.data.Dataset:
        """transforms ray dataset with text and target"""
        if self.dictionary is None:
            raise ValueError("no dictionary loaded")

        # Capture for closure
        dictionary = self.dictionary
        cfg = self.cfg

        # Drop neutral (3-star) reviews for strict binary classification
        ds = ds.filter(lambda row: row[cfg.label_col] != 3)

        def transform_batch(batch: dict) -> dict:
            inputs = []
            for text in batch[cfg.text_col]:
                inputs.append(text_sequencer(dictionary, text, cfg.max_len))

            batch[cfg.inputs] = np.array(inputs)

            if cfg.label_col in batch:
                batch[cfg.labels] = np.array([convert_rating(r) for r in batch[cfg.label_col]])

            # Drop variable-length columns that cause Arrow conversion issues
            # Keep only the numeric columns needed for training
            cols_to_keep = {cfg.inputs, cfg.labels}
            for col in list(batch.keys()):
                if col not in cols_to_keep:
                    del batch[col]

            return batch

        ds = ds.map_batches(transform_batch, batch_format="numpy")
        logger.info("converted tokens to numbers...")
        return ds

    def save(self, data: pd.DataFrame) -> None:
        _get_data(data, [self.cfg.inputs] + [self.cfg.labels]).to_parquet(
            f"{FileConfig.processed_reviews_file_path}", index=False
        )
        logger.info(f"file saved to {FileConfig.processed_reviews_file_path}")  # noqa: E501


def get_trained_tokenizer() -> Tokenizer:
    corp_dict = corpora.Dictionary.load(
        str(files("sentimentizer.data").joinpath("yelp.dictionary"))
    )
    return Tokenizer(dictionary=corp_dict)
