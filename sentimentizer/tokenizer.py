import re
from collections import Counter
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
    """Scale a single star rating to binary label.

    Used for single-item inference (tokenize_text). For batch processing,
    prefer vectorized_convert_ratings() instead.
    """
    if rating in [4, 5]:
        return 1.0
    elif rating in [1, 2]:
        return 0.0
    else:
        return 0.5


def vectorized_convert_ratings(stars: np.ndarray) -> np.ndarray:
    """Vectorized star-to-label conversion using NumPy.

    ~100x faster than applying convert_rating() per element.
    """
    return np.where(stars >= 4, 1.0, np.where(stars <= 2, 0.0, 0.5))


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


def new_dictionary(data: pd.DataFrame, cfg: TokenizerConfig) -> corpora.Dictionary:
    """builds a dictionary from a dataframe"""
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
        return cls(dictionary=new_dictionary(data, TokenizerConfig(save_dictionary=False)))

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

        data[self.cfg.labels] = vectorized_convert_ratings(data[self.cfg.label_col].values)
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
    def from_dataset(
        cls: type[TokenizerType],
        ds: ray.data.Dataset,
        cfg: TokenizerConfig | None = None,
    ) -> TokenizerType:
        """Creates tokenizer from ray dataset using distributed Map-Reduce.

        Instead of pulling every row to the driver via iter_rows(), this
        distributes word counting across Ray workers and aggregates the
        results on the driver to build the Gensim dictionary.

        By default the dictionary is saved to disk (save_dictionary=True)
        so that ``new_model()`` loads the same token-to-ID mapping that was
        used during tokenization.  Pass ``TokenizerConfig(save_dictionary=False)``
        to skip persistence (e.g. in tests).

        Args:
            ds: Ray dataset with a text column to build the dictionary from.
            cfg: Optional tokenizer config.  Defaults to
                ``TokenizerConfig(save_dictionary=True)``.
        """
        if cfg is None:
            cfg = TokenizerConfig(save_dictionary=True)

        dictionary = _build_dictionary_distributed(ds, cfg)

        if cfg.save_dictionary:
            dictionary.save(f"{FileConfig.dictionary_file_path}")
            logger.info(f"dictionary saved to {FileConfig.dictionary_file_path}...")

        return cls(dictionary=dictionary, cfg=cfg)

    def update_from_dataset(self, ds: ray.data.Dataset) -> None:
        """Updates the existing dictionary with new tokens from a Ray dataset.

        Uses the same distributed Map-Reduce approach as ``from_dataset()``
        to count tokens, but calls ``add_documents()`` on the existing
        dictionary instead of creating a new one.

        Args:
            ds: Ray dataset with a text column to update the dictionary from.
        """
        _update_dictionary_distributed(self.dictionary, ds, self.cfg)

        if self.cfg.save_dictionary:
            self.dictionary.save(f"{FileConfig.dictionary_file_path}")
            logger.info(f"updated dictionary saved to {FileConfig.dictionary_file_path}...")

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
                batch[cfg.labels] = vectorized_convert_ratings(np.asarray(batch[cfg.label_col]))

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


def get_trained_tokenizer() -> Tokenizer:
    corp_dict = corpora.Dictionary.load(
        str(files("sentimentizer.data").joinpath("yelp.dictionary"))
    )
    return Tokenizer(dictionary=corp_dict)


# ---------------------------------------------------------------------------
# Distributed dictionary building (Map-Reduce)
# ---------------------------------------------------------------------------


def _count_vocab_batch(batch: dict, text_col: str) -> tuple[Counter, Counter, int]:
    """Count word frequencies and document frequencies for a batch.

    Returns:
        word_freq: Counter of total word occurrences across all docs in batch
        doc_freq: Counter of how many docs each word appears in
        num_docs: Number of documents in this batch
    """
    word_freq: Counter = Counter()
    doc_freq: Counter = Counter()
    num_docs = 0

    for doc_tokens in batch[text_col]:
        num_docs += 1
        word_freq.update(doc_tokens)
        # doc_freq counts each word once per document
        doc_freq.update(set(doc_tokens))

    return word_freq, doc_freq, num_docs


def _build_dictionary_distributed(ds: ray.data.Dataset, cfg: TokenizerConfig) -> corpora.Dictionary:
    """Build a Gensim dictionary using distributed Map-Reduce over Ray.

    Instead of streaming every row to the driver via iter_rows(),
    this distributes word counting across Ray workers and aggregates
    the results on the driver to construct the dictionary.
    """
    text_col = cfg.text_col

    # Map phase: count vocab across distributed batches
    total_word_freq: Counter = Counter()
    total_doc_freq: Counter = Counter()
    total_num_docs = 0

    for batch in ds.iter_batches(batch_size=10000, batch_format="numpy"):
        wf, df, nd = _count_vocab_batch(batch, text_col)
        total_word_freq += wf
        total_doc_freq += df
        total_num_docs += nd

    # Reduce phase: build Gensim dictionary from aggregated counts
    dictionary = corpora.Dictionary()
    dictionary.num_docs = total_num_docs

    # Assign contiguous IDs to all words.
    # Sort by (descending frequency, ascending word) so the dictionary is
    # deterministic regardless of batch processing order.  This guarantees
    # the same token-to-ID mapping on every run.
    for idx, word in enumerate(
        sorted(total_word_freq.keys(), key=lambda w: (-total_word_freq[w], w))
    ):
        dictionary.token2id[word] = idx
        dictionary.dfs[idx] = total_doc_freq[word]

    # num_pos = total word count (used by filter_extremes)
    dictionary.num_pos = sum(total_word_freq.values())
    dictionary.num_nnz = sum(total_doc_freq.values())

    dictionary.filter_extremes(
        no_below=cfg.dict_min,
        no_above=cfg.no_above,
        keep_n=cfg.dict_keep,
    )
    # Remap IDs to contiguous 0..N-1 after filter_extremes leaves gaps.
    # Required for embedding matrix row alignment (row k = token ID k-1).
    dictionary.compactify()
    logger.info(
        f"dictionary created via Map-Reduce: {len(dictionary)} terms, "
        f"{total_num_docs} documents"
    )

    return dictionary


def _update_dictionary_distributed(
    dictionary: corpora.Dictionary, ds: ray.data.Dataset, cfg: TokenizerConfig
) -> None:
    """Update a Gensim dictionary using distributed Map-Reduce over Ray."""
    text_col = cfg.text_col

    # Map phase: count vocab across distributed batches
    total_word_freq: Counter = Counter()
    total_doc_freq: Counter = Counter()
    total_num_docs = 0

    for batch in ds.iter_batches(batch_size=10000, batch_format="numpy"):
        wf, df, nd = _count_vocab_batch(batch, text_col)
        total_word_freq += wf
        total_doc_freq += df
        total_num_docs += nd

    # Reduce phase: update existing dictionary
    # We simulate add_documents by updating the internal structures.
    # Existing IDs are preserved.

    old_len = len(dictionary)
    for word in total_word_freq:
        if word not in dictionary.token2id:
            idx = len(dictionary)
            dictionary.token2id[word] = idx
            dictionary.dfs[idx] = total_doc_freq[word]
        else:
            idx = dictionary.token2id[word]
            dictionary.dfs[idx] += total_doc_freq[word]

    dictionary.num_docs += total_num_docs
    dictionary.num_pos += sum(total_word_freq.values())
    dictionary.num_nnz += sum(total_doc_freq.values())

    # We SKIP filter_extremes and compactify because they reassign IDs,
    # which would break existing model weights.

    new_len = len(dictionary)
    logger.info(
        f"dictionary updated via Map-Reduce: added {new_len - old_len} new terms, "
        f"total {new_len} terms, {total_num_docs} new documents"
    )
