from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Generator
from typing import IO, TYPE_CHECKING

import numpy as np
import orjson as json
import pandas as pd
from gensim import corpora
from gensim import downloader as gensim_api

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

if TYPE_CHECKING:
    import ray.data

from sentimentizer import new_logger, time_decorator
from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    EMBEDDING_DTYPE,
    EMBEDDING_RANDOM_MEAN,
    EMBEDDING_RANDOM_STD,
    EXTRACT_LOG_INTERVAL,
    EmbeddingsConfig,
)
from sentimentizer.tokenizer import regex_tokenize

logger = new_logger(DEFAULT_LOG_LEVEL)


def process_json(json_file: IO[bytes], stop: int = 0) -> Generator:
    for i, line in enumerate(json_file):
        if i % EXTRACT_LOG_INTERVAL == 0:
            logger.debug(f"processing line {i}")
        dc = json.loads(line)
        if i >= stop and stop != 0:
            break
        yield dc


@time_decorator
def extract_data_ray(file_path: str, compressed_file_name: str, stop: int = 0) -> ray.data.Dataset:
    """Reads from zipped or tarred yelp data file using Ray.

    Supports both .zip and .tar/.tar.gz archives.
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is required for extract_data_ray. Use extract_data_local instead.")

    def generate_lines(_row: int) -> Generator:
        if file_path.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(file_path, "r:*") as tar:
                member = tar.getmember(compressed_file_name)
                f = tar.extractfile(member)
                if f is None:
                    raise ValueError(f"Could not extract {compressed_file_name} from {file_path}")
                yield from process_json(f, stop)
        else:
            with zipfile.ZipFile(file_path) as zfile:
                inf = zfile.open(compressed_file_name)
                yield from process_json(inf, stop)

    # Use flat_map to read from the archive
    ds = ray.data.range(1).flat_map(generate_lines)

    def tokenize(row: dict) -> dict:
        row["tokens"] = regex_tokenize(row["text"])
        return row

    return ds.map(tokenize)


@time_decorator
def extract_data_local(file_path: str, compressed_file_name: str, stop: int = 0) -> pd.DataFrame:
    """Reads from zipped or tarred yelp data file into a Pandas DataFrame.

    Local fallback when Ray is not available.
    """

    def generate_lines() -> Generator:
        if file_path.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(file_path, "r:*") as tar:
                member = tar.getmember(compressed_file_name)
                f = tar.extractfile(member)
                if f is None:
                    raise ValueError(f"Could not extract {compressed_file_name} from {file_path}")
                yield from process_json(f, stop)
        else:
            with zipfile.ZipFile(file_path) as zfile:
                inf = zfile.open(compressed_file_name)
                yield from process_json(inf, stop)

    data = list(generate_lines())
    df = pd.DataFrame(data)
    df["tokens"] = df["text"].apply(regex_tokenize)
    return df


@time_decorator
def extract_embeddings(
    dictionary: corpora.Dictionary, cfg: EmbeddingsConfig
) -> dict[int, np.ndarray]:
    """Load pre-trained word vectors via gensim.downloader.

    Auto-downloads and caches the model (e.g. glove-wiki-gigaword-100)
    to ~/gensim-data/ on first use.
    """
    logger.info(f"loading embeddings model: {cfg.model_name} ...")
    glove = gensim_api.load(cfg.model_name)

    embeddings_dict: dict[int, np.ndarray] = {}
    for word, token_id in dictionary.token2id.items():
        if word in glove:
            embeddings_dict[token_id + 1] = glove[word].astype(EMBEDDING_DTYPE)

    logger.info(
        f"matched {len(embeddings_dict)}/{len(dictionary)} dictionary words "
        f"to {cfg.model_name} vectors"
    )
    return embeddings_dict


@time_decorator
def new_embedding_weights(dictionary: corpora.Dictionary, cfg: EmbeddingsConfig) -> np.ndarray:
    """Build the embedding weight matrix from pre-trained vectors.

    Embedding matrix layout:
        Row 0:              padding (all zeros)
        Rows 1..N:          word embeddings, sorted by token ID
                            (row k = embedding for token ID k-1)
        Row N+1:            out-of-vocabulary (OOV) random vector
    Where N = len(dictionary).
    """

    embeddings_dict: dict = extract_embeddings(dictionary, cfg)

    for word in dictionary.values():
        key = dictionary.token2id[word] + 1
        if key not in embeddings_dict:
            embeddings_dict[key] = np.random.normal(
                EMBEDDING_RANDOM_MEAN, EMBEDDING_RANDOM_STD, cfg.emb_length
            )

    # Sort by key (token_id + 1) to ensure row alignment:
    # row k in the matrix must correspond to token ID k-1.
    # Without sorting, dict insertion order (GloVe file order)
    # shuffles embeddings relative to token IDs.
    sorted_embeddings = [embeddings_dict[k] for k in sorted(embeddings_dict.keys())]

    return np.vstack(
        (
            np.zeros(cfg.emb_length),
            sorted_embeddings,
            np.random.randn(cfg.emb_length),
        )
    )
