import tarfile
import zipfile
from collections.abc import Generator
from itertools import islice
from typing import IO

import numpy as np
import orjson as json
import ray
from gensim import corpora
from gensim import downloader as gensim_api

from sentimentizer import new_logger, time_decorator
from sentimentizer.config import (
    BATCH_SIZE,
    DEFAULT_LOG_LEVEL,
    EMBEDDING_DTYPE,
    EMBEDDING_RANDOM_MEAN,
    EMBEDDING_RANDOM_STD,
    EXTRACT_LOG_INTERVAL,
    EmbeddingsConfig,
)
from sentimentizer.tokenizer import regex_tokenize

logger = new_logger(DEFAULT_LOG_LEVEL)


def generate_batch(
    generator_input: Generator[dict, str, None], iter_size: int
) -> Generator[tuple[list, int, int], None, None]:
    for start in range(0, iter_size, BATCH_SIZE):
        end = min(start + BATCH_SIZE, iter_size)
        review_dicts: list[dict] = []
        review_dicts.extend(islice(generator_input, BATCH_SIZE))
        yield review_dicts, start, end


def process_json(json_file: IO[bytes], stop: int = 0) -> Generator:
    for i, line in enumerate(json_file):
        if i % EXTRACT_LOG_INTERVAL == 0:
            logger.debug(f"processing line {i}")
        dc = json.loads(line)
        if i >= stop and stop != 0:
            break
        yield dc


@time_decorator
def extract_data(file_path: str, compressed_file_name: str, stop: int = 0) -> ray.data.Dataset:
    """Reads from zipped or tarred yelp data file.

    Supports both .zip and .tar/.tar.gz archives.
    """

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
