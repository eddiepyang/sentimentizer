import tarfile
import zipfile
from collections.abc import Generator
from itertools import islice
from typing import IO

import numpy as np
import orjson as json
import ray
from gensim import corpora

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
) -> dict[str, np.ndarray]:
    """load glove vectors"""

    embeddings_dict: dict = {}

    with zipfile.ZipFile(cfg.file_path, "r") as f, f.open(cfg.sub_file_path, "r") as z:
        for line in z:
            values = line.split()
            key = values[0].decode()

            if key in dictionary.token2id:
                embeddings_dict.setdefault(
                    dictionary.token2id[key] + 1,
                    np.asarray(values[1:], dtype=EMBEDDING_DTYPE),
                )

    return embeddings_dict


@time_decorator
def new_embedding_weights(dictionary: corpora.Dictionary, cfg: EmbeddingsConfig) -> np.ndarray:
    """converts local dictionary to embeddings from glove"""

    embeddings_dict: dict = extract_embeddings(dictionary, cfg)

    for word in dictionary.values():
        if word not in embeddings_dict:
            embeddings_dict.setdefault(
                dictionary.token2id[word] + 1,
                np.random.normal(EMBEDDING_RANDOM_MEAN, EMBEDDING_RANDOM_STD, cfg.emb_length),
            )

    return np.vstack(
        (
            np.zeros(cfg.emb_length),
            list(embeddings_dict.values()),
            np.random.randn(cfg.emb_length),
        )
    )
