import zipfile
from itertools import islice
from typing import IO, Generator

import numpy as np
import orjson as json
import pyarrow as pa
from gensim import corpora

from sentimentizer import new_logger, time_decorator
from sentimentizer.config import DEFAULT_LOG_LEVEL, EmbeddingsConfig
from sentimentizer.tokenizer import regex_tokenize

logger = new_logger(DEFAULT_LOG_LEVEL)

import ray

BATCH_SIZE = 100000


def process_json(
    json_file: IO[bytes],
    stop: int = 0,
    text_field: str = "text",
    tokenize_func: callable[str, list] = None,
) -> Generator:
    for i, line in enumerate(json_file):
        if i % 100000 == 0:
            logger.debug(f"processing line {i}")
        dc = json.loads(line)
        if tokenize_func:
            dc["tokens"] = tokenize_func(dc.get(text_field))
        if i >= stop and stop != 0:
            break
        yield dc


@time_decorator
def extract_data(file_path: str, compressed_file_name: str, stop: int = 0) -> ray.data.Dataset:
    "reads from zipped yelp data file"

    def generate_lines():
        with zipfile.ZipFile(file_path) as zfile:
            inf = zfile.open(compressed_file_name)
            yield from process_json(inf, stop, tokenize_func=None)

    ds = ray.data.from_generators([generate_lines])
    
    def tokenize(row):
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
                    np.asarray(values[1:], dtype=np.float32),  # noqa: E501
                )

    return embeddings_dict


@time_decorator
def new_embedding_weights(
    dictionary: corpora.Dictionary, cfg: EmbeddingsConfig
) -> np.ndarray:
    """converts local dictionary to embeddings from glove"""

    embeddings_dict: dict = extract_embeddings(dictionary, cfg)

    for word in dictionary.values():
        if word not in embeddings_dict:
            embeddings_dict.setdefault(
                dictionary.token2id[word] + 1, np.random.normal(0, 0.32, cfg.emb_length)
            )

    return np.vstack(
        (
            np.zeros(cfg.emb_length),
            list(embeddings_dict.values()),
            np.random.randn(cfg.emb_length),
        )
    )
