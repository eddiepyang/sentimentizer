import numpy as np
import pandas as pd
import jsonlines as jsonl
import zipfile
from gensim import corpora
from collections import OrderedDict

from torch_sentiment.rnn.tokenizer import tokenize
from torch_sentiment.rnn.config import EmbeddingsConfig, LogLevels
from torch_sentiment.logging_utils import new_logger, time_decorator

logger = new_logger(LogLevels.debug.value)


@time_decorator
def extract_data(
    file_path: str, compressed_file_name: str, stop: int = 0
) -> pd.DataFrame:
    "reads from zipped yelp data file"
    ls = []

    with zipfile.ZipFile(file_path) as zfile:
        logger.info(f"archive contains the following: {zfile.namelist()}")
        inf = zfile.open(compressed_file_name)

        with jsonl.Reader(inf) as file:
            for i, line in enumerate(file):
                if i % 100000 == 0:
                    logger.debug(f"processing line {i}")
                line["text"] = tokenize(line.get("text"))
                ls.append(line)
                if i == stop - 1:
                    break

    return pd.DataFrame(ls)


@time_decorator
def extract_embeddings(dictionary: corpora.Dictionary, cfg: EmbeddingsConfig) -> dict[str, np.ndarray]:
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
                dictionary.token2id[word] + 1, 
                np.random.normal(0, 0.32, cfg.emb_length)
            )

    return np.vstack(
        (
            np.zeros(cfg.emb_length),
            list(embeddings_dict.values()),
            np.random.randn(cfg.emb_length),
        )
    )
