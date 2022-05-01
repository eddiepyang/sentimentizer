import numpy as np
import pandas as pd
import jsonlines as jsonl
import zipfile
from gensim import corpora

from torch_sentiment.rnn.transformer import tokenize
from torch_sentiment.rnn.config import LogLevels
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
                line["text"] = tokenize(line.get("text"))
                ls.append(line)
                if i == stop - 1:
                    break

    return pd.DataFrame(ls)


def extract_embeddings(
    emb_path: str,
    emb_subfile: str = "glove.6B.100d.txt",
) -> dict:

    """load glove vectors"""

    embeddings_index = {}

    with zipfile.ZipFile(emb_path, "r") as f:
        with f.open(emb_subfile, "r") as z:
            for line in z:
                values = line.split()
                embeddings_index[values[0].decode()] = np.asarray(
                    values[1:], dtype=np.float32
                )  # noqa: E501

    return embeddings_index


@time_decorator
def id_to_glove(
    dictionary: corpora.Dictionary, emb_path: str, emb_length: int = 100
) -> np.ndarray:

    """converts local dictionary to embeddings from glove"""

    embeddings_index = extract_embeddings(emb_path)
    conversion_table = {}

    for word in dictionary.values():
        if word in embeddings_index:
            conversion_table[dictionary.token2id[word] + 1] = embeddings_index[word]
        else:
            conversion_table[dictionary.token2id[word] + 1] = np.random.normal(
                0, 0.32, emb_length
            )
    return np.vstack(
        (
            np.zeros(emb_length),
            list(conversion_table.values()),
            np.random.randn(emb_length),
        )
    )
