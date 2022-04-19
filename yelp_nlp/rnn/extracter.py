import numpy as np
import zipfile
from gensim import corpora

from yelp_nlp.rnn.config import LogLevels
from yelp_nlp.logging_utils import new_logger, time_decorator

logger = new_logger(LogLevels.debug.value)


def load_embeddings(
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


def id_to_glove(
    dictionary: corpora.Dictionary, emb_path: str, emb_n: int = 100
) -> np.ndarray:

    """converts local dictionary to embeddings from glove"""

    embeddings_index = load_embeddings(emb_path)
    conversion_table = {}

    for word in dictionary.values():
        if word in embeddings_index:
            conversion_table[dictionary.token2id[word] + 1] = embeddings_index[word]
        else:
            conversion_table[dictionary.token2id[word] + 1] = np.random.normal(
                0, 0.32, emb_n
            )
    return np.vstack(
        (np.zeros(emb_n), list(conversion_table.values()), np.random.randn(emb_n))
    )
