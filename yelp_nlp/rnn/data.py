import numpy as np
import pandas as pd
import jsonlines as jsonl
import zipfile
from gensim import corpora
from os.path import expanduser
import re
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_embeddings(emb_path: str) -> dict:
    """load glove vectors"""

    embeddings_index = {}

    with zipfile.ZipFile(
        expanduser("~") + emb_path + 'glove.6B.zip', 
        'r'
    ) as f:
        with f.open('glove.6B.100d.txt', 'r') as z:
            for line in z:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float16')
                embeddings_index[word] = coefs

    return embeddings_index


def id_to_glove(dict_yelp: corpora.Dictionary, emb_path: str):

    """converts local dictionary to embeddings from glove"""

    embeddings_index = load_embeddings(emb_path)
    conversion_table = {}

    for word in dict_yelp.values():
        if bytes(word, 'utf-8') in embeddings_index.keys():
            conversion_table[dict_yelp.token2id[word]+1]\
                = embeddings_index[bytes(word, 'utf-8')]
        else:
            conversion_table[dict_yelp.token2id[word]+1]\
                = np.random.normal(0, .32, 100)

    embedding_matrix = np.vstack(
        (
            np.zeros(100),
            list(conversion_table.values()),
            np.random.randn(100)
        )
    )

    return embedding_matrix


def convert_rating(rating):

    """moving ratings from 0 to 1"""

    if rating in [4, 5]:
        return 1
    elif rating in [1, 2]:
        return 0


def text_sequencer(dictionary, text, max_len=200):

    """converts tokens to numeric representation by dictionary"""

    processed = np.zeros(200, dtype=int)
    # in case the word is not in the dictionary because it was
    # filtered out use this number to represent an out of set id
    dict_final = len(dictionary.keys()) + 1

    for i, word in enumerate(text):
        if i >= max_len:
            return processed
        if word in dictionary.token2id.keys():
            # the ids have an offset of 1 for this because 
            # 0 represents a padded value
            processed[i] = dictionary.token2id[word] + 1
        else:
            processed[i] = dict_final

    return processed


# regex tokenize, less accurate
def tokenize(x): return re.findall(r'\w+', x.lower())


def load_data(path: str, fname: str, stop: int = None) -> list:
    "reads from zipped yelp data file"
    ls = []
    with zipfile.ZipFile(path) as zfile:
        print(zfile.namelist())
        inf = zfile.open(fname)
        with jsonl.Reader(inf) as file:
            for i, line in enumerate(file):
                line['text'] = tokenize(line.get('text'))
                ls.append(line)
                if stop and i == stop-1:
                    break
    return ls


class CorpusData(Dataset):

    """Dataset class required for pytorch to output items by index"""

    def __init__(
        self,
        fpath: str,
        fname: str,
        stop: int = None,
        data: str = 'data',
        labels: str = 'target',
        test_size=0.25
    ):

        super().__init__()
        self.fpath: str
        self.fname: str
        self.stop: int = None
        self.data = data
        self.labels = labels
        self.dict_yelp: corpora.Dictionary = None
        self.df: pd.DataFrame = self.parse_data(fpath, fname, stop)
        self.test_size: float = test_size
        self.tr_idx: list = None
        self.val_idx: list = None
        self.split_df()

    def parse_data(self, fpath, fname, stop):

        df = pd.DataFrame(load_data(fpath, fname, stop))
        print('df loaded..')
        self.dict_yelp = corpora.Dictionary(df.text)
        self.dict_yelp.filter_extremes(no_below=10, no_above=.95, keep_n=30000)
        print('dictionary created...')
        df[self.data] = df.text.apply(
            lambda x: text_sequencer(self.dict_yelp, x)
            )
        df[self.labels] = df.stars.apply(convert_rating)

        return df.loc[df[self.labels].dropna()].reset_index(drop=True)

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, i):

        return self.df[self.data][i], self.df[self.labels][i]

    def split_df(self):

        self.tr_idx, self.val_idx = train_test_split(
            self.df.index.values,
            test_size=self.test_size
            )
        return self

    @property
    def train(self):
        return self.df.iloc[self.tr_idx]

    @property
    def val(self):
        return self.df.iloc[self.val_idx]