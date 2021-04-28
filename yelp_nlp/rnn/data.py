import numpy as np
import pandas as pd
import jsonlines as jsonl
import zipfile

from gensim import corpora
import os
import re

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_embeddings(
    emb_path: str,
    emb_file: str = 'glove.6B.zip',
    emb_subfile: str = 'glove.6B.100d.txt'
) -> dict:

    """load glove vectors"""

    embeddings_index = {}

    with zipfile.ZipFile(
        os.path.join(
            os.path.expanduser("~"),
            emb_path,
            emb_file
        ),
        'r'
    ) as f:
        with f.open(emb_subfile, 'r') as z:
            for line in z:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float16')
                embeddings_index[word] = coefs

    return embeddings_index


def id_to_glove(
    dictionary: corpora.Dictionary,
    emb_path: str,
    emb_n: int = 100
):

    """converts local dictionary to embeddings from glove"""

    embeddings_index = load_embeddings(emb_path)
    conversion_table = {}

    for word in dictionary.values():
        if bytes(word, 'utf-8') in embeddings_index.keys():
            conversion_table[dictionary.token2id[word]+1]\
                = embeddings_index[bytes(word, 'utf-8')]
        else:
            conversion_table[dictionary.token2id[word]+1]\
                = np.random.normal(0, .32, emb_n)

    return np.vstack(
        (
            np.zeros(emb_n),
            list(conversion_table.values()),
            np.random.randn(emb_n)
        )
    )


def convert_rating(rating: int) -> float:

    """scaling ratings from 0 to 1"""

    if rating in [4, 5]:
        return 1.0
    elif rating in [1, 2]:
        return 0.0
    else:
        return


def convert_rating_linear(rating: int, max_rating: int) -> float:

    """scaling ratings from 0 to 1 linearly"""
    return rating/max_rating


def text_sequencer(dictionary: corpora.Dictionary, text, max_len=200):

    """converts tokens to numeric representation by dictionary"""

    processed = np.zeros(max_len, dtype=int)
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
def tokenize(x: str) -> list: return re.findall(r'\w+', x.lower())


def load_data(path: str, fname: str, stop: int = None) -> list:
    "reads from zipped yelp data file"
    ls = []

    with zipfile.ZipFile(
        os.path.join(
            os.path.expanduser('~'),
            path
        )
    ) as zfile:
        print(f'archive contains the following: {zfile.namelist()}')
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
        max_len: int,
        dictionary: corpora.Dictionary = None,
        stop: int = None,
        data: str = 'data',
        labels: str = 'target',
        mode: str = 'training',
        test_size=0.25
    ):

        super().__init__()
        self.fpath = fpath
        self.fname = fname
        self.stop = stop
        self.data = data
        self.labels = labels
        self.mode = mode
        if dictionary:
            self.dict_yelp = dictionary
        else:
            self.dict_yelp = None
        self.df: pd.DataFrame = self.parse_data(fpath, fname, stop, max_len)
        self.test_size = test_size
        self.tr_idx: list = None
        self.val_idx: list = None
        self.split_df()
        self.train = None
        self.val = None

    def parse_data(self, fpath, fname, stop, max_len):

        df = pd.DataFrame(load_data(fpath, fname, stop))

        print('df loaded..')

        if self.dict_yelp is None:
            self.dict_yelp = corpora.Dictionary(df.text)
            self.dict_yelp.filter_extremes(
                no_below=10,
                no_above=.99,
                keep_n=10000
            )

        print('dictionary created...')

        df[self.data] = df.text.apply(
            lambda x: text_sequencer(self.dict_yelp, x, max_len)
        )
        df[self.labels] = df.stars.apply(convert_rating)

        print('converted tokens to numbers...')

        return df.loc[
            df[self.labels].dropna().index,
            [self.data, self.labels]
        ].reset_index(drop=True)

    def set_mode(self, mode: str):

        if mode not in ['fitting', 'training', 'eval']:
            raise ValueError('not an available mode')

        self.mode = mode

    def __len__(self):

        if self.mode == 'fitting':
            return self.df.__len__()
        if self.mode == 'training':
            return self.train.__len__()
        if self.mode == 'eval':
            return self.val.__len__()

    def __getitem__(self, i):
        if self.mode == 'fitting':
            return self.df[self.data].iat[i], self.df[self.labels][i]
        elif self.mode == 'training':
            return self.train[self.data].iat[i], self.train[self.labels].iat[i]
        elif self.mode == 'eval':
            return self.val[self.data].iat[i], self.val[self.labels].iat[i]

    def split_df(self):

        self.tr_idx, self.val_idx = train_test_split(
            self.df.index.values,
            test_size=self.test_size
            )
        return self

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = self.df.iloc[self.tr_idx]

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        self._val = self.df.iloc[self.val_idx]
