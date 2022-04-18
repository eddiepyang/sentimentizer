import unittest
from unittest.mock import Mock
import pytest
import pandas as pd

from yelp_nlp.rnn.data import (
    CorpusDataset,
    DataParser,
    FitModes,
    tokenize,
    load_data,
    load_embeddings,
    convert_rating,
)
from yelp_nlp import root


@pytest.fixture
def mock_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "the chicken never showed up".split(),
                "the food was terrific".split(),
            ],
            "stars": [2, 5],
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "the chicken never showed up",
                "the food was terrific",
            ],
            "stars": [2, 5],
        }
    )


@pytest.fixture
def finished_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data": [
                (1, 2, 3, 4, 5, 6),
                (5, 6, 7, 7, 8, 19),
            ],
            "target": [2, 5],
        }
    )


@pytest.fixture
def mock_parser(mock_df) -> DataParser:
    return DataParser(mock_df)


def test_convert_rating():

    assert 1 == convert_rating(5)
    assert 0 == convert_rating(1)
    assert convert_rating(3) == -1


def test_tokenize(raw_df):

    output = tokenize(raw_df.text[0])

    for item in output:
        assert isinstance(item, str)

    assert len(output) > 3


class TestLoadData:
    fpath = f"{root}/yelp_nlp/tests/test_data/archive.zip"
    fname = "artificial-reviews.jsonl"

    def test_success(self):
        df = load_data(compressed_file_name=self.fname, file_path=self.fpath)
        assert df.shape == (2, 2)
        print(df.columns)
        assert df.columns.tolist() == ["text", "stars"]

    def test_failure_empty_input(self):
        # todo
        return


class TestDataParser:
    mock_dictionary = Mock()

    def test_success(self, mock_df):
        parser = DataParser(mock_df)
        parser.convert_sentences()
        assert parser.df.shape == (2, 4)

    def test_failure(self):
        # todo
        return


class TestCorpusDataset:
    def test_success(self, finished_df):
        dataset = CorpusDataset(finished_df)
        item = dataset[1]
        assert len(item) == 2
        assert len(dataset) == 2
