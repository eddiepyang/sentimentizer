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
            "text": [("the chicken never showed up"), ("the food was terrific")],
            "stars": [2, 5],
        }
    )


@pytest.fixture
def mock_parser(mock_df) -> DataParser:
    return DataParser(mock_df, 10, 200)


def test_convert_rating():

    assert 1 == convert_rating(5)
    assert 0 == convert_rating(1)
    assert convert_rating(3) == -1


def test_tokenize(mock_df):

    output = tokenize(mock_df.text[0])

    for item in output:
        assert isinstance(item, str)

    assert len(output) > 3


class TestLoadData:
    fpath = f"{root}/tests/test_data/archive.zip"
    fname = "artificial-reviews.jsonl"

    def test_success(self):
        df = load_data(fname=self.fname, path=self.fpath)
        assert df.shape == (2, 2)
        print(df.columns)
        assert df.columns.tolist() == ["text", "stars"]

    def test_failure_empty_input(self):
        # todo
        return


class TestDataParser:
    def test_success(self, mock_df):
        parser = DataParser(mock_df, 10, 200)
        df = parser.convert_sentences()
        print(df.shape, df.columns)

    def test_failure(self):
        # todo
        return


class TestCorpusDataset:
    def test_success(self, mock_df):
        dataset = CorpusDataset(mock_df, "text", "stars")

        assert len(dataset[1]) == 2
        assert len(dataset) == 2
