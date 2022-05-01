from unittest.mock import Mock
import pytest
import pandas as pd

from torch_sentiment.rnn.loader import CorpusDataset
from torch_sentiment.rnn.transformer import (
    DataTransformer,
    tokenize,
    convert_rating,
)
from torch_sentiment.rnn.extractor import extract_data
from torch_sentiment import root


@pytest.fixture
def tokenized_df() -> pd.DataFrame:
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
def processed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data": [
                (1, 2, 3, 4, 5, 6),
                (5, 6, 7, 7, 8, 19),
            ],
            "target": [2, 5],
        }
    )


def test_convert_rating():

    assert 1 == convert_rating(5)
    assert 0 == convert_rating(1)
    assert convert_rating(3) == -1


def test_tokenize(raw_df):

    output = tokenize(raw_df.text[0])

    for item in output:
        assert isinstance(item, str)

    assert len(output) > 3


class TestExtractData:
    fpath = f"{root}/tests/test_data/archive.zip"
    fname = "artificial-reviews.jsonl"

    def test_success(self):
        df = extract_data(compressed_file_name=self.fname, file_path=self.fpath)
        assert df.shape == (2, 2)
        print(df.columns)
        assert df.columns.tolist() == ["text", "stars"]

    def test_failure_empty_input(self):
        # todo
        return


class TestDataParser:
    mock_dictionary = Mock()

    def test_success(self, tokenized_df):
        parser = DataTransformer(tokenized_df)
        parser.transform_sentences()
        assert parser.df.shape == (2, 4)

    def test_failure(self):
        # todo
        return


class TestCorpusDataset:
    def test_success(self, processed_df):
        dataset = CorpusDataset(processed_df)
        item = dataset[1]
        assert len(item) == 2
        assert len(dataset) == 2

    def test_failure(self):
        # todo
        return
