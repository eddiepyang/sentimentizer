import pytest
import pandas as pd

from torch_sentiment.rnn.config import TokenizerConfig
from torch_sentiment.rnn.loader import CorpusDataset
from torch_sentiment.rnn.tokenizer import (
    Tokenizer,
    tokenize,
    convert_rating,
)
from torch_sentiment.rnn.extractor import extract_data
from torch_sentiment.logging_utils import new_logger
from torch_sentiment.rnn.model import get_trained_model, RNN
from torch_sentiment.rnn.tokenizer import get_trained_tokenizer

logger = new_logger()


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
    assert convert_rating(3) == 0.5


def test_tokenize(raw_df):

    output = tokenize(raw_df.text[0])

    for item in output:
        assert isinstance(item, str)

    assert len(output) > 3


class TestExtractData:

    fname = "artificial-reviews.jsonl"

    def test_success(self, rel_path):

        df = extract_data(compressed_file_name=self.fname, file_path=rel_path)
        assert df.shape == (2, 2)
        logger.info(df.shape)
        assert df.columns.tolist() == ["text", "stars"]

    def test_failure_empty_input(self):
        # todo
        return


class TestDataTransformer:
    def test_success(self, tokenized_df):
        parser = Tokenizer(tokenized_df, TokenizerConfig(save_dictionary=False))
        parser.transform_dataframe(tokenized_df)
        assert tokenized_df.shape == (2, 4)

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


class TestGetTrainedModel:
    """tests if model loads"""

    def test_success(self):
        model = get_trained_model()
        assert isinstance(model, RNN)

    def test_failure(self):
        # todo
        return


class TestGetTrainedTokenizer:
    """tests if model loads"""

    def test_success(self):
        tokenizer = get_trained_tokenizer()
        assert isinstance(tokenizer, Tokenizer)

    def test_failure(self):
        # todo
        return
