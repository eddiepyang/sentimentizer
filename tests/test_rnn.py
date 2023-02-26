import pandas as pd
import polars as pl
import pytest

from torch_sentiment.tokenizer import Tokenizer, convert_rating, new_logger, tokenize
from torch_sentiment.config import DEFAULT_LOG_LEVEL
from torch_sentiment.extractor import extract_data, write_arrow
from torch_sentiment.loader import CorpusDataset
from torch_sentiment.models.rnn import RNN, get_trained_model
from torch_sentiment.tokenizer import get_trained_tokenizer

logger = new_logger(DEFAULT_LOG_LEVEL)


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
    stop = 2

    def test_success(self, rel_path, relative_root):

        gen = extract_data(
            compressed_file_name=self.fname, file_path=rel_path, stop=self.stop
        )
        write_arrow(gen, self.stop, f"{relative_root}/tests/test_data/file.arrow")
        df = pl.read_ipc(f"{relative_root}/tests/test_data/file.arrow")
        assert df.shape == (2, 2)
        logger.info(str(df.shape))

    def test_failure_empty_input(self):
        # todo
        return


class TestDataTokenizer:
    def test_success(self, tokenized_df):
        parser = Tokenizer.from_data(tokenized_df)
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
        model = get_trained_model(64, "cpu")
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


class TestTokenize:
    """tests regex"""

    def test_success(self):
        result = tokenize("chicken wasn't good")

        assert len(result) == 3
        assert result[0] == "chicken"
        assert result[1] == "wasn't"

    def test_success_one(self):
        result = tokenize("1st place food")
        assert len(result) == 3
        assert result[0] == "1st"
