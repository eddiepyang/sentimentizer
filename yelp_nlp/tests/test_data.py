import pytest
import pandas as pd

from yelp_nlp.rnn.data import FitModes, tokenize


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [("the chicken never showed up"), ("the food was terrific")],
            "stars": [2, 5],
        }
    )


def test_tokenize(test_df):

    output = tokenize(test_df.text[0])

    for item in output:
        assert isinstance(item, str)

    assert len(output) > 3
