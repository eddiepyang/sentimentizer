import pytest  # noqa: F401
from yelp_nlp.rnn.data import convert_rating


def test_convert_rating():

    assert 1 == convert_rating(5)
    assert 0 == convert_rating(1)
    assert convert_rating(3) is None
