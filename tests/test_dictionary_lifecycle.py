"""Tests for dictionary lifecycle: building, saving, and loading.

These tests verify that:
1. Tokenizer.from_dataset saves the dictionary when save_dictionary=True
2. The saved dictionary matches the one used for tokenization (no mismatch)
3. Tokenizer.from_dataset does NOT save when save_dictionary=False
4. An update run loads and uses the existing dictionary
5. _build_dictionary_distributed produces deterministic token IDs
"""

import os
import tempfile

import pytest

# ruff: noqa: E402
ray = pytest.importorskip("ray")
from gensim import corpora

from sentimentizer.config import DriverConfig, TokenizerConfig
from sentimentizer.tokenizer import Tokenizer, _build_dictionary_distributed, _count_vocab_batch


@pytest.fixture(scope="module")
def ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = DriverConfig.files.dictionary_file_path
        tmp_dict_path = os.path.join(tmpdir, "yelp.dictionary")
        DriverConfig.files.dictionary_file_path = tmp_dict_path

        original_processed_path = DriverConfig.files.processed_reviews_file_path
        DriverConfig.files.processed_reviews_file_path = os.path.join(tmpdir, "review_data.parquet")

        yield tmpdir

        DriverConfig.files.dictionary_file_path = original_path
        DriverConfig.files.processed_reviews_file_path = original_processed_path


# ---------------------------------------------------------------------------
# Unit tests for _count_vocab_batch
# ---------------------------------------------------------------------------


class TestCountVocabBatch:
    """Test the _count_vocab_batch helper function."""

    def test_basic_counting(self):
        """_count_vocab_batch should count word frequencies correctly."""
        batch = {
            "tokens": [
                ["hello", "world", "good"],
                ["hello", "world", "bad"],
                ["hello", "world", "good"],
            ],
            "stars": [5, 1, 4],
        }
        word_freq, doc_freq, num_docs = _count_vocab_batch(batch, "tokens")

        assert num_docs == 3
        assert word_freq["hello"] == 3
        assert word_freq["world"] == 3
        assert word_freq["good"] == 2
        assert word_freq["bad"] == 1
        assert doc_freq["hello"] == 3
        assert doc_freq["good"] == 2

    def test_empty_batch(self):
        """_count_vocab_batch should handle empty batches."""
        batch = {"tokens": [], "stars": []}
        word_freq, doc_freq, num_docs = _count_vocab_batch(batch, "tokens")

        assert num_docs == 0
        assert len(word_freq) == 0

    def test_numpy_array_tokens(self):
        """_count_vocab_batch should handle numpy array tokens without stringifying.

        Parquet stores list columns as numpy arrays of strings. When each
        row in the batch is a numpy array (not a Python list), the function
        must convert it to a list of clean strings, NOT call str() which
        would produce "['word1' 'word2' ...]" with wrapping quotes.
        """
        import numpy as np

        batch = {
            "tokens": np.array(
                [
                    np.array(["good", "food", "great"]),
                    np.array(["bad", "service"]),
                    np.array(["good", "place"]),
                ],
                dtype=object,
            ),
            "stars": [5, 1, 4],
        }
        word_freq, doc_freq, num_docs = _count_vocab_batch(batch, "tokens")

        assert num_docs == 3
        assert word_freq["good"] == 2
        assert word_freq["food"] == 1
        assert word_freq["bad"] == 1
        # Ensure no tokens have wrapping quotes (the old bug)
        for word in word_freq:
            assert not word.startswith("'"), f"Token {word!r} has a leading quote"
            assert not word.endswith("'"), f"Token {word!r} has a trailing quote"

    def test_pandas_series_tokens(self):
        """_count_vocab_batch should handle pandas-series-style token arrays.

        When iterating over a pandas DataFrame column, each row may be a
        numpy array rather than a Python list.
        """
        import numpy as np

        doc_tokens = np.array(["the", "food", "was", "good"], dtype=object)
        batch = {
            "tokens": [doc_tokens, np.array(["bad", "service"], dtype=object)],
            "stars": [5, 1],
        }
        word_freq, doc_freq, num_docs = _count_vocab_batch(batch, "tokens")

        assert num_docs == 2
        assert word_freq["good"] == 1
        assert word_freq["bad"] == 1
        assert "the" in word_freq
        assert "food" in word_freq
        # No wrapping quotes
        for word in word_freq:
            assert not word.startswith("'"), f"Token {word!r} has a leading quote"
        assert len(word_freq) == 6


# ---------------------------------------------------------------------------
# Dictionary save/load alignment tests
# ---------------------------------------------------------------------------


def test_from_dataset_saves_dictionary_by_default(ray_init, temp_data_dir):
    """Tokenizer.from_dataset should save the dictionary to disk by default."""
    # Create a dataset with enough data to pass dict_min=3 filter.
    # Use string tokens that will be split by _count_vocab_batch.
    # Ray Data with numpy format handles string columns properly.
    rows = [
        {"tokens": "hello world good", "stars": 5},
        {"tokens": "hello world bad", "stars": 1},
        {"tokens": "hello world good", "stars": 4},
    ] * 10  # Repeat to ensure words appear in enough documents
    ds = ray.data.from_items(rows)

    dict_path = DriverConfig.files.dictionary_file_path
    if os.path.exists(dict_path):
        os.remove(dict_path)

    # Use dict_min=1 so words survive the filter even with few unique docs
    cfg = TokenizerConfig(save_dictionary=True, dict_min=1, dict_keep=1000)
    tokenizer = Tokenizer.from_dataset(ds, cfg=cfg)

    # Dictionary should have been saved to disk
    assert os.path.exists(dict_path), "Dictionary was not saved to disk"

    # Load and verify the saved dictionary matches the in-memory one
    saved_dict = corpora.Dictionary.load(dict_path)
    assert len(saved_dict) == len(tokenizer.dictionary), (
        f"Saved dictionary has {len(saved_dict)} terms, "
        f"but in-memory has {len(tokenizer.dictionary)}"
    )

    # Verify token2id mappings are identical
    for word, token_id in tokenizer.dictionary.token2id.items():
        assert saved_dict.token2id[word] == token_id, (
            f"Token ID mismatch for word {word!r}: "
            f"saved={saved_dict.token2id[word]}, in-memory={token_id}"
        )


def test_from_dataset_no_save_when_disabled(ray_init, temp_data_dir):
    """Tokenizer.from_dataset with save_dictionary=False should NOT save."""
    rows = [
        {"tokens": "hello world good", "stars": 5},
        {"tokens": "hello world bad", "stars": 1},
        {"tokens": "hello world good", "stars": 4},
    ] * 10
    ds = ray.data.from_items(rows)

    dict_path = DriverConfig.files.dictionary_file_path
    if os.path.exists(dict_path):
        os.remove(dict_path)

    cfg = TokenizerConfig(save_dictionary=False, dict_min=1, dict_keep=1000)
    tokenizer = Tokenizer.from_dataset(ds, cfg=cfg)

    # Dictionary should NOT have been saved to disk
    assert not os.path.exists(dict_path), (
        "Dictionary was saved to disk even though save_dictionary=False"
    )

    # But the in-memory dictionary should still have words
    assert len(tokenizer.dictionary) > 0, "In-memory dictionary is empty"


def test_dictionary_alignment_after_save_and_load(ray_init, temp_data_dir):
    """Verify no token ID mismatch between tokenization and embedding matrix.

    This is the core bug fix: when from_dataset saves the dictionary,
    new_model() loads the same dictionary for the embedding matrix,
    ensuring token IDs align.
    """
    rows = [
        {"tokens": "hello world good", "stars": 5},
        {"tokens": "hello world bad", "stars": 1},
        {"tokens": "hello world good", "stars": 4},
    ] * 10
    ds = ray.data.from_items(rows)

    dict_path = DriverConfig.files.dictionary_file_path
    if os.path.exists(dict_path):
        os.remove(dict_path)

    cfg = TokenizerConfig(save_dictionary=True, dict_min=1, dict_keep=1000)
    tokenizer = Tokenizer.from_dataset(ds, cfg=cfg)

    # Load the saved dictionary (simulating what new_model does)
    loaded_dict = corpora.Dictionary.load(dict_path)

    # Every word should have the same token ID in both dictionaries
    mismatches = 0
    for word, token_id in tokenizer.dictionary.token2id.items():
        if loaded_dict.token2id.get(word) != token_id:
            mismatches += 1

    assert mismatches == 0, (
        f"Found {mismatches} token ID mismatches between in-memory and "
        f"saved dictionary — embedding matrix will be misaligned"
    )


def test_update_run_loads_existing_dictionary(ray_init, temp_data_dir):
    """In an update run (type='update'), the existing dictionary is loaded and used."""
    dict_path = DriverConfig.files.dictionary_file_path

    # 1. Create and save an initial dictionary with known words
    initial_dict = corpora.Dictionary([["initial", "words", "hello"]])
    initial_dict.save(dict_path)
    initial_id = initial_dict.token2id["hello"]

    # 2. Load it via Tokenizer (simulating type='update' in run_tokenize)
    loaded_dict = corpora.Dictionary.load(dict_path)
    tokenizer = Tokenizer(dictionary=loaded_dict)

    # The loaded dictionary should contain the initial words
    assert "initial" in tokenizer.dictionary.token2id
    assert "hello" in tokenizer.dictionary.token2id
    assert tokenizer.dictionary.token2id["hello"] == initial_id, (
        "Token IDs should be preserved when loading an existing dictionary"
    )


def test_build_dictionary_deterministic_ordering(ray_init, temp_data_dir):
    """_build_dictionary_distributed should produce deterministic token IDs.

    Regardless of batch processing order, the same input data should
    always produce the same token-to-ID mapping.
    """
    rows = [
        {"tokens": "alpha beta gamma", "stars": 5},
        {"tokens": "alpha beta delta", "stars": 1},
        {"tokens": "alpha beta gamma", "stars": 4},
    ] * 10
    ds = ray.data.from_items(rows)

    cfg = TokenizerConfig(save_dictionary=False, dict_min=1, dict_keep=1000)

    # Build dictionary twice — should produce identical token2id mappings
    dict1 = _build_dictionary_distributed(ds, cfg)
    dict2 = _build_dictionary_distributed(ds, cfg)

    assert len(dict1) == len(dict2), (
        f"Dictionaries have different sizes: {len(dict1)} vs {len(dict2)}"
    )

    for word in dict1.token2id:
        assert dict1.token2id[word] == dict2.token2id[word], (
            f"Token ID mismatch for {word!r}: {dict1.token2id[word]} vs {dict2.token2id[word]}"
        )
