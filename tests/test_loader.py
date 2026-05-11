import os
import tempfile

import numpy as np
import pandas as pd
import torch

from sentimentizer.config import TokenizerConfig
from sentimentizer.loader import CorpusDataset, compute_pos_weight, load_train_val_corpus_datasets
from sentimentizer.tokenizer import new_dictionary, text_sequencer


def test_load_train_val_corpus_datasets():
    """Test loading and splitting of parquet data into train/val datasets."""
    # Create dummy data
    data = {"data": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], "target": [1.0, 0.0, 1.0, 0.0, 1.0]}
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_data.parquet")
        df.to_parquet(data_path)

        # Test basic loading and splitting
        train_ds, val_ds = load_train_val_corpus_datasets(
            data_path, test_size=0.4, balance_classes=False, random_state=42
        )

        assert isinstance(train_ds, CorpusDataset)
        assert isinstance(val_ds, CorpusDataset)

        # With test_size=0.4 on 5 samples:
        # val should have 2 samples, train should have 3
        assert len(val_ds) == 2
        assert len(train_ds) == 3

        # Verify content type (tensors)
        x, y = train_ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.long
        assert y.dtype == torch.float32


def test_load_train_val_corpus_datasets_balancing():
    """Test that class balancing correctly undersamples the majority class."""
    # Create imbalanced dummy data: 80 positive, 20 negative
    data = {"data": [[i, i] for i in range(100)], "target": [1.0] * 80 + [0.0] * 20}
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "imbalanced_data.parquet")
        df.to_parquet(data_path)

        train_ds, val_ds = load_train_val_corpus_datasets(
            data_path, test_size=0.2, balance_classes=True, random_state=42
        )

        # Verify classes are balanced in train_ds
        targets = [train_ds[i][1].item() for i in range(len(train_ds))]
        assert targets.count(1.0) == targets.count(0.0)
        assert len(train_ds) > 0


# ─── Dictionary building with numpy arrays ─────────────────────────


class TestDictionaryNumpyArrays:
    """Test that dictionary building handles numpy array tokens correctly.

    Parquet stores list columns as numpy arrays of strings. The dictionary
    builder must convert these to Python lists rather than calling str()
    on them, which would produce tokens with wrapping quotes like "'the'".
    """

    def test_new_dictionary_with_numpy_tokens(self) -> None:
        """new_dictionary should handle DataFrame columns containing numpy arrays.

        When a DataFrame's 'tokens' column contains numpy arrays of strings
        (as stored in parquet), new_dictionary must convert them to Python
        lists rather than calling str() which produces "['word1' 'word2'...]".
        """
        rng = np.random.default_rng(42)
        words = ["good", "bad", "great", "terrible", "the", "food", "service"]
        rows = []
        for _ in range(30):
            n = rng.integers(3, 8)
            tokens = list(rng.choice(words, size=n))
            rows.append(tokens)

        df = pd.DataFrame({"tokens": rows, "stars": [5] * 15 + [1] * 15})
        cfg = TokenizerConfig(text_col="tokens", label_col="stars", save_dictionary=False)
        dictionary = new_dictionary(df, cfg)

        # All tokens in the dictionary must be clean strings
        for word in dictionary.token2id:
            assert not word.startswith("'"), f"Token {word!r} has a leading quote — numpy str() bug"
            assert not word.endswith("'"), f"Token {word!r} has a trailing quote — numpy str() bug"

    def test_text_sequencer_with_numpy_tokens(self) -> None:
        """text_sequencer should map numpy-array tokens to dictionary IDs correctly.

        When tokens are stored as numpy arrays in parquet, text_sequencer
        must still look them up in the dictionary. With the str() bug,
        nearly every token would be mapped to OOV (out-of-vocabulary).
        """
        rng = np.random.default_rng(42)
        words = ["good", "bad", "great", "terrible", "the", "food", "service"]
        rows = []
        for _ in range(30):
            n = rng.integers(3, 8)
            tokens = list(rng.choice(words, size=n))
            rows.append(tokens)

        df = pd.DataFrame({"tokens": rows, "stars": [5] * 15 + [1] * 15})
        cfg = TokenizerConfig(text_col="tokens", label_col="stars", save_dictionary=False)
        dictionary = new_dictionary(df, cfg)

        oov_index = len(dictionary) + 1

        # text_sequencer should find most words in the dictionary
        test_tokens = np.array(["good", "food", "the"], dtype=object)
        result = text_sequencer(dictionary, test_tokens, max_len=200)

        # No token should be mapped to OOV if the dictionary contains it
        for i, word in enumerate(test_tokens):
            if word in dictionary.token2id:
                assert result[i] != oov_index, f"Token {word!r} is in dictionary but mapped to OOV"

    def test_compute_pos_weight_imbalanced(self) -> None:
        """compute_pos_weight should return neg_count/pos_count for imbalanced data."""
        df = pd.DataFrame({"target": [1.0] * 80 + [0.0] * 20})
        weight = compute_pos_weight(df)
        assert abs(weight - 0.25) < 1e-4, f"Expected pos_weight=0.25 (20/80), got {weight}"

    def test_compute_pos_weight_balanced(self) -> None:
        """compute_pos_weight should return 1.0 for balanced data."""
        df = pd.DataFrame({"target": [1.0] * 50 + [0.0] * 50})
        weight = compute_pos_weight(df)
        assert abs(weight - 1.0) < 1e-4, f"Expected pos_weight=1.0, got {weight}"

    def test_compute_pos_weight_empty_class(self) -> None:
        """compute_pos_weight should return 1.0 when a class is empty."""
        df = pd.DataFrame({"target": [1.0] * 10})
        weight = compute_pos_weight(df)
        assert weight == 1.0
