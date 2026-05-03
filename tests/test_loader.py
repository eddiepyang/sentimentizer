import os
import tempfile

import pandas as pd
import torch

from sentimentizer.loader import CorpusDataset, load_train_val_corpus_datasets


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
    # Create imbalanced dummy data: 4 positive, 1 negative
    data = {"data": [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], "target": [1.0, 1.0, 1.0, 1.0, 0.0]}
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "imbalanced_data.parquet")
        df.to_parquet(data_path)

        # Test with balancing enabled (default)
        # On a small dataset with 1 negative, it should undersample positive to 1.
        # But wait, train_test_split happens BEFORE balancing in the code.
        # Let's check the code:
        # train_df, val_df = train_test_split(df, test_size=test_size)
        # if balance_classes:
        #     train_df = _balance_dataframe(train_df, ...)

        train_ds, val_ds = load_train_val_corpus_datasets(
            data_path, test_size=0.2, balance_classes=True, random_state=42
        )

        # Original: 5 rows. test_size=0.2 -> val=1, train=4.
        # If train has 3 pos and 1 neg (most likely split), balancing will make it 1 pos and 1 neg.
        # So len(train_ds) should be 2.

        assert len(train_ds) == 2

        # Verify classes are balanced in train_ds
        targets = [train_ds[i][1].item() for i in range(len(train_ds))]
        assert targets.count(1.0) == targets.count(0.0)
