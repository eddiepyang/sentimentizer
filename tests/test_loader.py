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
