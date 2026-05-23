"""PyTorch Dataset and collation utilities for Hugging Face models in sentimentizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from sentimentizer.loader import _balance_dataframe


class HFDataset(Dataset):
    """Dataset for Hugging Face models.

    Pre-converts DataFrame columns to tensors at initialization time
    for fast iteration during training, yielding dictionaries of inputs
    and target labels.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        x_labels: str = "input_ids",
        mask_labels: str = "attention_mask",
        y_labels: str = "target",
    ) -> None:
        super().__init__()
        self._input_ids = [
            torch.tensor(np.asarray(val), dtype=torch.long) for val in data[x_labels].values
        ]
        self._attention_mask = [
            torch.tensor(np.asarray(val), dtype=torch.long) for val in data[mask_labels].values
        ]

        # Targets are optional (e.g. during prediction/inference)
        if y_labels in data:
            self._target = [
                torch.tensor(np.asarray(val), dtype=torch.long) for val in data[y_labels].values
            ]
        else:
            self._target = None

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": self._input_ids[i],
            "attention_mask": self._attention_mask[i],
        }
        if self._target is not None:
            item["target"] = self._target[i]
        return item


class HFCollateFn:
    """Collation function that dynamically pads sequences to the longest in the batch.

    Avoids padding to fixed max length (512) to save VRAM and computation.
    """

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]

        max_len = max(len(ids) for ids in input_ids_list)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids_list, attention_mask_list, strict=True):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_ids = torch.cat(
                    [
                        ids,
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
                padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            else:
                padded_ids = ids
                padded_mask = mask
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        inputs = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
        }

        targets = None
        if "target" in batch[0]:
            targets = torch.stack([item["target"] for item in batch])

        return inputs, targets


def ray_hf_collate_fn(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Collation function for Ray Data ``iter_torch_batches(collate_fn=...)``.

    Ray passes a dict mapping column names to numpy arrays.  For HF models
    the ``input_ids`` and ``attention_mask`` columns are ragged (object-dtype
    numpy arrays of variable-length lists) which Ray cannot auto-convert to
    torch tensors.  This function pads them to the longest sequence in the
    batch and returns a dict of properly-shaped torch tensors that
    ``HFTransformerModel.prepare_batch`` expects.

    Args:
        batch: Dict mapping column names to numpy arrays.  ``input_ids`` and
            ``attention_mask`` are object-dtype (ragged); ``target`` is a
            plain int64 array.

    Returns:
        Dict of torch tensors with ``input_ids`` (B, max_len),
        ``attention_mask`` (B, max_len), and ``target`` (B,).
    """
    # Convert each ragged row to a 1-D long tensor and find max length.
    # Rows from object-dtype numpy arrays may themselves be object-dtype
    # (even when containing plain ints), so cast via .tolist() which
    # always yields Python ints that torch.tensor can handle.
    ids_rows = [torch.tensor(row.tolist(), dtype=torch.long) for row in batch["input_ids"]]
    mask_rows = [torch.tensor(row.tolist(), dtype=torch.long) for row in batch["attention_mask"]]
    max_len = max(len(r) for r in ids_rows)

    padded_ids = []
    padded_masks = []
    for ids, mask in zip(ids_rows, mask_rows, strict=True):
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), 0, dtype=torch.long)]))
            padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
        else:
            padded_ids.append(ids)
            padded_masks.append(mask)

    result: dict[str, torch.Tensor] = {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_masks),
    }

    if "target" in batch:
        result["target"] = torch.tensor(batch["target"], dtype=torch.long)

    return result


def load_train_val_hf_datasets(
    data_path: str,
    test_size: float = 0.2,
    balance_classes: bool = False,
    random_state: int = 42,
) -> tuple[HFDataset, HFDataset]:
    """Load tokenized Hugging Face data from Parquet into train/val HFDatasets."""
    df = pd.read_parquet(data_path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    del df

    if balance_classes:
        train_df = _balance_dataframe(train_df, target_col="target", random_state=random_state)

    return HFDataset(data=train_df), HFDataset(val_df)
