import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from sentimentizer.config import ModernBERTConfig
from sentimentizer.hf_dataset import HFCollateFn, ray_hf_collate_fn
from sentimentizer.hf_tokenizer import HFTokenizer
from sentimentizer.models.modernbert import ModernBERT


class TestModernBERT:
    """Unit tests for ModernBERT wrapper model class."""

    @pytest.fixture
    def mock_backbone(self) -> MagicMock:
        """Create a mocked AutoModel backbone."""
        mock_model = MagicMock()
        mock_model.config.hidden_size = 768

        # Mock parameters
        p = nn.Parameter(torch.randn(1, 1))
        mock_model.parameters.return_value = iter([p])

        # Mock forward return value
        mock_output = MagicMock()
        # Mock last_hidden_state (batch_size=2, seq_len=5, hidden_size=768)
        mock_output.last_hidden_state = torch.randn(2, 5, 768)
        mock_model.return_value = mock_output

        return mock_model

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mocked tokenizer."""
        mock_tok = MagicMock()
        mock_tok.model_name = "test-modernbert"
        return mock_tok

    def test_model_construction(self, mock_backbone, mock_tokenizer) -> None:
        """Verify model initializes and constructs classifier head with correct dimensions."""
        with patch(
            "sentimentizer.models.hf_base.AutoModel.from_pretrained", return_value=mock_backbone
        ):
            config = ModernBERTConfig(dropout=0.1, num_classes=3)
            model = ModernBERT(config=config, tokenizer=mock_tokenizer)

            assert isinstance(model, ModernBERT)
            assert model.MODEL_TYPE == "modernbert"
            assert isinstance(model.classifier, nn.Sequential)
            # Check classifier dimensions: Linear(768, 768) -> GELU -> Dropout -> Linear(768, 3)
            assert model.classifier[0].in_features == 768
            assert model.classifier[0].out_features == 768
            assert model.classifier[3].out_features == 3

    def test_forward_pass_with_attention_mask(self, mock_backbone, mock_tokenizer) -> None:
        """Verify forward pass with attention mask works and returns correct logit shape."""
        with patch(
            "sentimentizer.models.hf_base.AutoModel.from_pretrained", return_value=mock_backbone
        ):
            model = ModernBERT(tokenizer=mock_tokenizer)

            input_ids = torch.randint(0, 1000, (2, 5))
            attention_mask = torch.ones(2, 5, dtype=torch.long)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            assert logits.shape == (2, 3)
            mock_backbone.assert_called_once_with(
                input_ids=input_ids, attention_mask=attention_mask
            )

    def test_forward_pass_without_attention_mask(self, mock_backbone, mock_tokenizer) -> None:
        """Verify forward pass without attention mask works."""
        with patch(
            "sentimentizer.models.hf_base.AutoModel.from_pretrained", return_value=mock_backbone
        ):
            model = ModernBERT(tokenizer=mock_tokenizer)

            input_ids = torch.randint(0, 1000, (2, 5))
            logits = model(input_ids=input_ids)

            assert logits.shape == (2, 3)
            mock_backbone.assert_called_once_with(input_ids=input_ids, attention_mask=None)

    def test_prepare_batch(self, mock_backbone, mock_tokenizer) -> None:
        """Verify prepare_batch converts raw dict into model inputs and targets correctly."""
        with patch(
            "sentimentizer.models.hf_base.AutoModel.from_pretrained", return_value=mock_backbone
        ):
            model = ModernBERT(tokenizer=mock_tokenizer)

            batch = {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]]),
                "target": torch.tensor([0, 2]),
            }

            inputs, target = model.prepare_batch(batch, device="cpu")

            assert "input_ids" in inputs
            assert "attention_mask" in inputs
            assert torch.equal(inputs["input_ids"], batch["input_ids"])
            assert torch.equal(inputs["attention_mask"], batch["attention_mask"])
            assert torch.equal(target, batch["target"])

    def test_unfreeze_backbone(self, mock_backbone, mock_tokenizer) -> None:
        """Verify unfreeze_backbone sets requires_grad to True for all parameters."""
        with patch(
            "sentimentizer.models.hf_base.AutoModel.from_pretrained", return_value=mock_backbone
        ):
            model = ModernBERT(tokenizer=mock_tokenizer)

            # Manually set backbone parameters requires_grad to False
            for param in model.backbone.parameters():
                param.requires_grad = False

            model.unfreeze_backbone()

            for param in model.backbone.parameters():
                assert param.requires_grad is True

    def test_save_load_checkpoint(self, mock_backbone, mock_tokenizer) -> None:
        """Verify save_to_checkpoint_dir and load_from_checkpoint_dir work."""
        with (
            patch(
                "sentimentizer.models.hf_base.AutoModel.from_pretrained", return_value=mock_backbone
            ),
            patch(
                "sentimentizer.models.hf_base.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            model = ModernBERT(tokenizer=mock_tokenizer)

            with tempfile.TemporaryDirectory() as tmp_dir:
                ckpt_dir = Path(tmp_dir)

                # Save model
                metadata = model.save_to_checkpoint_dir(ckpt_dir, tokenizer=mock_tokenizer)

                assert mock_backbone.save_pretrained.called
                assert mock_tokenizer.save_pretrained.called
                assert "classifier_state_dict" in metadata
                # New format: config serialized as plain dict for weights_only=True compatibility
                assert "config_dict" in metadata
                assert "config_class" in metadata
                assert metadata["config_class"] == "ModernBERTConfig"
                assert "backbone_dir" in metadata

                # Mock exists for checkpoint load checks
                with patch.object(Path, "exists", return_value=True):
                    loaded_model = ModernBERT.load_from_checkpoint_dir(
                        ckpt_dir, metadata, device="cpu"
                    )
                    assert isinstance(loaded_model, ModernBERT)


class TestHFTokenizer:
    """Tests for the HFTokenizer wrapper class."""

    def test_tokenize_text(self) -> None:
        """Verify tokenizer tokenizes text correctly."""
        mock_raw_tokenizer = MagicMock()
        mock_raw_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2054, 2003, 1037, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        tokenizer = HFTokenizer(tokenizer=mock_raw_tokenizer, model_name="test-modernbert")
        res = tokenizer.tokenize_text("hello world")

        assert "input_ids" in res
        assert "attention_mask" in res
        assert res["input_ids"].shape == (5,)
        assert res["attention_mask"].shape == (5,)


class TestHFCollateFn:
    """Tests for the HFCollateFn collator class."""

    def test_collation_with_padding(self) -> None:
        """HFCollateFn should pad sequences to max length in batch."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "target": torch.tensor(0),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "target": torch.tensor(2),
            },
        ]

        collate = HFCollateFn(pad_token_id=0)
        collated_inputs, collated_target = collate(batch)

        assert "input_ids" in collated_inputs
        assert "attention_mask" in collated_inputs
        assert collated_inputs["input_ids"].shape == (2, 3)
        assert collated_inputs["attention_mask"].shape == (2, 3)
        assert torch.equal(collated_target, torch.tensor([0, 2]))

        # Check second sample padding in batch
        assert collated_inputs["input_ids"][1, 2].item() == 0  # padded
        assert collated_inputs["attention_mask"][1, 2].item() == 0  # mask is 0


class TestRayHFCollateFn:
    """Tests for the ray_hf_collate_fn used in distributed training."""

    def test_pads_ragged_numpy_arrays(self) -> None:
        """ray_hf_collate_fn should pad ragged object-dtype numpy arrays to equal length."""
        # Simulate what Ray Data passes: dict of column-name → numpy array
        batch = {
            "input_ids": np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
            "attention_mask": np.array([np.array([1, 1, 1]), np.array([1, 1])], dtype=object),
            "target": np.array([0, 2], dtype=np.int64),
        }

        result = ray_hf_collate_fn(batch)

        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["target"].shape == (2,)
        assert torch.equal(result["target"], torch.tensor([0, 2]))

        # Second sample should be padded with 0
        assert result["input_ids"][1, 2].item() == 0
        assert result["attention_mask"][1, 2].item() == 0

    def test_equal_length_sequences(self) -> None:
        """ray_hf_collate_fn should work when all sequences are the same length."""
        batch = {
            "input_ids": np.array([np.array([1, 2, 3]), np.array([4, 5, 6])], dtype=object),
            "attention_mask": np.array([np.array([1, 1, 1]), np.array([1, 1, 1])], dtype=object),
            "target": np.array([1, 1], dtype=np.int64),
        }

        result = ray_hf_collate_fn(batch)

        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        # No padding needed — values should be exact
        assert torch.equal(result["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]]))

    def test_without_target(self) -> None:
        """ray_hf_collate_fn should handle batches without target column."""
        batch = {
            "input_ids": np.array([np.array([1, 2]), np.array([3])], dtype=object),
            "attention_mask": np.array([np.array([1, 1]), np.array([1])], dtype=object),
        }

        result = ray_hf_collate_fn(batch)

        assert "target" not in result
        assert result["input_ids"].shape == (2, 2)
        assert result["input_ids"][1, 1].item() == 0  # padded
