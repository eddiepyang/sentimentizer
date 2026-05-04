"""Tests for Hugging Face Hub weight downloading and pushing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentimentizer.config import HF_WEIGHTS_REPOS
from sentimentizer.hf import download_weights, pull_model_from_hub, push_model_to_hub


class TestHFWeightsRepos:
    """Test the HF_WEIGHTS_REPOS configuration mapping."""

    def test_rnn_repo_exists(self) -> None:
        assert "rnn" in HF_WEIGHTS_REPOS
        assert HF_WEIGHTS_REPOS["rnn"] == "ryeyoo/sentimentizer-rnn"

    def test_encoder_repo_exists(self) -> None:
        assert "encoder" in HF_WEIGHTS_REPOS
        assert HF_WEIGHTS_REPOS["encoder"] == "ryeyoo/sentimentizer-encoder"

    def test_decoder_repo_exists(self) -> None:
        assert "decoder" in HF_WEIGHTS_REPOS
        assert HF_WEIGHTS_REPOS["decoder"] == "ryeyoo/sentimentizer-decoder"

    def test_unknown_model_type_not_in_repos(self) -> None:
        assert "unknown" not in HF_WEIGHTS_REPOS


class TestDownloadWeights:
    """Test the download_weights convenience function."""

    def test_download_weights_unknown_model_type(self) -> None:
        """download_weights returns None for unknown model types."""
        result = download_weights("unknown", "/tmp/unknown_weights.pth")
        assert result is None

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_success(self, mock_download: MagicMock) -> None:
        """download_weights copies downloaded file to local path."""
        import os
        import tempfile

        # Create a fake downloaded file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                dest_path = os.path.join(dest_dir, "rnn_weights.pth")
                result = download_weights("rnn", dest_path)

                assert result is not None
                assert Path(dest_path).exists()
                mock_download.assert_called_once_with(
                    repo_id="ryeyoo/sentimentizer-rnn",
                    filename="rnn_weights.pth",
                )
        finally:
            os.unlink(tmp_path)

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_entry_not_found(self, mock_download: MagicMock) -> None:
        """download_weights returns None when file not found on Hub."""
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("rnn_weights.pth not found")

        result = download_weights("rnn", "/tmp/rnn_weights.pth")
        assert result is None

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_network_error(self, mock_download: MagicMock) -> None:
        """download_weights returns None on network errors."""
        mock_download.side_effect = ConnectionError("No network")

        result = download_weights("rnn", "/tmp/rnn_weights.pth")
        assert result is None

    @patch("sentimentizer.hf.hf_hub_download")
    def test_download_weights_creates_parent_dir(self, mock_download: MagicMock) -> None:
        """download_weights creates parent directories if they don't exist."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                nested_path = os.path.join(dest_dir, "nested", "dir", "rnn_weights.pth")
                result = download_weights("rnn", nested_path)

                assert result is not None
                assert Path(nested_path).exists()
        finally:
            os.unlink(tmp_path)


class TestPullModelFromHub:
    """Test the pull_model_from_hub function."""

    @patch("sentimentizer.hf.hf_hub_download")
    def test_pull_success(self, mock_download: MagicMock) -> None:
        """pull_model_from_hub returns True on successful download."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            mock_download.return_value = tmp_path

            with tempfile.TemporaryDirectory() as dest_dir:
                dest_path = os.path.join(dest_dir, "rnn_weights.pth")
                result = pull_model_from_hub("ryeyoo/sentimentizer-rnn", "rnn", dest_path)
                assert result is True
        finally:
            os.unlink(tmp_path)

    @patch("sentimentizer.hf.hf_hub_download")
    def test_pull_entry_not_found(self, mock_download: MagicMock) -> None:
        """pull_model_from_hub returns False when file not found."""
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("not found")

        result = pull_model_from_hub("ryeyoo/sentimentizer-rnn", "rnn", "/tmp/weights.pth")
        assert result is False

    @patch("sentimentizer.hf.hf_hub_download")
    def test_pull_network_error(self, mock_download: MagicMock) -> None:
        """pull_model_from_hub returns False on network errors."""
        mock_download.side_effect = ConnectionError("No network")

        result = pull_model_from_hub("ryeyoo/sentimentizer-rnn", "rnn", "/tmp/weights.pth")
        assert result is False


class TestPushModelToHub:
    """Test the push_model_to_hub function."""

    @patch("sentimentizer.hf.HfApi")
    def test_push_file_not_found(self, mock_api_cls: MagicMock) -> None:
        """push_model_to_hub logs error and returns when file doesn't exist."""
        push_model_to_hub("/nonexistent/path.pth", "ryeyoo/sentimentizer-rnn", "rnn")
        # HfApi should not be instantiated if file doesn't exist
        mock_api_cls.assert_not_called()

    @patch("sentimentizer.hf.HfApi")
    def test_push_success(self, mock_api_cls: MagicMock) -> None:
        """push_model_to_hub uploads file to Hub."""
        import tempfile

        mock_api_instance = MagicMock()
        mock_api_cls.return_value = mock_api_instance

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"fake weights data")
            tmp_path = tmp.name

        try:
            push_model_to_hub(tmp_path, "ryeyoo/sentimentizer-rnn", "rnn")
            mock_api_instance.upload_file.assert_called_once()
        finally:
            import os

            os.unlink(tmp_path)


class TestGetTrainedModelHFDownload:
    """Test that get_trained_model falls back to HF Hub when local weights missing."""

    def test_rnn_downloads_from_hub_when_local_missing(self) -> None:
        """RNN get_trained_model downloads from HF Hub when local file missing."""
        from unittest.mock import patch

        import torch

        from sentimentizer.models.rnn import get_trained_model

        # Create fake weights that match the RNN architecture
        emb_weights = torch.zeros(100, 100)
        from sentimentizer.models.rnn import RNN

        model = RNN(emb_weights=emb_weights)
        fake_state_dict = model.state_dict()

        with (
            patch("sentimentizer.models.rnn.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights") as mock_download,
            patch("sentimentizer.models.rnn.torch.load", return_value=fake_state_dict),
        ):
            # Simulate successful download
            mock_download.return_value = Path("/fake/rnn_weights.pth")

            model = get_trained_model("cpu")
            assert isinstance(model, RNN)
            mock_download.assert_called_once_with("rnn", mock_download.call_args[0][1])

    def test_rnn_raises_when_hub_fails(self) -> None:
        """RNN get_trained_model raises FileNotFoundError when Hub download fails."""
        from unittest.mock import patch

        from sentimentizer.models.rnn import get_trained_model

        with (
            patch("sentimentizer.models.rnn.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights", return_value=None),
            pytest.raises(FileNotFoundError, match="Weights file not found"),
        ):
            get_trained_model("cpu")

    def test_encoder_downloads_from_hub_when_local_missing(self) -> None:
        """Encoder get_trained_model downloads from HF Hub when local file missing."""
        from unittest.mock import patch

        import torch

        from sentimentizer.models.encoder import Encoder, get_trained_model

        # Create fake weights that match the Encoder architecture
        emb_weights = torch.zeros(100, 100)
        model = Encoder(emb_weights=emb_weights)
        fake_state_dict = model.state_dict()

        with (
            patch("sentimentizer.models.encoder.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights") as mock_download,
            patch("sentimentizer.models.encoder.torch.load", return_value=fake_state_dict),
        ):
            mock_download.return_value = Path("/fake/encoder_weights.pth")

            model = get_trained_model("cpu")
            assert isinstance(model, Encoder)
            mock_download.assert_called_once_with("encoder", mock_download.call_args[0][1])

    def test_decoder_downloads_from_hub_when_local_missing(self) -> None:
        """Decoder get_trained_model downloads from HF Hub when local file missing."""
        from unittest.mock import patch

        import torch

        from sentimentizer.models.decoder import Decoder, get_trained_model

        # Create fake weights that match the Decoder architecture
        emb_weights = torch.zeros(100, 100)
        model = Decoder(emb_weights=emb_weights)
        fake_state_dict = model.state_dict()

        with (
            patch("sentimentizer.models.decoder.Path.exists", return_value=False),
            patch("sentimentizer.hf.download_weights") as mock_download,
            patch("sentimentizer.models.decoder.torch.load", return_value=fake_state_dict),
        ):
            mock_download.return_value = Path("/fake/decoder_weights.pth")

            model = get_trained_model("cpu")
            assert isinstance(model, Decoder)
            mock_download.assert_called_once_with("decoder", mock_download.call_args[0][1])
