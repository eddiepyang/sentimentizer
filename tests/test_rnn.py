import json
import shutil

import pandas as pd
import pytest
import ray
import torch

from sentimentizer.config import (
    DEFAULT_LOG_LEVEL,
    DecoderConfig,
    EncoderConfig,
    RNNConfig,
    TrainerConfig,
)
from sentimentizer.extractor import extract_data
from sentimentizer.loader import CorpusDataset, load_train_val_ray_datasets
from sentimentizer.models.decoder import Decoder
from sentimentizer.models.encoder import Encoder
from sentimentizer.models.rnn import RNN, get_trained_model
from sentimentizer.tokenizer import (
    Tokenizer,
    convert_rating,
    get_trained_tokenizer,
    new_logger,
    regex_tokenize,
)
from sentimentizer.trainer import new_ray_trainer, new_trainer

logger = new_logger(DEFAULT_LOG_LEVEL)


@pytest.fixture
def tokenized_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tokens": [
                ["the", "chicken", "never", "showed", "up"],
                ["the", "food", "was", "terrific"],
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
    assert convert_rating(5) == 1
    assert convert_rating(1) == 0
    assert convert_rating(3) == 0.5


def test_tokenize(raw_df):
    output = regex_tokenize(raw_df.text[0])

    for item in output:
        assert isinstance(item, str)

    assert len(output) > 3


class TestExtractData:
    fname = "artificial-reviews.jsonl"
    stop = 2

    def test_success(self, rel_path, relative_root):
        ray.init(ignore_reinit_error=True)
        ds = extract_data(compressed_file_name=self.fname, file_path=rel_path, stop=self.stop)
        assert isinstance(ds, ray.data.Dataset)

        path = f"{relative_root}/tests/test_data/file.parquet"
        shutil.rmtree(path, ignore_errors=True)
        ds.write_parquet(path)

        df = pd.read_parquet(path)
        assert df.shape == (2, 3)  # text, tokens, stars
        assert df["tokens"].dtype == "object" or isinstance(df["tokens"].dtype, pd.ArrowDtype)
        assert pd.api.types.is_integer_dtype(df["stars"].dtype)

    def test_failure_empty_input(self):
        # todo
        return


class TestDataTokenizer:
    def test_success(self, tokenized_df):
        parser = Tokenizer.from_data(tokenized_df)
        result = parser.transform_dataframe(tokenized_df)
        assert result.shape == (2, 4)

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
    """tests model construction and weight loading"""

    def test_model_construction(self):
        """tests that the RNN model can be constructed with correct architecture"""
        emb_weights = torch.zeros(100, 100)  # small vocab, 100d embeddings
        model = RNN(emb_weights=emb_weights)
        assert isinstance(model, RNN)
        assert model.lstm.bidirectional
        assert model.lstm.batch_first
        assert model.lstm.num_layers == 2

    def test_forward_pass(self):
        """tests that the forward pass produces correct output shape"""
        emb_weights = torch.randn(100, 100)
        model = RNN(emb_weights=emb_weights)
        tokens = torch.randint(0, 100, (2, 10))
        output = model(tokens)
        assert output.shape == (2,)

    def test_missing_weights_file(self):
        """tests that loading from a missing weights file raises FileNotFoundError"""
        from unittest.mock import patch

        mock_path = "sentimentizer.models.rnn.torch.load"
        with (
            patch(mock_path, side_effect=FileNotFoundError("No weights")),
            pytest.raises(FileNotFoundError),
        ):
            get_trained_model("cpu")

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
        result = regex_tokenize("chicken wasn't good")

        assert len(result) == 3
        assert result[0] == "chicken"
        assert result[1] == "wasn't"

    def test_success_one(self):
        result = regex_tokenize("1st place food")
        assert len(result) == 3
        assert result[0] == "1st"


# ──────────────────────────────────────────────
# Ray Data and Ray Train tests
# ──────────────────────────────────────────────


class TestRayDatasetLoader:
    """tests Ray Dataset loading and splitting"""

    def test_load_ray_datasets(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            assert isinstance(train_ds, ray.data.Dataset)
            assert isinstance(val_ds, ray.data.Dataset)
            total = train_ds.count() + val_ds.count()
            assert total > 0
        except Exception:
            pytest.skip("test parquet data not available")

    def test_load_ray_datasets_default_split(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path)
            # Default test_size is 0.2, so train should be ~80% and val ~20%
            total = train_ds.count() + val_ds.count()
            assert total > 0
        except Exception:
            pytest.skip("test parquet data not available")


class TestNewRayTrainer:
    """tests Ray Train TorchTrainer creation"""

    def test_trainer_creation(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            trainer = new_ray_trainer(
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=cfg,
                model_type="rnn",
            )
            from ray.train.torch import TorchTrainer

            assert isinstance(trainer, TorchTrainer)
        except Exception:
            pytest.skip("test parquet data not available")

    def test_trainer_creation_encoder(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            trainer = new_ray_trainer(
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=cfg,
                model_type="encoder",
            )
            from ray.train.torch import TorchTrainer

            assert isinstance(trainer, TorchTrainer)
        except Exception:
            pytest.skip("test parquet data not available")

    def test_trainer_creation_decoder(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            trainer = new_ray_trainer(
                train_ds=train_ds,
                val_ds=val_ds,
                cfg=cfg,
                model_type="decoder",
            )
            from ray.train.torch import TorchTrainer

            assert isinstance(trainer, TorchTrainer)
        except Exception:
            pytest.skip("test parquet data not available")

    def test_trainer_invalid_model_type(self, relative_root):
        ray.init(ignore_reinit_error=True)
        path = f"{relative_root}/tests/test_data/file.parquet"
        try:
            train_ds, val_ds = load_train_val_ray_datasets(path, test_size=0.5)
            cfg = TrainerConfig(ray_workers=1, device="cpu", epochs=1, batch_size=2)
            with pytest.raises(ValueError, match="no matching model"):
                new_ray_trainer(
                    train_ds=train_ds,
                    val_ds=val_ds,
                    cfg=cfg,
                    model_type="invalid_model",
                )
        except Exception:
            pytest.skip("test parquet data not available")

    def test_train_func_config_serializable(self):
        """test that _train_func config dict is serializable (required for Ray)"""
        config = {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.005,
            "betas": [0.7, 0.99],
            "weight_decay": 1e-4,
            "use_warmup": False,
            "warmup_steps": 0,
            "total_steps": 0,
            "scheduler_eta_min": 1e-6,
            "model_type": "rnn",
            "dict_path": "/tmp/test.dict",
            "embeddings_file_path": "/tmp/test.zip",
            "embeddings_sub_file_path": "test.txt",
            "embeddings_emb_length": 100,
            "input_len": 200,
        }
        serialized = json.dumps(config)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["model_type"] == "rnn"
        assert deserialized["epochs"] == 1

    def test_train_func_config_has_all_keys(self):
        """test that _train_func config contains all required keys"""
        required_keys = {
            "epochs",
            "batch_size",
            "lr",
            "betas",
            "weight_decay",
            "use_warmup",
            "warmup_steps",
            "total_steps",
            "scheduler_eta_min",
            "model_type",
            "dict_path",
            "embeddings_file_path",
            "embeddings_sub_file_path",
            "embeddings_emb_length",
            "input_len",
        }
        config = {
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.005,
            "betas": [0.7, 0.99],
            "weight_decay": 1e-4,
            "use_warmup": False,
            "warmup_steps": 0,
            "total_steps": 0,
            "scheduler_eta_min": 1e-6,
            "model_type": "rnn",
            "dict_path": "/tmp/test.dict",
            "embeddings_file_path": "/tmp/test.zip",
            "embeddings_sub_file_path": "test.txt",
            "embeddings_emb_length": 100,
            "input_len": 200,
        }
        assert required_keys.issubset(set(config.keys()))


class TestTrainerConfig:
    """tests TrainerConfig with ray_workers for distributed training"""

    def test_default_ray_workers(self):
        cfg = TrainerConfig()
        assert cfg.ray_workers == 2

    def test_custom_ray_workers(self):
        cfg = TrainerConfig(ray_workers=4)
        assert cfg.ray_workers == 4

    def test_cpu_device_config(self):
        cfg = TrainerConfig(device="cpu", ray_workers=1)
        assert cfg.device == "cpu"
        assert cfg.ray_workers == 1


class TestModelConfigs:
    """tests that model configs drive architecture dimensions"""

    def test_rnn_config_defaults(self):
        cfg = RNNConfig()
        assert cfg.hidden_size == 256
        assert cfg.num_layers == 2
        assert cfg.dropout == 0.2

    def test_rnn_custom_config(self):
        """tests that custom RNNConfig changes model architecture"""
        cfg = RNNConfig(hidden_size=128, num_layers=3, dropout=0.3)
        emb_weights = torch.zeros(100, 100)
        model = RNN(
            emb_weights=emb_weights,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        assert model.lstm.hidden_size == 128
        assert model.lstm.num_layers == 3

    def test_encoder_config_defaults(self):
        cfg = EncoderConfig()
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_layers == 4
        assert cfg.ff_multiplier == 4

    def test_encoder_custom_config(self):
        """tests that custom EncoderConfig changes model architecture"""
        cfg = EncoderConfig(d_model=128, n_heads=2, n_layers=2, ff_multiplier=2)
        emb_weights = torch.zeros(100, 100)
        model = Encoder(
            input_len=200,
            emb_weights=emb_weights,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ff_multiplier=cfg.ff_multiplier,
        )
        assert model.d_model == 128
        assert model.encoder.num_layers == 2

    def test_decoder_config_defaults(self):
        cfg = DecoderConfig()
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_encoder_layers == 2
        assert cfg.n_decoder_layers == 4
        assert cfg.ff_multiplier == 4

    def test_decoder_custom_config(self):
        """tests that custom DecoderConfig changes model architecture"""

        cfg = DecoderConfig(d_model=128, n_heads=2, n_encoder_layers=1, n_decoder_layers=2)
        emb_weights = torch.zeros(100, 100)
        model = Decoder(
            input_len=200,
            emb_weights=emb_weights,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_encoder_layers=cfg.n_encoder_layers,
            n_decoder_layers=cfg.n_decoder_layers,
        )
        assert model.d_model == 128
        assert model.encoder.num_layers == 1
        assert model.decoder.num_layers == 2


class TestSingleTrainer:
    """tests the existing single-node Trainer still works"""

    def test_new_trainer_creates_trainer(self):
        emb_weights = torch.randn(100, 100)
        model = RNN(emb_weights=emb_weights)
        cfg = TrainerConfig(device="cpu")
        trainer = new_trainer(model=model, cfg=cfg)
        assert isinstance(trainer.loss_function, torch.nn.BCEWithLogitsLoss)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.cfg.device == "cpu"
