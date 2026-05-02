import enum
import os
from dataclasses import dataclass
from logging import INFO

from sentimentizer import root

data_path = os.path.join(root, "sentimentizer")
external_data = root.parent / "data"  # data/ one level above project root

DEFAULT_LOG_LEVEL = INFO

BATCH_SIZE: int = 100000
WRITE_BYTES: str = "wb"
READ_BYTES: str = "rb"
TEXT_COLUMN: str = "text"

Devices: frozenset[str] = frozenset(("cpu", "cuda", "mps"))


class Device(enum.Enum):
    CPU = 0
    CUDA = 1
    MPS = 2


class FitModes(enum.Enum):
    fitting = 0
    training = 1
    evaluation = 2


@dataclass
class OptimizationParams:
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4


@dataclass
class SchedulerParams:
    T_max: int = 4
    eta_min: float = 1e-6
    last_epoch: int = -1


@dataclass(frozen=True)
class TokenizerConfig:
    text_col: str = "tokens"
    label_col: str = "stars"
    inputs: str = "data"
    labels: str = "target"
    stop: int = 10000
    max_len: int = 200
    dict_min: int = 3
    dict_keep: int = 20000
    no_above: float = 0.99999
    save_dictionary: bool = True


@dataclass(frozen=True)
class FileConfig:
    archive_file_path: str = str(external_data / "yelp_dataset.tar")
    raw_file_path: str = "yelp_academic_dataset_review.json"
    dictionary_file_path: str = f"{data_path}/data/yelp.dictionary"
    raw_reviews_file_path: str = f"{data_path}/data/review_data_raw.parquet"
    processed_reviews_file_path: str = f"{data_path}/data/review_data.parquet"
    weights_file_path: str = f"{data_path}/data/weights.pth"


@dataclass
class TrainerConfig:
    batch_size: int = 64
    epochs: int = 4
    dataloader_workers: int = 10  # DataLoader subprocesses for data loading
    ray_workers: int = 2  # Ray Train workers (only used with --distributed)
    device: str = "cuda"
    memory: bool = True


@dataclass
class EmbeddingsConfig:
    file_path: str = str(external_data / "glove.6B.zip")
    sub_file_path: str = "glove.6B.zip/glove.6B.100d.txt"
    emb_length: int = 100


@dataclass(frozen=True)
class RNNConfig:
    """Configuration for the RNN model architecture."""

    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the Transformer Encoder model architecture."""

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.2
    ff_multiplier: int = 4  # dim_feedforward = d_model * ff_multiplier


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for the Transformer Decoder model architecture."""

    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 4
    dropout: float = 0.2
    ff_multiplier: int = 4  # dim_feedforward = d_model * ff_multiplier


@dataclass
class DriverConfig:
    files: type[FileConfig] = FileConfig
    embeddings: type[EmbeddingsConfig] = EmbeddingsConfig
    tokenizer: type[TokenizerConfig] = TokenizerConfig
    trainer: type[TrainerConfig] = TrainerConfig
    rnn: type[RNNConfig] = RNNConfig
    encoder: type[EncoderConfig] = EncoderConfig
    decoder: type[DecoderConfig] = DecoderConfig
