import enum
import os
from dataclasses import dataclass
from logging import INFO

import numpy as np
import torch

from sentimentizer import root

data_path = os.path.join(root, "sentimentizer")
external_data = root.parent / "data"  # data/ one level above project root

DEFAULT_LOG_LEVEL = INFO

BATCH_SIZE: int = 100000
WRITE_BYTES: str = "wb"
READ_BYTES: str = "rb"
TEXT_COLUMN: str = "text"

# Embedding constants
EMBEDDING_DTYPE = np.float32
EMBEDDING_RANDOM_MEAN: float = 0.0
EMBEDDING_RANDOM_STD: float = 0.32

# Logging constants
EXTRACT_LOG_INTERVAL: int = 100000


def auto_detect_device() -> str:
    """Detect the best available compute device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_epochs(model_type: str) -> int:
    """Return default epochs for the model type.

    RNNs converge faster with simple patterns.
    Transformers need more epochs for attention patterns to develop.
    """
    if model_type in ("encoder", "decoder"):
        return 8
    return 4


def default_dataloader_workers(device: str) -> int:
    """Return optimal DataLoader workers for the device.

    MPS has issues with multiprocessing, so use 0 workers.
    CUDA and CPU benefit from multiple workers for data loading.
    """
    if device == "mps":
        return 0
    return min(os.cpu_count() or 4, 10)


Devices: frozenset[str] = frozenset(("auto", "cpu", "cuda", "mps"))


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
    """Default optimization params (used for RNN).

    Encoder and Decoder models override these via their config.
    """

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4


@dataclass
class EncoderOptimizationParams:
    """Optimization params for Transformer models (Encoder, Decoder).

    Uses lower LR and AdamW-style weight decay for stable training.
    """

    lr: float = 0.0005
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01


@dataclass
class SchedulerParams:
    """Default scheduler params (used for RNN).

    Encoder and Decoder models override these via their config.
    """

    T_max: int = 4
    eta_min: float = 1e-6
    last_epoch: int = -1


@dataclass
class EncoderSchedulerParams:
    """Scheduler params for Transformer models.

    Includes warmup_epochs for linear LR warmup at the start of training.
    """

    T_max: int = 4
    eta_min: float = 1e-6
    last_epoch: int = -1
    warmup_epochs: int = 1  # linear warmup for this many epochs


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
    epochs: int = -1  # -1 means use model-specific default (4 for RNN, 8 for encoder/decoder)
    early_stopping_patience: int = 2  # stop if val_loss doesn't improve for this many epochs
    dataloader_workers: int = -1  # -1 means auto-detect based on device
    ray_workers: int = 2  # Ray Train workers (only used with --distributed)
    device: str = "auto"  # "auto" detects best available: cuda > mps > cpu
    memory: bool = True
    checkpoint_dir: str = ""  # directory to save checkpoints (empty = no checkpointing)
    checkpoint_every: int = 1  # save checkpoint every N epochs (0 = disabled)
    checkpoint_best: bool = True  # save the best model (lowest val loss) separately


@dataclass
class EmbeddingsConfig:
    file_path: str = str(external_data / "glove.6B.zip")
    # sub_file_path: str = "glove.6B.zip/glove.6B.100d.txt"
    sub_file_path: str = "glove.6B.100d.txt"

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
