import os
from dataclasses import dataclass
from logging import INFO

import numpy as np

from sentimentizer import root
from sentimentizer.device import resolve_device as auto_detect_device  # noqa: F401 — re-export

data_path = os.path.join(root, "sentimentizer")
external_data = root.parent / "data"  # data/ one level above project root

DEFAULT_LOG_LEVEL = INFO

# Embedding constants
EMBEDDING_DTYPE = np.float32
EMBEDDING_RANDOM_MEAN: float = 0.0
EMBEDDING_RANDOM_STD: float = 0.32

# Logging constants
EXTRACT_LOG_INTERVAL: int = 100000


def default_epochs(model_type: str) -> int:
    """Return default epochs for the model type.

    RNNs need enough epochs to learn compositional patterns like negation.
    Transformers need more epochs for attention patterns to develop.
    """
    if model_type in ("encoder", "decoder"):
        return 8
    return 1


def default_dataloader_workers(device: str) -> int:
    """Return optimal DataLoader workers for the device.

    MPS has issues with multiprocessing, so use 0 workers.
    CUDA and CPU benefit from multiple workers for data loading.
    """
    if device == "mps":
        return 0
    return min(os.cpu_count() or 4, 10)


# Valid device strings for model loading (excludes "auto" — must be resolved first)
VALID_DEVICES: frozenset[str] = frozenset(("cpu", "cuda", "mps"))


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
class DecoderOptimizationParams:
    """Optimization params for Decoder model.

    The decoder has more parameters than the encoder due to cross-attention
    layers, so it uses a lower LR and higher weight decay to prevent
    overfitting on the sentiment classification task.
    """

    lr: float = 0.0003
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.02


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
    """Scheduler params for Encoder model.

    Includes warmup_epochs for linear LR warmup at the start of training.
    """

    T_max: int = 8
    eta_min: float = 1e-6
    last_epoch: int = -1
    warmup_epochs: int = 1  # linear warmup for this many epochs


@dataclass
class DecoderSchedulerParams:
    """Scheduler params for Decoder model.

    Uses a longer warmup and higher minimum LR than the encoder because
    the decoder has more parameters and is more prone to overfitting
    during early training.
    """

    T_max: int = 8
    eta_min: float = 1e-5
    last_epoch: int = -1
    warmup_epochs: int = 2  # longer warmup for stable decoder training


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
    rnn_weights_file_path: str = f"{data_path}/data/rnn_weights.pth"
    encoder_weights_file_path: str = f"{data_path}/data/encoder_weights.pth"
    decoder_weights_file_path: str = f"{data_path}/data/decoder_weights.pth"


def weights_path_for(model_type: str) -> str:
    """Return the weights file path for the given model type.

    Args:
        model_type: One of 'rnn', 'encoder', 'decoder'.

    Returns:
        Absolute path to the weights file for that model type.
    """
    if model_type == "rnn":
        return FileConfig.rnn_weights_file_path
    elif model_type == "encoder":
        return FileConfig.encoder_weights_file_path
    elif model_type == "decoder":
        return FileConfig.decoder_weights_file_path
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


@dataclass
class TrainerConfig:
    batch_size: int = 64
    epochs: int = -1  # -1 means use model-specific default (4 for RNN, 8 for encoder/decoder)
    early_stopping_patience: int = 3  # stop if val_loss doesn't improve for this many epochs
    dataloader_workers: int = -1  # -1 means auto-detect based on device
    ray_workers: int = 1  # Ray Train workers (only used with --distributed)
    device: str = "auto"  # "auto" detects best available: cuda > mps > cpu
    memory: bool = True
    checkpoint_dir: str = ""  # directory to save checkpoints (empty = no checkpointing)
    checkpoint_every: int = 1  # save checkpoint every N epochs (0 = disabled)
    checkpoint_best: bool = True  # save the best model (lowest val loss) separately
    pos_weight: float = 1.0  # weight for the positive class in the loss function


@dataclass
class EmbeddingsConfig:
    model_name: str = "glove-wiki-gigaword-100"  # auto-downloaded via gensim.downloader
    emb_length: int = 100


# Per-model Hugging Face Hub repository IDs for pre-trained weights.
# Each model type maps to its own repo; weights are stored as
# ``{model_type}_weights.pth`` inside the repo.
HF_WEIGHTS_REPOS: dict[str, str] = {
    "rnn": "ryeyoo/sentimentizer-rnn",
    "encoder": "ryeyoo/sentimentizer-encoder",
    "decoder": "ryeyoo/sentimentizer-decoder",
}


@dataclass
class HuggingFaceConfig:
    """Configuration for Hugging Face Hub interactions.

    Attributes:
        repo_id: Default repo ID (used by push_model_to_hub).
            Per-model repos are defined in HF_WEIGHTS_REPOS above.
    """

    repo_id: str = "ryeyoo/sentimentizer"


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
    """Configuration for the Transformer Decoder model architecture.

    The decoder uses fewer layers and higher dropout than the encoder
    because the cross-attention mechanism (2 encoder + 2 decoder = 4
    layers total) adds significant capacity.  With the original 2+4
    configuration the model had ~5.9M parameters and overfit quickly.
    """

    d_model: int = 256
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    dropout: float = 0.3
    ff_multiplier: int = 4


@dataclass
class DriverConfig:
    files: type[FileConfig] = FileConfig
    embeddings: type[EmbeddingsConfig] = EmbeddingsConfig
    tokenizer: type[TokenizerConfig] = TokenizerConfig
    trainer: type[TrainerConfig] = TrainerConfig
    rnn: type[RNNConfig] = RNNConfig
    encoder: type[EncoderConfig] = EncoderConfig
    decoder: type[DecoderConfig] = DecoderConfig
    hf: type[HuggingFaceConfig] = HuggingFaceConfig
