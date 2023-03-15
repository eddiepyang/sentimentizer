import os
import enum

from dataclasses import dataclass
from logging import NOTSET, DEBUG, INFO, WARN, ERROR, CRITICAL

from typing import Tuple, Callable
from sentimentizer import root

data_path = os.path.join(root, "sentimentizer")

DEFAULT_LOG_LEVEL = INFO

BATCH_SIZE = 100000
WRITE_BYTES = "wb"
READ_BYTES = "rb"
TEXT_COLUMN = "text"

Devices = set(("cpu", "cuda", "mps"))

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
    lr: float = 0.005
    betas: Tuple[float, float] = (0.7, 0.99)
    weight_decay: float = 1e-4


@dataclass
class SchedulerParams:
    T_max: int = 100
    eta_min: int = 0
    last_epoch: int = -1


@dataclass(frozen=True)
class TokenizerConfig:
    text_col: str = "text"
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
    archive_file_path: str = f"{data_path}/data/archive.zip"
    raw_file_path: str = "yelp_academic_dataset_review.json"
    dictionary_file_path: str = f"{data_path}/data/yelp.dictionary"
    raw_reviews_file_path: str = f"{data_path}/data/review_data.arrow"
    processed_reviews_file_path: str = f"{data_path}/data/review_data.parquet"
    weights_file_path: str = f"{data_path}/data/weights.pth"


@dataclass
class TrainerConfig:
    batch_size: int = 64
    epochs: int = 4
    workers: int = 10
    device: str = "cuda"
    memory: bool = True


@dataclass
class EmbeddingsConfig:
    file_path: str = f"{data_path}/data/glove.6B.zip"
    sub_file_path: str = "glove.6B.100d.txt"
    emb_length: int = 100


@dataclass
class DriverConfig:
    files: Callable = FileConfig
    embeddings: Callable = EmbeddingsConfig
    tokenizer: Callable = TokenizerConfig
    trainer: Callable = TrainerConfig
