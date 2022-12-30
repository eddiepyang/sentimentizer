import enum
from typing import Tuple, Callable
from torch import nn
from dataclasses import dataclass
from torch_sentiment import root
from logging import NOTSET, DEBUG, INFO, WARN, ERROR, CRITICAL
import os 


data_path = os.path.join(root, "torch_sentiment")

class LogLevels(enum.Enum):
    unset = NOTSET
    debug = DEBUG
    info = INFO
    warn = WARN
    error = ERROR
    critical = CRITICAL


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


@dataclass
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
    reviews_file_path: str = f"{data_path}/data/review_data.parquet"
    weights_file_path: str = f"{data_path}/data/weights.pth"


@dataclass
class TrainerConfig:
    batch_size: int = 256
    epochs: int = 4
    workers: int = 6
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
