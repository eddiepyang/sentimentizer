import enum
from typing import Tuple
from torch import nn
from dataclasses import dataclass
from yelp_nlp import root
from logging import NOTSET, DEBUG, INFO, WARN, ERROR, CRITICAL


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
class ParserConfig:
    dictionary_save_path: str = f"{root}/data/yelp_data.dictionary"
    data_save_path: str = f"{root}/data/review_data.parquet"
    text_col: str = "text"
    label_col: str = "stars"
    x_labels: str = "data"
    y_labels: str = "target"
    stop: int = 10000
    max_len: int = 200
    dict_min: int = 10
    dict_keep: int = 5000
    no_above: float = 0.99


@dataclass(frozen=True)
class FileConfig:
    archive_path: str = f"{root}/data/archive.zip"
    review_filename: str = "yelp_academic_dataset_review.json"


@dataclass
class TrainerConfig:
    batch_size: int = 256
    epochs: int = 4
    workers: int = 6
    device: str = "cuda"
    memory: bool = True


@dataclass
class EmbeddingsConfig:
    emb_path: str = f"{root}/data/glove.6B.zip"
    emb_file: str = "glove.6B.zip"
    emb_subfile: str = "glove.6B.100d.txt"


@dataclass
class DriverConfig:
    embeddings: EmbeddingsConfig = EmbeddingsConfig
    parser: ParserConfig = ParserConfig
    trainer: TrainerConfig = TrainerConfig
