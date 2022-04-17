import enum
from torch import nn
from dataclasses import dataclass


class FitModes(enum.Enum):
    fitting = 0
    training = 1
    evaluation = 2


@dataclass
class OptParams:

    lr: float = 0.005
    betas: tuple = (0.7, 0.99)
    weight_decay: float = 1e-4


@dataclass
class SchedulerParams:

    T_max: int = 100
    eta_min: int = 0
    last_epoch: int = -1


@dataclass(frozen=True)
class ParserConfig:
    text_col: str = "text"
    label_col: str = "stars"
    x_labels: str = "data"
    y_labels: str = "target"
    save_path: str = "projects/yelp_nlp/data/yelp_data"
    stop: int = 10000
    max_len: int = 200
    dict_min: int = 10
    dict_keep: int = 5000
    no_above: float = 0.99


loss_function = nn.BCEWithLogitsLoss()


@dataclass
class TrainerConfig:
    batch_size: int
    epochs: int
    workers: int
    device: str
    memory: bool = True
