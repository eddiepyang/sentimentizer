from torch import nn
from dataclasses import dataclass


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


loss_function = nn.BCEWithLogitsLoss()
