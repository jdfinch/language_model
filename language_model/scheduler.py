import ezpyzy as ez
import dataclasses as dc
import pathlib as pl
import json

import transformers as hf
import torch as pt

from language_model.utils.get_name_of_subclass import get_name_of_subclass


@dc.dataclass
class Scheduler(ez.Config):

    def __post_init__(self):
        self.scheduler = get_name_of_subclass(self, Scheduler)

    def construct_scheduler(self, optimizer):
        raise TypeError("A subclass of Scheduler must be used to specify a learning rate schedule.")


@dc.dataclass
class LinearWarmupSchedule(ez.Config):
    num_warmup_steps: int = 100
    """The number of training steps where the learning rate is linearly increased. Setting to 0 results in no warmup."""
    start_factor: float = 0.0
    """The factor by which the learning rate is multiplied at the start of warmup."""
    end_factor: float = 1.0
    """The factor by which the learning rate is multiplied at the end of warmup."""

    def construct_scheduler(self, optimizer):
        return pt.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=self.start_factor,
            end_factor=self.end_factor,
            total_iters=self.num_warmup_steps,)

