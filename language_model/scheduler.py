import ezpyzy as ez
import dataclasses as dc
import pathlib as pl
import json

import transformers as hf
import torch as pt

import typing as T

from language_model.utils.get_name_of_subclass import get_name_of_subclass

from language_model.optimizer import Optimizer


@dc.dataclass
class Scheduler(ez.Config):
    algorithm: str = None
    """The type of scheduler"""

    def __post_init__(self):
        self.algorithm = get_name_of_subclass(self, Scheduler)
        self.scheduler = None

    def schedule(self, optimizer):
        raise TypeError("A subclass of Scheduler must be used to specify a learning rate schedule.")


@dc.dataclass
class LinearWarmupSchedule(Scheduler):
    num_warmup_steps: int = 100
    """The number of training steps where the learning rate is linearly increased. Setting to 0 results in no warmup."""
    start_factor: float = 0.01
    """The factor by which the learning rate is multiplied at the start of warmup."""
    end_factor: float = 1.0
    """The factor by which the learning rate is multiplied at the end of warmup."""

    def schedule(self, optimizer):
        assert isinstance(optimizer, Optimizer), \
            f"An Optimizer object attached to trainable model parameters should be provided to Scheduler constructor, but got {optimizer}"
        self.scheduler = pt.optim.lr_scheduler.LinearLR(
            optimizer=optimizer.optimizer,
            start_factor=self.start_factor,
            end_factor=self.end_factor,
            total_iters=self.num_warmup_steps,)
        return self.scheduler


if __name__ == '__main__':

    schedule_config = LinearWarmupSchedule()
    print(schedule_config.configured.json())
