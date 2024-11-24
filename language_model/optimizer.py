import ezpyzy as ez
import dataclasses as dc
import pathlib as pl
import json
import transformers as hf
import bitsandbytes as bnb
import torch as pt

from language_model.utils.get_name_of_subclass import get_name_of_subclass


@dc.dataclass
class Optimizer(ez.ImmutableConfig):
    algorithm: str = None
    """The optimizer algorithm to use for training."""
    learning_rate: float = None
    """The learning rate to use for training."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm = get_name_of_subclass(self, Optimizer)

    def construct_optimizer(self, model):
        raise TypeError("A subclass of Optimizer must be used to specify an optimizer.")


@dc.dataclass
class Adafactor(Optimizer):
    eps: tuple[float, float] = (1e-30, 1e-3)
    """Regularization constants for square gradient and parameter scale respectively"""
    clip_threshold: float = 1.0
    """Threshold of root mean square of final gradient update"""
    decay_rate: float = -0.8
    """Coefficient used to compute running averages of square"""
    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""
    weight_decay: float = 0.0
    """Weight decay (L2 penalty)"""
    scale_parameter: bool = True
    """If True, learning rate is scaled by root mean square"""

    def construct_optimizer(self, model):
        return hf.optimization.Adafactor(
            model.parameters(),
            lr=self.learning_rate,
            eps=self.eps,
            clip_threshold=self.clip_threshold,
            decay_rate=self.decay_rate,
            beta1=self.beta1,
            weight_decay=self.weight_decay,
            scale_parameter=self.scale_parameter)


@dc.dataclass
class AdamW(Optimizer):
    betas: tuple[float, float] = (0.9, 0.999)
    """The beta values are the decay rates of the first and second-order moment of the optimizer."""
    eps: float = 1e-8
    """The epsilon value prevents division by zero in the optimizer."""
    weight_decay: float = 1e-2
    """The weight decay value for the optimizer."""
    quantization: str|None = None
    """The quantization scheme to use for the optimizer, either None or '8bit'."""

    def construct_optimizer(self, model):
        if self.quantization is None:
            optimizer = hf.optimization.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay)
        elif self.quantization == '8bit':
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Invalid quantization for Optimizer: {self.quantization}")
        return optimizer

if __name__ == '__main__':

    optconfig = Adafactor(clip_threshold=2.0)
    print(optconfig.json())