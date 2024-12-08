

import ezpyzy as ez

import dataclasses as dc

import language_model.scheduler as sch
import language_model.optimizer as opt


@dc.dataclass
class Training(ez.Config):
    epochs: int | float = 1
    """The number of epochs to train for. If a float, the model will train for that fraction of the dataset."""
    optimizer: opt.Optimizer = opt.Adam()
    """The optimizer to use for training, which determines the optimization algorithm, learning rate, weight decay, etc."""
    scheduler: sch.Scheduler = sch.LinearWarmupSchedule()
    """The learning rate algorithm to use for training, which determines how the learning rate changes over time."""
    batch_size: int = 16
    """The effective batch size to use for training. Divide by gradient_accumulation_steps to get the actual batch size used on the hardware."""
    physical_batch_size: int = 1
    """The actual batch size sent to hardware during training (i.e. batch_size // gradient_accumulation_steps)."""
    gradient_accumulation_steps: int = 16
    """The number of times to acummulate gradients before updating the model (i.e. the actual batch size sent to hardware is batch_size // gradient_accumulation_steps)."""
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing to reduce memory usage. Does not affect training performance."""
    shuffle_data: bool = True
    """Whether to shuffle data samples before each training epoch."""
    resume_previous_training: bool = False
    """Whether to resume training, loading/keeping the optimizer and scheduler states from previous training runs."""
    current_epoch: int = 0
    """The epoch of current training (mostly used for resuming training)"""
    current_step: int = 0
    """The current step of training the current epoch (mostly used for resuming training)"""
    num_kept_examples: int = 0
    """The number of examples sent to training runs to be kept in a list."""

    def __post_init__(self):
        super().__post_init__()
        with self.configured.not_configuring():
            if self.configured.has.batch_size and self.configured.has.physical_batch_size:
                assert self.batch_size % self.physical_batch_size == 0, \
                    "batch_size must be divisible by physical_batch_size"
                if not self.configured.has.gradient_accumulation_steps:
                    self._gradient_accumulation_steps = self.batch_size // self.physical_batch_size
                else:
                    assert self.batch_size % self.gradient_accumulation_steps == 0, \
                        "batch_size must be divisible by gradient_accumulation_steps"
                    assert self.physical_batch_size == self.batch_size // self.gradient_accumulation_steps, \
                        "physical_batch_size must be equal to batch_size // gradient_accumulation_steps"
            elif self.configured.has.batch_size and self.configured.has.gradient_accumulation_steps:
                assert self.batch_size % self.gradient_accumulation_steps == 0, \
                    "batch_size must be divisible by gradient_accumulation_steps"
                if not self.configured.has.physical_batch_size:
                    self._physical_batch_size = self.batch_size // self.gradient_accumulation_steps
                else:
                    assert self.physical_batch_size == self.batch_size // self.gradient_accumulation_steps, \
                        "physical_batch_size must be equal to batch_size // gradient_accumulation_steps"
            elif self.configured.has.physical_batch_size and self.configured.has.gradient_accumulation_steps:
                assert self.physical_batch_size % self.gradient_accumulation_steps == 0, \
                    "physical_batch_size must be divisible by gradient_accumulation_steps"
                if not self.configured.has.batch_size:
                    self._batch_size = self.physical_batch_size * self.gradient_accumulation_steps
                else:
                    assert self.batch_size == self.physical_batch_size * self.gradient_accumulation_steps, \
                        "batch_size must be equal to physical_batch_size * gradient_accumulation_steps"
            elif self.configured.has.batch_size:
                self._physical_batch_size = self.batch_size
                self._gradient_accumulation_steps = 1
            elif self.configured.has.physical_batch_size:
                self._batch_size = self.physical_batch_size
                self._gradient_accumulation_steps = 1
            elif self.configured.has.gradient_accumulation_steps:
                self._batch_size = self.gradient_accumulation_steps
                self._physical_batch_size = 1
        self.examples = []


    def _set_batch_size(self, value):
        if not self.configured.initialized:
            return value
        if (self.configured.has.physical_batch_size
            and not self.configured.has.gradient_accumulation_steps
        ):
            assert value % self.physical_batch_size == 0, \
                "batch_size must be divisible by physical_batch_size"
            self._gradient_accumulation_steps = value // self.physical_batch_size
        elif (self.configured.has.gradient_accumulation_steps
              and not self.configured.has.physical_batch_size
        ):
            assert value % self.gradient_accumulation_steps == 0, \
                "batch_size must be divisible by gradient_accumulation_steps"
            self._physical_batch_size = value // self.gradient_accumulation_steps
        elif (not self.configured.has.physical_batch_size
              and not self.configured.has.gradient_accumulation_steps
        ):
            self._physical_batch_size = value
            self._gradient_accumulation_steps = 1
        else:
            assert value % self.gradient_accumulation_steps == 0, \
                "batch_size must be divisible by gradient_accumulation_steps"
            self._physical_batch_size = value // self.gradient_accumulation_steps
        return value

    def _set_physical_batch_size(self, value):
        if not self.configured.initialized:
            return value
        if (self.configured.has.gradient_accumulation_steps
            and not self.configured.has.batch_size
        ):
            self._physical_batch_size = self.gradient_accumulation_steps * value
        elif (self.configured.has.batch_size
              and not self.configured.has.gradient_accumulation_steps
        ):
            assert self.batch_size % value == 0, \
                "batch_size must be divisible by physical_batch_size"
            self._gradient_accumulation_steps = self.batch_size // value
        elif (not self.configured.has.batch_size
              and not self.configured.has.gradient_accumulation_steps
        ):
            self._physical_batch_size = value
            self._gradient_accumulation_steps = 1
        else:
            assert self.batch_size % value == 0, \
                "batch_size must be divisible by physical_batch_size"
            self._gradient_accumulation_steps = self.batch_size // value
        return value

    def _set_gradient_accumulation_steps(self, value):
        if not self.configured.initialized:
            return value
        if (self.configured.has.physical_batch_size
            and not self.configured.has.batch_size
        ):
            self._gradient_accumulation_steps = value
        elif (self.configured.has.batch_size
              and not self.configured.has.physical_batch_size
        ):
            self._physical_batch_size = self.batch_size // value
        elif (not self.configured.has.batch_size
              and not self.configured.has.physical_batch_size
        ):
            self._gradient_accumulation_steps = value
            self._physical_batch_size = 1
        else:
            self._physical_batch_size = self.batch_size // value
        return value


if __name__ == '__main__':

    training = Training(batch_size=64, physical_batch_size=16)
    print(training.configured.json())