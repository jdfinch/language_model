


import ezpyzy as ez
import dataclasses as dc
import pathlib as pl
import json
import transformers as hf
import torch as pt

# black magic type hinting: sneak the "base" decorator into "dataclass" var name
from dataclasses import dataclass; vars().update(dataclass=ez.config)

hf.logging.set_verbosity_error()


def get_name_of_subclass_for_field(self, base_cls, field_name):
    for supercls in self.__class__.__mro__:
        if issubclass(supercls, base_cls):
            field = supercls.__name__
            if getattr(self, field_name, None) is None:
                setattr(self, field_name, field)
            else:
                assert getattr(self, field_name) == field, \
                    f"Conflicting {field_name}: {repr(getattr(self, field_name))} was given but {field} was discovered in the class hierarchy as the implemented {field_name}."
            break
    else:
        raise TypeError(f"A subclass of {base_cls.__name__} must be used to specify a {field_name}.")
    return None

import language_model.generate as generate
import language_model.optimizer as optimizer
import language_model.scheduler as scheduler


@dataclass
class Sequence(ez.Config):
    tokenizer: str = None
    format: dict[str, str] = None
    """The format to use for training and generation. Use TokIn and TokOut objects to specify __template_slots__ for input and output sequences (or just specify __template_slots__ like #[input=myinput, trunc_side=L, trunc_rank=1.0]# or #[output=myoutput, max=50, min=20]# and customize truncation for sequences exceeding the max_sequence_length. When training and generating, data is passed in to fill __template_slots__ in this format template."""
    max_length: int = 1024
    """The maximum token length of sequences the model trains on or can be fed as input for generation."""
    trunc_segments_side: str = 'L'
    """When entire segments (e.g. dialogue turns) must be removed, this determines which side they are removed from. 'L' for left, 'R' for right. Note that segments corresponding to templates with trunc_segment=False will never be removed."""
    max_segments: int | None = None
    """The maximum number of segments to keep in a token_ids. If None, no segment pruning will be performed. The primary use case for setting max_segments is to trim extremely long sequences by a number-of-segments threshold BEFORE any tokenization is performed, which can improve preprocessing efficiency."""
    pad_side: str = 'L'
    """The side to pad sequences on. 'L' for left, 'R' for right."""
    pad_to_multiple_of: int = 8
    """Pads sequences so that total token_ids lengths are a multiple of this value, which can improve performance on GPU."""
    # tokenization_num_processes: int|float = 1
    # """The number of processes to use for tokenization. Inputting a float will use that fraction of the available CPUs. Strangely, setting this to 2.0 (double the CPUs) is sometimes fastest."""
    # tokenization_batches_per_chunk: int|None = None
    # """The number of batches a single process will tokenize for multiprocessed tokenization. Tune for more fast."""


@dataclass
class Train(ez.Config):
    epochs: int | float = 1
    """The number of epochs to train for. If a float, the model will train for that fraction of the dataset."""
    optimizer: optimizer.Optimizer = ez.default(optimizer.Adafactor())
    """The optimizer to use for training, which determines the optimization algorithm, learning rate, weight decay, etc."""
    scheduler: scheduler.Scheduler = ez.default(scheduler.LinearWarmupSchedule())
    """The learning rate scheduler to use for training, which determines how the learning rate changes over time."""
    batch_size: int = 16
    """The effective batch size to use for training. Divide by gradient_accumulation_steps to get the actual batch size used on the hardware."""
    physical_batch_size: int | None = 1
    """The actual batch size sent to hardware during training (i.e. batch_size // gradient_accumulation_steps)."""
    gradient_accumulation_steps: int | None = None
    """The number of times to acummulate gradients before updating the model (i.e. the actual batch size sent to hardware is batch_size // gradient_accumulation_steps)."""
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing to reduce memory usage. Does not affect training performance."""
    shuffle_data: bool = True
    """Whether to shuffle data samples before each training epoch."""

    def _set_batch_size(self, value):
        if value is None:
            assert (hasattr(self, 'args')
                    and self.args.physical_batch_size is not None
                    and self.args.gradient_accumulation_steps is not None
            ), "batch_size must be set if physical_batch_size and gradient_accumulation_steps are not"
            return self.args.physical_batch_size * self.args.gradient_accumulation_steps
        else:
            return value

    def _set_physical_batch_size(self, value):
        if value is None:
            return self.batch_size // getattr(self, 'gradient_accumulation_steps', 1)
        else:
            self.__dict__['gradient_accumulation_steps'] = self.batch_size // value
            return value

    def _set_gradient_accumulation_steps(self, value):
        if value is None:
            return self.batch_size // getattr(self, 'physical_batch_size', 1)
        else:
            self.__dict__['physical_batch_size'] = self.batch_size // value
            return value


@dataclass
class LoRA(ez.ImmutableConfig):
    rank: int = 2
    """The LoRA rank to use. If None, no LoRA adapter be used at all."""
    modules: tuple[str] = None
    """The modules to apply LoRA adapters to. Modules must be names from model architecture and can be found using print(model.model)"""
    alpha: float | None = None
    """The alpha value to use for LoRA. If None, the alpha will be set to 2*lora (recommended)."""
    dropout: float = 0.0
    """The dropout rate to use when training LoRA adapter weights."""


@dataclass
class LMConfig(ez.ImmutableConfig):
    """Configuration for a Llama model"""
    model_to_load: str | None = None
    """Path to a custom model to load. The base will be used instead if this is None."""
    model_base: str | None = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    """The base model repository ID. If no model_to_load is specified, the base will be loaded from this ID."""
    lora_merge_on_load: bool = False
    """Whether to merge the LoRA adapter into the original model weights on load."""
    quantization: str | None = 'nf4'
    """Quantization mode to use. If None, no quantization will be used (half-precision bf16)."""
    sequence_params: Sequence = ez.default(Sequence())
    """The template and configuration to use for tokenization"""
    adapters: tuple[LoRA] | None = ez.default((LoRA(),))
    """The LoRA adapters to use for the model (leave this empty/None for full fine-tuning)."""
    train_params: Train|None = ez.default(Train())
    """Hyperparameters and configuration for training the model."""
    generate_params: generate.Generate = ez.default(generate.Greedy())
    """Hyperparameters and configuration for generating text from the model."""
    resume_training: bool = False
    """Resumes training from the checkpoint saved at model_to_load by loading the optimizer state."""
    load_locally_saved_models_only: bool = False
    """Whether to load models only from the local cache, not from the Hugging Face model hub."""
    hardware_device: str = 'cuda'
    """The hardware device to use for training and/or generation, such as 'cuda', 'cuda:7', or 'cpu'."""

    def __post_init__(self):
        # If base is a path to an experiment or iteration folder, load the config from that folder
        if isinstance(self.base, (str, pl.Path)):
            config_path = pl.Path(self.base).expanduser()
            if config_path.is_dir() and config_path.exists():
                if self.model_to_load is None:
                    self.args.model_to_load = str(config_path)
                if not (config_path / 'config.json_e').exists() and (config_path.parent / 'config.json_e').exists():
                    self.args.base = config_path.parent / 'config.json_e'
        # Load the config
        ez.Config.__post_init__(self)
        if self.model_to_load is None:
            self.model_to_load = self.model_base
        if self.sequence_params.tokenizer is None:
            self.sequence_params.tokenizer = self.model_base
        # validate that any loaded adapter or model has the right base model specified
        model_to_load_path = pl.Path(self.model_to_load).expanduser()
        if (model_to_load_path / 'adapter_config.json_e').exists():
            hf_adapter_config = json.loads((model_to_load_path / 'adapter_config.json_e').read_text())
            if 'base_model_name_or_path' in hf_adapter_config:
                base = hf_adapter_config['base_model_name_or_path']
                if self.model_base is None:
                    self.model_base = base
                elif self.model_base != base:
                    raise ValueError(f"Model base {self.model_base} does not match adapter base {base}")
        elif (model_to_load_path / 'base.json_e').exists():
            hf_model_config = json.loads((model_to_load_path / 'base.json_e').read_text())
            if '_name_or_path' in hf_model_config:
                base = hf_model_config['_name_or_path']
                if self.model_base is None:
                    self.model_base = base
                elif self.model_base != base:
                    raise ValueError(f"Model base {self.model_base} does not match loaded base base {base}")
        else:
            self.model_to_load = self.model_base
        # validate hyperparam values
        assert self.quantization in {'nf4', 'nf4dq', 'int8', 'bf16', None}, "quantization must be one of 'nf4', 'nf4dq', 'int8', 'bf16', or None"

    def _set_quantization(self, value):
        quantization_options = {'nf4', 'nf4dq', 'int8', 'bf16', None}
        assert value in quantization_options, \
            f"quantization must be one of {sorted(quantization_options)}"
        if value is None:
            value = 'bf16'
        return value