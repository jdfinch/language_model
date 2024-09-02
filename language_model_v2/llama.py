

import transformers as hf
import torch as pt
import dataclasses as dc
import textwrap as tw
import pathlib as pl
import itertools as it
import json
from language_model_v2.utils.config import config, Config
from language_model_v2.utils.batch import batched, batching
from language_model_v2.utils.peek import peek
from language_model_v2.utils.default import default
import language_model_v2.llama3_format as llama3format
import language_model_v2.tokenizer as tok

import typing as T

def _imports(): pass

# black magic type hinting: sneak the "config" decorator into "dataclass" var name
from dataclasses import dataclass; vars().update(dataclass=config)


@dataclass
class LlamaHypers(Config):
    """Configuration for a Llama model, representing all performance-affecting parameters."""
    base: str|None = "meta-llama/Meta-Llama-3.1-{param_magnitude}-Instruct"
    """The base model repository ID. If no model_to_load is specified, the base will be loaded from this ID."""
    model_to_load: str|None = None
    """Path to a custom model to load. The base will be used instead if this is None."""
    param_magnitude: str | None = '8B'
    """The magnitude of the model parameters, replacing {param_magnitude} in the base."""
    lora_merge_on_load: bool = False
    """Whether to merge the LoRA adapter into the original model weights on load."""
    quantization: str | None = 'nf4'
    """Quantization mode to use. If None, no quantization will be used (half-precision bf16)."""
    lora: int | None = 2
    """The LoRA rank to use. If None, no LoRA adapter be used at all."""
    lora_applied_to: tuple[str] = (
    'embed_tokens', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head')
    """The modules to apply LoRA adapters to. Modules must be names from model architecture and can be found using print(model.model)"""
    lora_alpha: float | None = None
    """The alpha value to use for LoRA. If None, the alpha will be set to 2*lora (recommended)."""
    lora_dropout: float = 0.0
    """The dropout rate to use when training LoRA adapter weights."""
    lora_adapters: tuple = ()
    """EXPERIMENTAL, DO NOT TOUCH"""
    format: str = tw.dedent(
        f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

        {tok.TokIn()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {tok.TokOut()}"""
    ).lstrip()
    """The format to use for training and generation. Use TokIn and TokOut objects to specify slots for input and output sequences (or just specify slots like #[input=myinput, trunc_side=L, trunc_rank=1.0, is_label=False]# and customize truncation for sequences exceeding the max_sequence_length. When training and generating, data is passed in to fill slots in this format template."""
    max_sequence_length: int = 1024
    """The maximum token length of sequences the model trains on or can be fed as input for generation."""
    max_output_length: int = 512
    """The maximum number of tokens the model will generate."""
    epochs: int | float = 1
    """The number of epochs to train for. If a float, the model will train for that fraction of the dataset."""
    train_batch_size: int = 16
    """The effective batch size to use for training. Divide by gradient_accumulation_steps to get the actual batch size used on the hardware."""
    learning_rate: float = 1e-2
    """The learning rate to use for training."""
    optimizer: str = 'adafactor'
    """The optimizer to use for training."""
    warmup_steps: int = 0
    """The number of warmup steps to use for the learning rate scheduler."""
    max_gradient_norm: float|None = 1.0
    """The maximum gradient norm to clip to."""
    weight_decay: float = 0.0
    """The weight decay to use for training."""
    scheduler_type: str = 'constant'
    """The type of learning rate scheduler to use."""
    repetition_penalty: float = 1.2
    """The repetition penalty to use for generation."""
    num_beams: int = 1
    """The number of beams to use for generation."""
    temperature: float = 0.6
    """The temperature to use for generation."""
    sampled_generation: bool = False
    """Whether to use sampled generation instead of beam search."""
    top_p: float = 0.9
    """The top-p value to use for generation."""
    top_k: int = 50
    """The top-k value to use for generation."""
    tokenizer_repo_id: str = None
    """The repository ID of the tokenizer to use. If None, the tokenizer will be loaded from the base ID."""

    def __post_init__(self):
        # Load config from file if specified, overriding loaded args with args passed to constructor
        if self.model_to_load is not None:
            model_to_load_path = pl.Path(self.model_to_load).expanduser()
            if model_to_load_path.exists():
                loaded_config = json.loads((model_to_load_path / 'emory_config.json').read_text())
                specified_config = self.__config__  # grab the args passed to the constructor
                merged_config = loaded_config | specified_config
                vars(self).update(merged_config)
            # validate that any loaded adapter or model has the right base model specified
            if (model_to_load_path / 'adapter_config.json').exists():
                hf_adapter_config = json.loads((model_to_load_path / 'adapter_config.json').read_text())
                if 'base_model_name_or_path' in hf_adapter_config:
                    base = hf_adapter_config['base_model_name_or_path']
                    if self.base is None:
                        self.base = base
                    elif self.base != base:
                        raise ValueError(f"Model base {self.base} does not match adapter base {base}")
            elif (model_to_load_path / 'config.json').exists():
                hf_model_config = json.loads((model_to_load_path / 'config.json').read_text())
                if '_name_or_path' in hf_model_config:
                    base = hf_model_config['_name_or_path']
                    if self.base is None:
                        self.base = base
                    elif self.base != base:
                        raise ValueError(f"Model base {self.base} does not match loaded config base {base}")
        # apply rules for specification of what model and tokenizer should be loaded
        if '{param_magnitude}' in self.base:
            self.base = self.base.format(param_magnitude=str(self.param_magnitude))
        if self.model_to_load is None:
            self.model_to_load = self.base
        if self.tokenizer_repo_id is None:
            self.tokenizer_repo_id = self.base
        # calculate default hyperparam values
        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora # this heuristic has now been validated many times
        if self.quantization is None:
            self.quantization = 'bf16'
        # validate hyperparam values
        assert self.quantization in {'nf4', 'nf4dq', 'int8', 'bf16', None}, "quantization must be one of 'nf4', 'nf4dq', 'int8', 'bf16', or None"

    def save(self, path: str|pl.Path=None):
        """Serialize the config as JSON (and save to disk if a file path is passed)."""
        serial_config = json.dumps(dc.asdict(self), indent=4)
        if path:
            path = pl.Path(path).expanduser()
            path.write_text(serial_config)
        return serial_config


@dataclass
class LlamaConfig(LlamaHypers):
    """Complete configuration for a Llama model, including implementation-level params."""
    checkpoint_to_load: str|None = None
    """Path to a specific checkpoint to load. This overrides base and model_to_load to resume training from a specific checkpoint, including the optimizer state."""
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing to reduce memory usage."""
    physical_train_batch_size: int | None = 1
    """The actual batch size sent to hardware during training. Either this or gradient_accumulation_steps can be set, not both."""
    gradient_accumulation_steps: int|None = None
    """The number of times to acummulate gradients before updating the model (i.e. the actual batch size sent to hardware is train_batch_size / gradient_accumulation_steps). Either this or physical_train_batch_size can be set, not both."""
    pad_to_multiple_of: int = 8
    """Pads sequences so that total sequence lengths are a multiple of this value, which can improve performance on GPU."""
    save_checkpoint_every_x_epochs: float = 1.0
    """The frequency to save checkpoints, in epochs."""
    clean_up_checkpoint_after_training: bool = True
    """Whether to clean up checkpoints after training."""
    gen_batch_size: int = None
    """The batch size to use for generation. Defaults to the actual train batch size."""
    ppl_batch_size: int = 1
    """The batch size to use for perplexity calculation."""
    dynamic_tokenization: bool = True
    """Whether to dynamically send tokens to GPU batch-by-batch to save on memory usage."""

    def __post_init__(self):
        super().__post_init__()
        # calculate and validate hardware-level batch sizes and gradient accumulation
        if self.physical_train_batch_size is None and self.gradient_accumulation_steps is None:
            self.physical_train_batch_size = self.train_batch_size
            self.gradient_accumulation_steps = 1
        elif self.physical_train_batch_size is not None:
            assert self.train_batch_size % self.physical_train_batch_size == 0, "train_batch_size must be divisible by physical_train_batch_size"
            self.gradient_accumulation_steps = self.train_batch_size // self.physical_train_batch_size
        elif self.gradient_accumulation_steps is not None:
            assert self.train_batch_size % self.gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
            self.physical_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps
        else:
            assert (self.train_batch_size // self.gradient_accumulation_steps == self.physical_train_batch_size
                    and self.train_batch_size % self.gradient_accumulation_steps == 0
                   ), "train_batch_size must be divisible by physical_train_batch_size and their quotient must equal gradient_accumulation_steps (the easiest fix is to specify ONLY ONE of physical_train_batch_size or gradient_accumulation_steps)"
        if self.gen_batch_size is None:
            self.gen_batch_size = self.physical_train_batch_size


class Llama(LlamaConfig):
    """A Llama model configured for training and/or generation."""

    def __post_init__(self):
        # Config post init responsible for loading config from file (if applicable) and setting defaults
        super().__post_init__()
        # Set up the quantization config
        if self.quantization == 'nf4':
            quantization_kwargs = dict(
                quantization_config=hf.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=pt.bfloat16,
                    bnb_4bit_use_double_quant=False
                ),
                torch_dtype=pt.bfloat16,
                device_map='auto')
        elif self.quantization == 'nf4dq':
            quantization_kwargs = dict(
                quantization_config=hf.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=pt.bfloat16,
                    bnb_4bit_use_double_quant=True
                ),
                torch_dtype=pt.bfloat16,
                device_map='auto')
        elif self.quantization == 'int8':
            quantization_kwargs = dict(
                load_in_8bit=True,
                torch_dtype=pt.bfloat16,
                device_map='auto')
        elif self.quantization == 'bf16':
            quantization_kwargs = dict(
                torch_dtype=pt.bfloat16,
                device_map='auto')
        else:
            raise ValueError(f"Invalid quantization mode: {self.quantization}")
        # Load the model and tokenizer
        self.tokenizer = tok.Tokenizer(self.tokenizer_repo_id) # wrapper around hf tokenizer
        self.model: hf.LlamaForCausalLM = hf.AutoModelForCausalLM.from_pretrained(
            self.model_to_load, **quantization_kwargs)
        self.template = self.tokenizer.templatize(self.format)


    def preprocess(self, data=None, /, **datas) -> tok.TokenSequence | T.Generator[tok.TokenSequence, None, None]:
        return_single_sequence = False
        if isinstance(data, dict):
            data = [data]
            return_single_sequence = True
        elif datas and isinstance(next(iter(datas.values())), (str, tok.TokenSequence)):
            data = [datas]
            return_single_sequence = True
        elif data is None:
            data = (dict(zip(datas, values)) for values in zip(*datas.values()))
        first, data = peek(data)
        if first is None:
            return () # noqa
        elif set(first) == set(self.template.slots):
            # preprocess for training
            batch_size = self.physical_train_batch_size
            def generator(data=data, batch_size=batch_size):
                for data_batch in batching(data, size=batch_size):
                    batch = self.template.fill(data_batch, max_length=self.max_sequence_length)
                    yield batch
            return generator() if not return_single_sequence else next(generator())
        else:
            # preprocess for generation
            batch_size = self.gen_batch_size
            def generator(data=data, batch_size=batch_size):
                for data_batch in batching(data, size=batch_size):
                    batch = self.template.fill(data_batch, max_length=self.max_sequence_length)
                    yield batch
            return generator() if not return_single_sequence else next(generator())
























def main():
    config = LlamaConfig()
    print(config.save())

if __name__ == '__main__':
    main()
