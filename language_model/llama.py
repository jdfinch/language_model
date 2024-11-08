

import transformers as hf
import torch as pt
import dataclasses as dc
import textwrap as tw
import pathlib as pl
import itertools as it
import json
import copy as cp
from language_model.utils.config import config, Config, defaults_from_self
from language_model.utils.batch import batched, batching
from language_model.utils.bind import bind
from language_model.utils.peek import peek
from language_model.utils.default import default
import language_model.llama3_format as llama3format
import language_model.tokenizer as tok

import typing as T

def _imports(): pass

# black magic type hinting: sneak the "config" decorator into "dataclass" var name
from dataclasses import dataclass; vars().update(dataclass=config)


@dataclass
class LlamaHypers(Config):
    """Configuration for a Llama model, representing all performance-affecting parameters."""
    model_to_load: str | None = None
    """Path to a custom model to load. The base will be used instead if this is None."""
    base: str|None = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    """The base model repository ID. If no model_to_load is specified, the base will be loaded from this ID."""
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
    format: dict = default({
        'system (trunc_content=False, trunc_segment=False)':
            '''<|begin_of_text|><|start_header_id|>system<|end_header_id>\n\n#[input=text]#<|eot_id|>''',
        'user (trunc_content=False, trunc_segment=True)':
            '''<|start_header_id|>user<|end_header_id>\n\n#[input=text]#<|eot_id|>''',
        'assistant (trunc_content=False, trunc_segment=True)':
            '''<|start_header_id|>assistant<|end_header_id>\n\n#[output=text]#''',
        'assistant_history (trunc_content=False, trunc_segment=True)':
            '''<|start_header_id|>assistant<|end_header_id>\n\n#[input=text]#<|eot_id|>''',
        'info (trunc_segment=False, trunc_content=True)':
            '''<|start_header_id|>user<|end_header_id>\n\n#[input=text]#<|eot_id|>'''
    })
    """The format to use for training and generation. Use TokIn and TokOut objects to specify slots for input and output sequences (or just specify slots like #[input=myinput, trunc_side=L, trunc_rank=1.0]# or #[output=myoutput, max=50, min=20]# and customize truncation for sequences exceeding the max_sequence_length. When training and generating, data is passed in to fill slots in this format template."""
    max_sequence_length: int = 1024
    """The maximum token length of sequences the model trains on or can be fed as input for generation."""
    trunc_segments_side: str = 'L'
    """When entire segments (e.g. dialogue turns) must be removed, this determines which side they are removed from. 'L' for left, 'R' for right. Note that segments corresponding to templates with trunc_segment=False will never be removed."""
    max_segments: int|None = None
    """The maximum number of segments to keep in a sequence. If None, no segment pruning will be performed. The primary use case for setting max_segments is to trim extremely long sequences by a number-of-segments threshold BEFORE any tokenization is performed, which can improve preprocessing efficiency."""
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
    warmup_steps: int = 16
    """The number of warmup steps to use for the learning rate scheduler."""
    warmup_schedule: tuple[float, float] = (0.01, 1.0)
    """The start and end factors to multiply the learning rate by, corresponding to the start and end of warmup."""
    max_gradient_norm: float|None = 1.0
    """The maximum gradient norm to clip to."""
    weight_decay: float = 0.0
    """The weight decay to use for training."""
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
    tokenizer_id: str = None
    """The repository ID of the tokenizer to use. If None, the tokenizer will be loaded from the base ID."""

    def __post_init__(self):
        # apply rules for specification of what model and tokenizer should be loaded
        if self.model_to_load is None:
            self.model_to_load = self.base
        if self.tokenizer_id is None:
            self.tokenizer_id = self.base
        # Load config from file if specified, overriding loaded args with args passed to constructor
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
        else:
            self.model_to_load = self.base
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
    """Path to a specific checkpoint to load. This overrides base and model_to_load to resume training from a specific checkpoint, including the optimizer slots."""
    load_locally_saved_models_only: bool = False
    """Whether to load models only from the local cache, not from the Hugging Face model hub."""
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing to reduce memory usage."""
    physical_train_batch_size: int | None = 1
    """The actual batch size sent to hardware during training. Either this or gradient_accumulation_steps can be set, not both."""
    gradient_accumulation_steps: int|None = None
    """The number of times to acummulate gradients before updating the model (i.e. the actual batch size sent to hardware is train_batch_size / gradient_accumulation_steps). Either this or physical_train_batch_size can be set, not both."""
    pad_side: str = 'L'
    """The side to pad sequences on. 'L' for left, 'R' for right."""
    pad_to_multiple_of: int = 8
    """Pads sequences so that total sequence lengths are a multiple of this value, which can improve performance on GPU."""
    gen_batch_size: int = None
    """The batch size to use for generation. Defaults to the actual train batch size."""
    ppl_batch_size: int = 1
    """The batch size to use for perplexity calculation."""
    device: str = 'cuda'
    """The device to use for training and/or generation, such as 'cuda', 'cuda:7', or 'cpu'."""

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
        assert self.max_sequence_length % self.pad_to_multiple_of == 0, \
            f"max_sequence_length (got {self.max_sequence_length}) must be divisible by pad_to_multiple_of (got {self.pad_to_multiple_of})"


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
                    bnb_4bit_use_double_quant=False),
                torch_dtype=pt.bfloat16)
        elif self.quantization == 'nf4dq':
            quantization_kwargs = dict(
                quantization_config=hf.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=pt.bfloat16,
                    bnb_4bit_use_double_quant=True),
                torch_dtype=pt.bfloat16)
        elif self.quantization == 'int8':
            quantization_kwargs = dict(
                load_in_8bit=True,
                torch_dtype=pt.bfloat16)
        elif self.quantization == 'bf16':
            quantization_kwargs = dict(
                torch_dtype=pt.bfloat16)
        else:
            raise ValueError(f"Invalid quantization mode: {self.quantization}")
        # Load the model and tokenizer
        self.tokenizer = tok.Tokenizer( # wrapper around hf tokenizer
            tokenizer=self.tokenizer_id,
            local_files_only=self.load_locally_saved_models_only)
        self.model: hf.LlamaForCausalLM = hf.AutoModelForCausalLM.from_pretrained(
            self.model_to_load,
            **quantization_kwargs,
            device_map=self.device,
            local_files_only=self.load_locally_saved_models_only)
        self.template: tok.TokenTemplateCollection = self.tokenizer.templatize(
            self.format,
            max_length=self.max_sequence_length,
            pad_to_same_length=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            pad_side=self.pad_side,
            trunc_segments_side=self.trunc_segments_side,
            max_segments=self.max_segments)


    def save(self, path: str|pl.Path=None, save_as_checkpoint=False):
        return super().save(path)

    @defaults_from_self
    def preprocess(self,
        sequences: T.Iterable[T.Iterable[dict[str, str]]],
        batch_size: int = None,
        max_sequence_length: int = None,
        pad_to_multiple_of: int = None,
        pad_side: str = None,
        trunc_segments_side: str = None,
        max_segments: int = None
    ) -> list[tok.TokenSequenceBatch]:
        template = cp.copy(self.template)
        template.max_length = max_sequence_length
        template.pad_to_multiple_of = pad_to_multiple_of
        template.pad_side = pad_side
        template.trunc_segments_side = trunc_segments_side
        template.max_segments = max_segments
        if batch_size is None:
            batches = [sequences]
        else:
            batches = batched(sequences, batch_size)
        preprocessed = [self.template.fill(batch) for batch in batches]
        return preprocessed

    @defaults_from_self
    def generate(self,
        sequences: T.Iterable[T.Iterable[dict[str, str]]],
        max_sequence_length: int = None,
        pad_to_multiple_of: int = None,
        pad_side: str = None,
        trunc_segments_side: str = None,
        max_segments: int = None,
        max_output_length: int = None,
        gen_batch_size: int = None,
        num_beams: int = None,
        temperature: float = None,
        sampled_generation: bool = None,
        top_p: float = None,
        top_k: int = None,
        repetition_penalty: float = None,

    ):
        batches = self.preprocess(
            sequences,
            batch_size=gen_batch_size,
            max_sequence_length=max_sequence_length,
            pad_to_multiple_of=pad_to_multiple_of,
            pad_side=pad_side,
            trunc_segments_side=trunc_segments_side,
            max_segments=max_segments)
        generation_config = hf.GenerationConfig(
            max_length=max_sequence_length,
            max_new_tokens=max_output_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.tokenizer.pad_token_id,
        )
        generated_results = []
        for batch in batches:
            tokens = batch.dict(seq_type=bind(pt.tensor)(dtype=pt.long, device=self.device)) # noqa
            outputs = self.model.generate(**tokens, generation_config=generation_config)
            generated_tokens = [output[len(input):] for input, output in zip(batch, outputs)]
            generated_texts = [self.tokenizer.decode(tokens) for tokens in generated_tokens]
            generated_results.extend(generated_texts)
        return generated_results


if __name__ == '__main__':
    llama = Llama()

    dialogues = [[
        dict(temp='system', text='You are a helpful assistant.'),
        dict(temp='assistant', text='Hi! How can I help you today?'),
        dict(temp='user', text='What is the capital of France?'),
        dict(temp='assistant', text='The capital of France is Paris.'),
        dict(temp='user', text='Thank you!'),
        dict(temp='assistant', text=None),
    ], [
        dict(temp='system', text='You are an evil robot.'),
        dict(temp='assistant', text='Hello! How can I destroy you today?'),
        dict(temp='user', text='What is the capital of France?'),
        dict(temp='assistant', text='France is a country of no importance.'),
        dict(temp='user', text='Thank you!'),
        dict(temp='assistant', text=None),
    ]]

    preprocessed = llama.preprocess(dialogues)
    print(preprocessed[0].display())
    generated = llama.generate(dialogues)
    print('\n\n'.join(generated))





























