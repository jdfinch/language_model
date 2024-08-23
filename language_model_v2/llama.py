

import transformers as tf
import torch as pt
from pprint import pp
import pathlib as pl
import json
from language_model_v2.utils.settings import settings # noqa
from language_model_v2.utils.default import default
import language_model_v2.llama3_format as llama3format

from dataclasses import dataclass as settings; vars().update(settings=settings)


@settings
class LlamaHypers:
    settings = {}
    """Hyperparameters that were passed to the constructor. Used internally to override loaded hyperparameters."""
    base: str = "meta-llama/Meta-Llama-3.1-{param_magnitude}-Instruct"
    """The base model repository ID. If no model_to_load is specified, the base will be loaded from this ID."""
    model_to_load: str = None
    """Path to a custom model to load. The base will be used instead if this is None."""
    tokenizer_repo_id: str = None
    """The repository ID of the tokenizer to use. If None, the tokenizer will be loaded from the base ID."""
    param_magnitude: str|None = '8B'
    """The magnitude of the model parameters, replacing {param_magnitude} in the base."""
    checkpoint_to_load: str|None = None
    """Path to a specific checkpoint to load. This overrides base and model_to_load to resume training from a specific checkpoint, including the optimizer state."""
    quantization: str|None = 'int8'
    """Quantization mode to use. If None, no quantization will be used (16 bit half-precision)."""
    lora: int|None = 8
    """The LoRA rank to use. If None, no LoRA adapter be used at all."""
    lora_modules: tuple[str] = ('embed_tokens', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head')
    """The modules to apply LoRA adapters to. Modules must be names from model architecture and can be found using print(model.model)"""
    lora_alpha: float|None = None
    """The alpha value to use for LoRA. If None, the alpha will be set to 2*lora (recommended)."""
    lora_dropout: float|None = None
    """The dropout rate to use for LoRA (or None)."""
    lora_merge_on_load: bool = False
    """Whether to merge the LoRA adapter into the original model weights on load."""
    gradient_checkpointing: bool = True
    """Whether to use gradient checkpointing to reduce memory usage."""
    epochs: int|float = 1
    """The number of epochs to train for. If a float, the model will train for that fraction of the dataset."""
    train_batch_size: int = 1
    """The effective batch size to use for training. Divide by gradient_accumulation_steps to get the actual batch size used on the hardware. The actual_train_batch_size property can be used to get the real batch size as well."""
    gradient_accumulation_steps: int = 1
    """..."""
    save_checkpoint_every_x_epochs: float = 1.0
    clean_up_checkpoint_after_training: bool = True
    learning_rate: float = 2e-4
    optimizer: str = 'adafactor'
    warmup_steps: int = 0
    clip_gradient_norm: float = 1.0
    weight_decay: float = 0.0
    scheduler_type: str = 'constant'
    train_on_s2s_inputs: bool = False
    format: str = ''''''
    max_sequence_length: int = 1024
    protected_input_length: int = 512
    max_output_length: int = 512
    repetition_penalty: float = 1.2
    num_beams: int = 1
    temperature: float = 0.6
    sampled_generation: bool = False
    top_p: float = 0.9
    top_k: int = 50
    gen_batch_size: int = 1
    ppl_batch_size: int = 1
    dynamic_tokenization: bool = True

    def actual_train_batch_size(self):
        return self.train_batch_size // self.gradient_accumulation_steps

    def __post_init__(self):
        if str(self.param_magnitude):
            self.base = self.base.format(param_magnitude=str(self.param_magnitude))
        if self.model_to_load is None:
            self.model_to_load = self.base


class Llama(LlamaHypers):
    def __post_init__(self):
        LlamaHypers.__post_init__(self)
        assert self.protected_input_length < self.max_sequence_length, \
            f"Protected input length {self.protected_input_length} must not exceed max sequence length {self.max_sequence_length}"
        if pl.Path(self.base).exists() and (pl.Path(self.base) / 'hyperparameters.json').exists():
            loaded_hyperparams: dict = json.loads((pl.Path(self.base) / 'hyperparameters.json').read_text())
            specified_hyperparameters = vars(self).pop('settings')
            hyperparameters = {**loaded_hyperparams, **specified_hyperparameters}
            vars(self).update(hyperparameters)
        self.hyperparameters: dict = dict(vars(self))
        tokenizer_reponame = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = tf.AutoTokenizer.from_pretrained(tokenizer_reponame, trust_remote_code=True)

import time
import os
device = 'cuda'
if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
initialization_config = dict(
    # quantization_config = tf.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_compute_dtype=pt.bfloat16,
    #     bnb_4bit_use_double_quant=False
    # ),
    torch_dtype=pt.bfloat16,
    device_map='auto'
)
model = tf.AutoModelForCausalLM.from_pretrained(model_id, **initialization_config)
# print(model)
# model = pt.compile(model)
tokenizer: tf.LlamaTokenizer = tf.AutoTokenizer.from_pretrained(model_id)

def get_current_temperature(location: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

# messages = [
#     {"role": "system", "content": "You are a bot that responds to weather queries."},
#     {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
# ]
#
# prompt = tokenizer.apply_chat_template(messages, tools=[dict(
#     type="function", name="get_current_temperature", description="Get the current temperature (Celsius) at a location.",
#     parameters=dict(type="object", properties=dict(location=dict(type="string", description="The location to get the temperature for, in the format 'City, Country'")), required=["location"]),
# )], add_generation_prompt=True, return_dict=True, return_tensors='pt')

input = '''
A: i am looking for a museum in the centre of town .
B: i will recommend primavera museum . their entrance fee is free and you can reach them on 01223357708
A: what is their address ?
B: the address is 10 king s parade . do you need directions ?
A: just the postcode would help and i also will need a place to stay .
B: the postcode is cb21s . where in town were you looking to stay ?
A: i would like to stay in the centre are there any expensive hotel -s ?
B: university arms hotel , 4 star rating located on regent street . phone number is 01223351241 . would you like a reservation ?
A: no thank you i just wanted to get that information .

Identify the information from the above dialogue:
hotel stars: The rating or number of stars of the hotel [0, 1, 2, 3, 4, 5, any]?'''

prompt = tokenizer.apply_chat_template([
    dict(role="system", content="You are a helpful assistant."),
    dict(role="user", content=input),
], return_dict=True, return_tensors='pt')
prompt['input_ids'].to(device)
prompt['attention_mask'].to(device)
print(f"Max GPU utilization: {pt.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")
# pp([f"{token.item()}: {tokenizer.decode(token.item())}" for token in prompt['input_ids'].squeeze()])
# pp(f"<|end_of_text|>: {tokenizer.encode('<|end_of_text|>', add_special_tokens=False)}")
t1 = time.perf_counter()
output, = model.generate(**prompt, max_length=1000, max_new_tokens=300, num_return_sequences=1)
t2 = time.perf_counter()
print(f"Generation took {t2 - t1:.3f} seconds")
response = tokenizer.decode(output)
print(response)

