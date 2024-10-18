import torch.cuda
import transformers as hf
import torch as pt
import dataclasses as dc
import textwrap as tw
import pathlib as pl
import itertools as it
import json
import copy as cp
import random as rng

import ezpyzy as ez

import language_model.tokenizer as tok
import language_model.lm_config as lm_config

import typing as T


# black magic type hinting: sneak the "base" decorator into "dataclass" var name
from dataclasses import dataclass; vars().update(dataclass=ez.config)


""" ------ Format templates for Llama models ------"""

llama3format = {
    'system (trunc_content=False, trunc_segment=False)':
        '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n#[input=text]#<|eot_id|>''',
    'user (trunc_content=False, trunc_segment=True)':
        '''<|start_header_id|>user<|end_header_id|>\n\n#[input=text]#<|eot_id|>''',
    'assistant (trunc_content=False, trunc_segment=True)':
        '''<|start_header_id|>assistant<|end_header_id|>\n\n#[output=text]#''',
    'assistant_history (trunc_content=False, trunc_segment=True)':
        '''<|start_header_id|>assistant<|end_header_id|>\n\n#[input=text]#<|eot_id|>''',
    'asssistant_completion (trunc_content=False, trunc_segment=False)':
        '''<|start_header_id|>assistant<|end_header_id|>\n\n#[input=prefix]##[output=completion]#''',
    'info (trunc_segment=False, trunc_content=True)':
        '''<|start_header_id|>user<|end_header_id|>\n\n#[input=text]#<|eot_id|>'''}

llama2format = ...


""" ------ Overriding new defaults for Llama models ------"""

@dataclass
class LlamaLoRA(lm_config.LoRA):
    modules: tuple[str] = (
        'embed_tokens', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head')

@dataclass
class LlamaConfig(lm_config.LMConfig):
    model_base: str = 'meta-llama/Llama-3.2-3B-Instruct'
    adapters: tuple[lm_config.LoRA] = ez.default((LlamaLoRA(),))

    def __post_init__(self):
        super().__post_init__()
        self.sequence_params.tokenizer = self.model_base
        if self.sequence_params.format is None:
            if 'Llama-3' in self.model_base:
                self.sequence_params.format = llama3format
            else:
                raise ValueError(f"Could not infer tokenizer from model base: {self.model_base}")


""" ------ Llama model implementation ------"""

@dataclass
class Llama(LlamaConfig):
    """A Llama model configured for training and/or generation."""

    def __post_init__(self):
        # if base is a Llama object, avoid reloading the model if it matches
        if isinstance(self.base, Llama) and self.base.model_base == self.model_base:
            ...
        # LMConfig post init responsible for loading config from file (if applicable)
        super().__post_init__()
        # Set up the quantization base
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
            tokenizer=self.sequence_params.tokenizer,
            local_files_only=self.load_locally_saved_models_only)
        self.template: tok.TokenTemplateCollection = self.tokenizer.templatize(
            self.sequence_params.format,
            max_length=self.sequence_params.max_length,
            pad_to_same_length=True,
            pad_to_multiple_of=self.sequence_params.pad_to_multiple_of,
            pad_side=self.sequence_params.pad_side,
            trunc_segments_side=self.sequence_params.trunc_segments_side,
            max_segments=self.sequence_params.max_segments)
        self.model: hf.LlamaForCausalLM = hf.AutoModelForCausalLM.from_pretrained(
            self.model_to_load,
            **quantization_kwargs,
            device_map=self.hardware_device,
            local_files_only=self.load_locally_saved_models_only)


    def save(self, path: str|pl.Path=None, save_as_checkpoint=False):
        return super().save(path)

    def preprocess(self,
        inputs: T.Iterable[T.Iterable[dict[str, str]]],
        batch_size: int = None
    ) -> list[tok.TokenSequenceBatch]:
        batches = ez.batched(inputs, size=batch_size)
        preprocessed = [self.template.fill(batch) for batch in batches]
        return preprocessed

    def generate(self,
        sequences: T.Iterable[T.Iterable[dict[str, str]]]
    ) -> list[str]:
        batches = self.preprocess(sequences, self.generate_params.batch_size)
        generation_config = self.generate_params.construct_hf_config()
        generated_results = []
        for batch in batches:
            tokens = batch.dict(seq_type=ez.bind(pt.tensor)(dtype=pt.long, device=self.hardware_device))
            outputs = self.model.generate(**tokens, generation_config=generation_config)
            generated_tokens = [output[len(input):] for input, output in zip(batch, outputs)]
            generated_texts = [self.tokenizer.decode(tokens) for tokens in generated_tokens]
            generated_results.extend(generated_texts)
        return generated_results

    def training(self, data: T.Iterable[T.Iterable[dict[str, str]]]):
        optimizer = ...
        scheduler = ...
        data = list(data)
        if self.train_params.shuffle_data and isinstance(self.train_params.shuffle_data, int):
            shuffle = rng.Random(self.train_params.shuffle_data).shuffle
        elif self.train_params.shuffle_data:
            shuffle = rng.Random().shuffle
        else:
            shuffle = lambda x: x
        for epoch in range(self.train_params.epochs):
            shuffle(data)
            batches = self.preprocess(data, self.train_params.physical_batch_size)
            def training_steps_over_one_epoch(batches):
                nlls, token_counts = [], []
                i = 0
                for i, batch in enumerate(batches):
                    tokens = batch.dict(seq_type=ez.bind(pt.tensor)(dtype=pt.long, device=self.hardware_device))
                    outputs = self.model(**tokens)
                    num_output_tokens = batch['labels'].ne(-100).sum().item()
                    nlls.append(outputs.loss.item() * num_output_tokens)
                    token_counts.append(num_output_tokens)
                    loss = outputs.loss / self.train_params.gradient_accumulation_steps
                    loss.backward()
                    if i % self.train_params.gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        ppl = sum(nlls) / sum(token_counts)
                        nlls, token_counts = [], []
                        yield ppl
                if i % self.train_params.gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    ppl = sum(nlls) / sum(token_counts)
                    yield ppl
            yield training_steps_over_one_epoch(batches)


if __name__ == '__main__':
    llama = Llama(model_base='meta-llama/Llama-3.2-1B-Instruct', hardware_device='cuda', quantization='bf16')

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
        dict(temp='user', text='Alright...'),
        dict(temp='assistant', text=None),
    ]] * (2 // 2)

    with ez.Timer('preprocess'):
        preprocessed = llama.preprocess(dialogues, batch_size=4)
    print(preprocessed[0].display())
    generated = llama.generate(dialogues)
    print('\n\n'.join(generated))

    print(llama.json())





























