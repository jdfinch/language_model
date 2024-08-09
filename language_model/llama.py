import dataclasses

import peft
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    get_constant_schedule_with_warmup as warmup_scheduler,
    Adafactor,
    AdamW
)
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig

from accelerate import Accelerator
from tqdm import tqdm
import ezpyzy as ez
from dataclasses import dataclass as settings; vars().update(settings=ez.settings)
import pathlib as pl
import shutil
import os
import math


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


loss_mask = -100


def load_merge_and_save_lora(lora_path: ez.filelike, merged_path: ez.filelike=None, dtype='bf16'):
    dtypes = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    dtype = dtypes[dtype]
    lora_path = ez.File(lora_path).path
    print(lora_path)
    name = lora_path.name
    adapter_config = ez.File(lora_path / 'adapter_config.json').load()
    base_model_name = adapter_config['base_model_name_or_path']
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=dtype, device_map='auto'
    )
    model = peft.PeftModel.from_pretrained(base_model, lora_path, device_map='auto')
    merged = model.merge_and_unload()
    if merged_path is None:
        merged_path = lora_path.parent / f"{name}.MERGED"
    merged.save_pretrained(merged_path, safe_serialization=False, save_peft_format=False)
    return merged_path

def get_merged_model(model_path, dtype='bf16'):
    model_path = pl.Path(model_path)
    if model_path.suffix == '.MERGED':
        model_path = model_path.parent / model_path.name.split('.')[0]
    if model_path.exists() and not (model_path/'adapter_config.json').exists():
        return model_path
    elif model_path.exists() and (model_path/'adapter_config.json').exists():
        merged_model_path = model_path.parent / f"{model_path.name}.MERGED"
        if merged_model_path.exists():
            return merged_model_path
        else:
            adapter_config = ez.File(model_path/'adapter_config.json').load()
            adapter_base_path = pl.Path(adapter_config['base_model_name_or_path'])
            merged_adapter_base_path = get_merged_model(adapter_base_path)
            if merged_adapter_base_path != adapter_base_path:
                adapter_config['base_model_name_or_path'] = str(merged_adapter_base_path)
                ez.File(model_path/'adapter_config.json').save(adapter_config)
            merged_model_path = model_path.parent / f"{model_path.name}.MERGED"
            ez.subproc(load_merge_and_save_lora, model_path, merged_model_path, dtype)
            return merged_model_path
    else:
        return model_path



@settings
class LlamaHyperparameters(ez.Settings):
    base: ez.ColStr = ez.Def("meta-llama/Llama-2-{param_magnitude}-chat-hf")
    param_magnitude: ez.ColStr = ez.Def('7b')
    format: ez.ColStr = ez.Def('''[INST] <<SYS>> You are a helpful, respectful, and honest assistant. <</SYS>> {input} [/INST]''')
    train_on_s2s_inputs: ez.ColBool = ez.Def(False)
    quantize: ez.ColStr = ez.Def('nf4')
    checkpoint: ez.ColStr = None
    checkpoint_after_every_x_epochs: ez.ColFloat = ez.Def(1.0)
    checkpoint_clean_up_after_train: ez.ColBool = ez.Def(True)
    epoch: ez.ColInt = ez.Def(0)
    step: ez.ColInt = ez.Def(0)
    epochs: ez.ColInt = ez.Def(1)
    max_sequence_length: ez.ColInt = ez.Def(4096)
    protected_input_length: ez.ColInt = ez.Def(512)
    train_batch_size: ez.ColInt = ez.Def(1)
    gradient_accumulation_steps: ez.ColInt = ez.Def(1)
    optimizer: ez.ColStr = ez.Def('adamw_bnb_8bit')
    learning_rate: ez.ColFloat = ez.Def(2e-4)
    weight_decay: ez.ColFloat = ez.Def(0.001)
    max_gradient_norm: ez.ColFloat = ez.Def(0.3)
    warmup_steps: ez.ColInt = ez.Def(0)
    lr_scheduler_type: ez.ColStr = ez.Def('constant')
    lora: ez.ColInt = ez.Def(8)
    lora_alpha: ez.ColInt = None
    lora_dropout: ez.ColFloat = ez.Def(0.0)
    lora_modules: ez.Column[list[str]]|list[str]|None = None
    lora_merge_on_load: ez.ColBool = ez.Def(True)
    gradient_checkpointing: ez.ColBool = ez.Def(True)
    max_output_length: ez.ColInt = ez.Def(512)
    repetition_penalty: ez.ColFloat = ez.Def(1.2)
    num_beams: ez.ColInt = ez.Def(1)
    temperature: ez.ColFloat = ez.Def(0.6)
    sampled_generation: ez.ColBool = ez.Def(False)
    top_p: ez.ColFloat = ez.Def(0.9)
    top_k: ez.ColInt = ez.Def(50)
    gen_batch_size: ez.ColInt = ez.Def(1)
    ppl_batch_size: ez.ColInt = ez.Def(1)
    dynamic_tokenization: ez.ColBool = ez.Def(True)
    tokenizer_reponame: ez.ColStr = ez.Def("meta-llama/Llama-2-7b-chat-hf")

    def actual_train_batch_size(self):
        return self.train_batch_size // self.gradient_accumulation_steps

    def __post_init__(self):
        if '{param_magnitude}' in self.base:
            self.base = self.base.replace('{param_magnitude}', str(self.param_magnitude))


class Llama(LlamaHyperparameters):
    def __post_init__(self):
        LlamaHyperparameters.__post_init__(self)
        assert self.protected_input_length < self.max_sequence_length, \
            f"Protected input length {self.protected_input_length} must not exceed max sequence length {self.max_sequence_length}"
        if pl.Path(self.base).exists() and (pl.Path(self.base)/'hyperparameters.json').exists():
            loaded_hyperparams:dict = ez.File(pl.Path(self.base)/'hyperparameters.json').load()
            specified_hyperparameters = vars(self).pop('settings')
            hyperparameters = {**loaded_hyperparams, **specified_hyperparameters}
            vars(self).update(hyperparameters)
        self.hyperparameters: dict = dict(vars(self))
        tokenizer_reponame = self.tokenizer_reponame
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_reponame, trust_remote_code=True, padding_side='left')
        self.tokenizer.padding_side = 'left'
        self.tokenizer.return_special_tokens_mask = True
        if 'Llama-2' in tokenizer_reponame:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        quant_kwargs = {}
        if self.quantize is not None:
            if self.quantize == 'nf4':
                quant_kwargs = dict(
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False
                    ),
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
            elif self.quantize == 'int8':
                quant_kwargs = dict(
                    load_in_8bit=True,
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
            elif self.quantize == 'bf16':
                quant_kwargs = dict(
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
            else:
                raise ValueError(f"Invalid quantization level: {self.quantize}.\n"
                                 f"Supported quantizations are: 'nf4', 'int8', 'bf16', None")
        load_path = pl.Path(self.base)
        delete_merge_path = None
        if load_path.exists() and (load_path/'adapter_config.json').exists():
            adapter_config = ez.File(load_path/'adapter_config.json').load()
            adapter_base_path = pl.Path(adapter_config['base_model_name_or_path'])
            if not adapter_base_path.exists() and adapter_base_path.suffix == '.MERGED':
                merged_adapter_base_path = get_merged_model(adapter_base_path, 'bf16')
                if adapter_base_path != merged_adapter_base_path:
                    adapter_config['base_model_name_or_path'] = str(merged_adapter_base_path)
                    ez.File(load_path/'adapter_config.json').save(adapter_config)
            if self.lora_merge_on_load:
                merged_path = load_path.parent / f"{load_path.name}.MERGED"
                if not merged_path.exists():
                    merged_path = load_path.parent / f"{load_path.name}.{ez.uuid()}.MERGED"
                    delete_merge_path = merged_path
                ez.subproc(load_merge_and_save_lora, load_path, merged_path, 'bf16')
                load_path = merged_path
        elif load_path.suffix == '.MERGED':
            load_path = get_merged_model(load_path, 'bf16')
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            **quant_kwargs,
        )
        if delete_merge_path is not None:
            shutil.rmtree(delete_merge_path, ignore_errors=True)
        if 'Llama-2' in tokenizer_reponame:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.use_cache = False
        self.acclerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        assert self.train_batch_size % self.gradient_accumulation_steps == 0
        if self.gen_batch_size is None:
            self.gen_batch_size = self.actual_train_batch_size()
        if self.lora is not None and self.lora_alpha is None:
            self.lora_alpha = self.lora * 2  # heuristic usually works well
        if self.lora_modules is None:
            self.lora_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
            ]
        self.model.eval()

    def save(self, path:ez.filelike):
        path = ez.File(path).path
        self.model.save_pretrained(path)
        hyperparameters = {k:v for k,v in self.hyperparameters.items() if k != 'settings'}
        ez.File(path/'hyperparameters.json').save(hyperparameters)
        return path

    def save_checkpoint(self, path: ez.filelike = None):
        if path is None:
            path = 'ex/scratch/checkpoint'
        path = ez.File(path).path
        self.acclerator.save_state(path)
        return path

    def preprocess(self, inputs=None, outputs=None):
        if inputs and not isinstance(next(iter(inputs)), str):
            inputs, outputs = zip(*inputs)
        elif outputs is None:
            outputs = [None] * len(inputs)
        elif inputs is None:
            inputs = [None] * len(outputs)
        if 'Llama-3' in self.tokenizer_reponame:
            outputs = list(outputs)
            for i, output in enumerate(outputs):
                if output is not None:
                    outputs[i] = f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}"
        data = zip(inputs, outputs)
        input_pre_format, input_post_format = self.format.split('{input}')
        pre_format_tokens = self.tokenizer(input_pre_format.rstrip(), padding=False)['input_ids']
        post_format_tokens = self.tokenizer(
            input_post_format.lstrip(), add_special_tokens=False, padding=False
        )['input_ids']
        num_format_tokens = len(pre_format_tokens) + len(post_format_tokens)
        assert num_format_tokens <= self.max_sequence_length - self.protected_input_length, \
            f"Format tokens {num_format_tokens} exceed difference between max_sequence_length and protected_input_length, increase the difference or shorten the format by {self.max_sequence_length - self.protected_input_length}"
        datalist = []
        for input, output in data:
            input_tokens = self.tokenizer(
                input, add_special_tokens=False, padding=False
            )['input_ids'] if input else []
            output_tokens = self.tokenizer(
                output, add_special_tokens=False, padding=False
            )['input_ids'] if output else []
            out = int(output is not None)
            overflow = num_format_tokens + len(input_tokens) + len(output_tokens) + out - self.max_sequence_length
            if overflow > 0:
                input_overflow = min(overflow, max(0, len(input_tokens) - self.protected_input_length))
                input_tokens = input_tokens[input_overflow:]
            overflow = num_format_tokens + len(input_tokens) + len(output_tokens) + out - self.max_sequence_length
            if overflow > 0:
                output_overflow = min(overflow, len(output_tokens))
                output_tokens = output_tokens[:-output_overflow]
            if self.train_on_s2s_inputs:
                labels = (
                    pre_format_tokens + input_tokens + post_format_tokens + output_tokens +
                    ([self.tokenizer.eos_token_id] if out else [])
                )
            else:
                labels = (
                    [loss_mask] * len(pre_format_tokens + input_tokens + post_format_tokens) + output_tokens +
                    ([self.tokenizer.eos_token_id] if out else [])
                )
            all_tokens = (
                pre_format_tokens + input_tokens + post_format_tokens + output_tokens +
                ([self.tokenizer.eos_token_id] if out else [])
            )
            attention_mask = [1] * len(all_tokens)
            assert len(all_tokens) <= self.max_sequence_length, \
                f"Token length {len(all_tokens)} exceeds max sequence length {self.max_sequence_length}"
            datalist.append(dict(input_ids=all_tokens, attention_mask=attention_mask, labels=labels))
        return datalist

    def _set_up_dataloader(self, inputs, outputs, batch_size=1, shuffle=False):
        if not self.dynamic_tokenization:
            datalist = self.preprocess(inputs, outputs)
            dataset = Dataset.from_list(datalist)
            collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                return_tensors='pt',
                pad_to_multiple_of=8,
                padding=True
            )
        else:
            if inputs and not isinstance(next(iter(inputs)), str):
                inputs, outputs = zip(*inputs)
            elif outputs is None:
                outputs = [None] * len(inputs)
            elif inputs is None:
                inputs = [None] * len(outputs)
            datalist = [{'input': input, 'output': output} for input, output in zip(inputs, outputs)]
            dataset = Dataset.from_list(datalist)
            collator = DataCollatorWithPreprocessing(
                tokenizer=self.tokenizer,
                model=self,
                return_tensors='pt'
            )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, shuffle=shuffle
        )
        return dataloader

    def training(self, inputs=None, outputs=None, yield_every_x_epochs=1):
        if self.lora is not None:
            if self.lora_merge_on_load: # if lora is merged, we can set up another lora for subsequent training; however, if lora is not merged, we want to further tune the original
                peft_parameters = LoraConfig(
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    r=self.lora,
                    target_modules=self.lora_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                    inference_mode=False
                )
                self.model = peft.get_peft_model(self.model, peft_parameters) # noqa
                self.model.base_model.model.enable_input_require_grads()
        self.model.train()
        dataloader = self._set_up_dataloader(
            inputs, outputs, self.actual_train_batch_size(), shuffle=True
        )
        if self.optimizer == 'adamw_bnb_8bit':
            optimizer = AdamW8bit(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'adafactor':
            optimizer = Adafactor(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, relative_step=False
            )
        elif self.optimizer == 'adam' or self.optimizer == 'adamw':
            optimizer = AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(
                f"Invalid optimizer: {self.optimizer}."
                f" Supported optimizers are: 'adam', 'adamw_bnb_8bit', 'adafactor'"
            )
        scheduler = warmup_scheduler(optimizer, num_warmup_steps=self.warmup_steps)
        dataloader, model, optimizer, scheduler = self.acclerator.prepare(
            dataloader, self.model, optimizer, scheduler  # noqa
        )
        if self.checkpoint:
            self.acclerator.load_state(self.checkpoint)
            self.checkpoint = None
        display = tqdm(total=self.epochs * len(inputs), desc='Training', position=0, leave=True)
        epoch_checkpoint_progress = 0
        epoch_yield_progress = 0
        num_per_yield = int(yield_every_x_epochs * len(inputs))
        num_per_checkpoint = int(self.checkpoint_after_every_x_epochs * len(inputs))
        for self.epoch in range(self.epochs):
            nlls = []
            for self.step, batch in enumerate(dataloader):
                epoch_checkpoint_progress += len(batch['input_ids'])
                epoch_yield_progress += len(batch['input_ids'])
                with self.acclerator.accumulate(model):
                    loss = model(**batch).loss
                    if not math.isnan(loss.item()):
                        self.acclerator.backward(loss)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        num_tokens = batch['labels'].ne(loss_mask).sum().item()
                        token_loss = loss.item() * num_tokens
                        nlls.append((token_loss, num_tokens))
                    else:
                        print('Warning: NaN loss encountered')
                display.update(len(batch['input_ids']))
                if (
                    self.checkpoint_after_every_x_epochs and
                    epoch_checkpoint_progress // num_per_checkpoint
                ):
                    self.save_checkpoint()
                    epoch_checkpoint_progress = epoch_checkpoint_progress % num_per_checkpoint
                if epoch_yield_progress // num_per_yield:
                    total_nll = sum(nll for nll, _ in nlls)
                    total_tokens = sum(num_tokens for _, num_tokens in nlls)
                    perplexity = math.exp(total_nll / total_tokens)
                    nlls = []
                    epoch_yield_progress = epoch_yield_progress % num_per_yield
                    yield perplexity
            if epoch_yield_progress and nlls:
                total_nll = sum(nll for nll, _ in nlls)
                total_tokens = sum(num_tokens for _, num_tokens in nlls)
                perplexity = math.exp(total_nll / total_tokens)
                yield perplexity
        if self.checkpoint_clean_up_after_train:
            shutil.rmtree('ex/scratch/checkpoint', ignore_errors=True)

    def train(self, inputs=None, outputs=None):
        return list(self.training(inputs=inputs, outputs=outputs))

    def perplexity(self, inputs, outputs=None):
        num_items = len(inputs) if outputs is None else len(outputs)
        dataloader = self._set_up_dataloader(inputs, outputs, self.ppl_batch_size, shuffle=False)
        dataloader, model = self.acclerator.prepare(dataloader, self.model)  # noqa
        display = tqdm(total=num_items, desc='Calculating Perplexity', position=0)
        nlls = []
        for step, batch in enumerate(dataloader):
            loss = model(**batch).loss
            if not math.isnan(loss.item()):
                num_tokens = batch['labels'].ne(loss_mask).sum().item()
                token_loss = loss.item() * num_tokens
                nlls.append((token_loss, num_tokens))
            else:
                print('Warning: NaN loss encountered')
            display.update(len(batch.data['input_ids']))
        total_nll = sum(nll for nll, _ in nlls)
        total_tokens = sum(num_tokens for _, num_tokens in nlls)
        ppl = math.exp(total_nll / total_tokens)
        self.model.eval()
        return ppl

    def generate(self, prompt):
        single = False
        if isinstance(prompt, str):
            prompt = [prompt]
            single = True
        with ez.shush():
            dataloader = self._set_up_dataloader(prompt, None, batch_size=self.gen_batch_size)
            config = GenerationConfig(
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_output_length,
                temperature=self.temperature,
                num_beams=self.num_beams,
                do_sample=self.sampled_generation,
                top_p=self.top_p,
                top_k=self.top_k,
                eos_token_id=2 if 'Llama-2' in self.base else self.tokenizer.eos_token_id,
            )
        encoded_gens = []
        input_lens = []
        display = tqdm(
            total=len(prompt), desc='Generating', position=0,
            disable=self.gen_batch_size >= len(prompt)
        )
        for step, batch in enumerate(dataloader):
            input_lens.extend(len(x) for x in batch.data['input_ids'])
            with ez.shush():
                batch_dict = {k: v.to('cuda') for k, v in batch.items()}
                gen = self.model.generate(
                    **batch_dict,
                    generation_config=config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            encoded_gens.extend(gen)
            display.update(len(batch.data['input_ids']))
        with ez.shush():
            decoded_gens = []
            for gen, input_len in zip(encoded_gens, input_lens):
                if 'Llama-2' in self.tokenizer_reponame:
                    generated = self.tokenizer.decode(gen[input_len:], skip_special_tokens=True)
                else:
                    generated = self.tokenizer.decode(gen[input_len:], skip_special_tokens=False)
                    generated = generated[
                        max(generated.rfind('<|end_header_id|>') + len('<|end_header_id|>'), 0):
                        generated.find('<|eot_id|>')
                    ]
                decoded_gens.append(generated)
        return decoded_gens[0] if single else decoded_gens


class DataCollatorWithPreprocessing:

    def __init__(self, tokenizer, model, return_tensors):
        self.seq2seq_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            pad_to_multiple_of=8,
            padding=True
        )
        self.model = model

    def __call__(self, datapoints):
        if isinstance(datapoints, dict):
            inputs = [datapoints['input']]
            outputs = [datapoints['output']]
        else:
            inputs = [d['input'] for d in datapoints]
            outputs = [d['output'] for d in datapoints]
        if all(output is None for output in outputs):
            outputs = None
        if all(input is None for input in inputs):
            inputs = None
        preprocessed = self.model.preprocess(inputs, outputs)
        return self.seq2seq_collator(preprocessed)


llama3format = '''<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{input}<|eot_id|>'''

llama3id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

llama3config = """
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128008,
    128009
  ],
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": { "factor": 8.0, "type": "dynamic" },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.3",
  "use_cache": true,
  "vocab_size": 128256
}
"""

def main():
    llama3 = Llama(
        base="meta-llama/Meta-Llama-3.1-8B-Instruct",
        param_magnitude='8B',
        format=llama3format,
        temperature=0.6,
        sampled_generation=True,
        repetition_penalty=1.0,
        tokenizer_reponame="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    response = llama3.generate(['''A: i am looking for a museum in the centre of town.
B: i will recommend primavera museum. their entrance fee is free and you can reach them on 01223357708
A: what is their address?
B: the address is 10 king s parade. do you need directions?
A: just the postcode would help and i also will need a place to stay.
B: the postcode is cb21s. where in town were you looking to stay?
A: i would like to stay in the centre are there any expensive hotel -s?
B: university arms hotel, 4 star rating located on regent street. phone number is 01223351241. would you like a reservation?
A: no thank you i just wanted to get that information.

Identify the information from the above dialogue:
hotel stars: The rating or number of stars of the hotel [0, 1, 2, 3, 4, 5, any]?''', "What is the capital of France?"])
    print('Response:', response)

if __name__ == '__main__':
    main()

