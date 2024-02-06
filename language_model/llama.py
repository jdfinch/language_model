import dataclasses

import peft
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    get_constant_schedule_with_warmup as warmup_scheduler
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


def load_merge_and_save_lora(lora_path: ez.filelike, merged_path: ez.filelike=None):
    lora_path = ez.File(lora_path).path
    print(lora_path)
    name = lora_path.name
    adapter_config = ez.File(lora_path / 'adapter_config.json').load()
    base_model_name = adapter_config['base_model_name_or_path']
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model = peft.PeftModel.from_pretrained(base_model, lora_path)
    merged = model.merge_and_unload()
    if merged_path is None:
        merged_path = lora_path.parent / f"{name}.MERGED"
    merged.save_pretrained(merged_path, safe_serialization=False, save_peft_format=False)
    return merged_path

@settings
class LlamaHyperparameters(ez.Settings):
    base: ez.ColStr = ez.Def("meta-llama/Llama-2-{param_magnitude}-chat-hf")
    param_magnitude: ez.ColStr = ez.Def('7b')
    format: ez.ColStr = ez.Def('''[INST] <<SYS>> You are a helpful, respectful, and honest assistant. <</SYS>> {input} [/INST] {output} </s>''')
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
    lora_dropout: ez.ColFloat = ez.Def(0.1)
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

    def actual_train_batch_size(self):
        return self.train_batch_size // self.gradient_accumulation_steps

    def __post_init__(self):
        if '{param_magnitude}' in self.base:
            self.base = self.base.replace('{param_magnitude}', str(self.param_magnitude))


class Llama(LlamaHyperparameters):

    def __post_init__(self):
        LlamaHyperparameters.__post_init__(self)
        if pl.Path(self.base).exists() and (pl.Path(self.base)/'hyperparameters.json').exists():
            loaded_hyperparams:dict = ez.File(pl.Path(self.base)/'hyperparameters.json').load()
            specified_hyperparameters = vars(self).pop('settings')
            hyperparameters = {**loaded_hyperparams, **specified_hyperparameters}
            vars(self).update(hyperparameters)
        self.hyperparameters: dict = dict(vars(self))
        tokenizer_reponame = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_reponame, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.return_special_tokens_mask = True
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        quant_kwargs = {}
        if self.quantize is not None:
            if self.quantize == 'nf4':
                quant_kwargs = dict(
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=False
                    ),
                    device_map='auto'
                )
            elif self.quantize == 'int8':
                quant_kwargs = dict(
                    load_in_8bit=True,
                    device_map='auto'
                )
            elif self.quantize == 'fp16':
                quant_kwargs = dict(
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
            else:
                raise ValueError(f"Invalid quantization level: {self.quantize}.\n"
                                 f"Supported quantizations are: 'nf4', 'int8', 'fp16', None")
        load_path = pl.Path(self.base)
        delete_merge_path = None
        if load_path.exists() and (load_path/'adapter_config.json').exists() and self.lora_merge_on_load:
            merged_path = load_path.parent / f"{load_path.name}.MERGED"
            delete_after = not merged_path.exists()
            ez.subproc(load_merge_and_save_lora, load_path)
            if delete_after:
                delete_merge_path = merged_path
            load_path = merged_path
        else:
            load_path = self.base
        self.model = AutoModelForCausalLM.from_pretrained(load_path, **quant_kwargs)
        if delete_merge_path is not None:
            shutil.rmtree(delete_merge_path, ignore_errors=True)
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
        ez.File(path/'hyperparameters.json').save(self.hyperparameters)
        return path

    def save_checkpoint(self, path: ez.filelike = None):
        if path is None:
            path = 'ex/scratch/checkpoint'
        path = ez.File(path).path
        self.acclerator.save_state(path)
        return path

    def preprocess(self, inputs=None, outputs=None):
        if outputs is None:
            if inputs and not isinstance(next(iter(inputs)), str):
                inputs, outputs = zip(*inputs)
        if inputs is not None and outputs is not None:
            data = zip(inputs, outputs)
            splitter = self.format.find('{output}')
            input_format = self.format[:splitter].strip()
            output_format = self.format[splitter:].strip()
            datalist = []
            for input, output in data:
                input_tokens = self.tokenizer(input_format.format(input=input))
                output_tokens = self.tokenizer(output_format.format(output=output), add_special_tokens=False)
                input_ids = input_tokens['input_ids']
                output_ids = output_tokens['input_ids']
                overflow = len(input_ids) + len(output_ids) - self.max_sequence_length
                if overflow > 0:
                    input_overflow = min(overflow, max(0, len(input_ids) - self.protected_input_length))
                    input_ids = input_ids[input_overflow:]
                    output_ids = output_ids[:max(0, self.max_sequence_length - len(input_ids))]
                if self.train_on_s2s_inputs:
                    labels = input_ids + output_ids
                else:
                    labels = [loss_mask]*len(input_ids) + output_ids
                datalist.append(dict(
                    input_ids=input_ids + output_ids,
                    attention_mask=[1] * len(input_ids) + [1] * len(output_ids),
                    labels=labels
                ))
        elif outputs is not None:
            splitter = self.format.find('{input}')
            format = self.format.replace('{input}', '')
            if format[splitter-1:splitter+1] == '  ':
                format = format[:splitter-1] + format[splitter:]
            datalist = [
                self.tokenizer(format.format(output=input), text_target=input)
                for input in inputs
            ]
        elif inputs is not None:
            splitter = self.format.find('{output}')
            format = self.format[:splitter].rstrip()
            datalist = [
                self.tokenizer(format.format(input=input))
                for input in inputs
            ]
        else:
            raise ValueError("Must provide either inputs or outputs")
        if self.max_sequence_length is not None:
            for case in datalist:
                truncation = max(0, len(case['input_ids']) - self.max_sequence_length)
                case['input_ids'] = case['input_ids'][truncation:]
                case['attention_mask'] = case['attention_mask'][truncation:]
                if 'labels' in case:
                    case['labels'] = case['labels'][truncation:]
        return datalist

    def _set_up_dataloader(self, inputs, outputs, batch_size=1, shuffle=False):
        if not self.dynamic_tokenization:
            datalist = self.preprocess(inputs, outputs)
            dataset = Dataset.from_list(datalist)
            collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                return_tensors='pt',
                pad_to_multiple_of=8,
            )
        else:
            if outputs is None:
                if isinstance(next(iter(inputs)), str):
                    outputs = [None] * len(inputs)
                else:
                    inputs, outputs = list(zip(*inputs))
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
        optimizer = AdamW8bit(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
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
        for self.epoch in range(self.epochs):
            nlls = []
            for self.step, batch in enumerate(dataloader):
                epoch_checkpoint_progress += len(batch['input_ids'])
                epoch_yield_progress += len(batch['input_ids'])
                with self.acclerator.accumulate(model):
                    loss = model(**batch).loss
                    self.acclerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                num_tokens = batch['labels'].ne(loss_mask).sum().item()
                nlls.append((loss.item() * num_tokens, num_tokens))
                display.update(len(batch['input_ids']))
                if (
                    self.checkpoint_after_every_x_epochs and
                    epoch_checkpoint_progress // int(self.checkpoint_after_every_x_epochs * len(inputs))
                ):
                    self.save_checkpoint()
                    epoch_checkpoint_progress = (
                        epoch_checkpoint_progress % int(self.checkpoint_after_every_x_epochs * len(inputs))
                    )
                if epoch_yield_progress // int(yield_every_x_epochs * len(inputs)):
                    total_nll = sum(nll for nll, _ in nlls)
                    total_tokens = sum(num_tokens for _, num_tokens in nlls)
                    perplexity = math.exp(total_nll / total_tokens)
                    nlls = []
                    epoch_yield_progress = epoch_yield_progress % int(yield_every_x_epochs * len(inputs))
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
            num_tokens = batch['labels'].ne(loss_mask).sum().item()
            if num_tokens > 0:
                nlls.append((loss.item() * num_tokens, num_tokens))
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
                eos_token_id=2
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
                generated = self.tokenizer.decode(gen[input_len:], skip_special_tokens=True)
                decoded_gens.append(generated)
        return decoded_gens[0] if single else decoded_gens


class DataCollatorWithPreprocessing:

    def __init__(self, tokenizer, model, return_tensors):
        self.seq2seq_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            pad_to_multiple_of=8
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



def main():
    pass


if __name__ == '__main__':
    main()

