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
import dataclasses as dc
from tqdm import tqdm
import ezpyzy as ez
import pathlib as pl
import shutil
import os
import math


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


loss_mask = -100


def load_merge_and_save_lora(lora_path: ez.filelike, merged_path: ez.filelike=None):
    lora_path = ez.File(lora_path).path
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


@dc.dataclass
class LlamaArgs:
    base: str = "meta-llama/Llama-2-{param_magnitude}-chat-hf"
    param_magnitude: str = '7b'
    format: str = '''[INST] <<SYS>> You are a helpful, respectful, and honest assistant. <</SYS>> {input} [/INST] {output} </s>'''
    train_on_s2s_inputs: bool = False
    quantize: str | None = 'nf4'
    checkpoint: str | None = None
    checkpoint_after_every_x_epochs: float | None = 1.0
    checkpoint_clean_up_after_train: bool = True
    data: str = None
    epoch: int = 0
    step: int = 0
    epochs: int = 1
    max_sequence_length: int = 4096
    protected_input_length: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optimizer: str = 'adamw_bnb_8bit'
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    max_gradient_norm: float = 0.3
    warmup_steps: int = 0
    lr_scheduler_type: str = 'constant'
    lora: int | None = 8
    lora_alpha: int | None = None
    lora_dropout: float | None = 0.1
    lora_modules: list[str] = None
    lora_merge_on_load: bool = True
    gradient_checkpointing: bool = True
    max_output_length: int = 512
    repetition_penalty: float = 1.2
    num_beams: int = 1
    temperature: float = 0.6
    sampled_generation: bool = False
    top_p: float = 0.9
    top_k: int = 50
    gen_batch_size: int = None
    experiment: str = None

@dc.dataclass
class Llama(LlamaArgs):

    def __post_init__(self):
        if '{param_magnitude}' in self.base:
            self.base = self.base.replace('{param_magnitude}', str(self.param_magnitude))
        tokenizer_reponame = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_reponame, trust_remote_code=True)
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
            self.gen_batch_size = self.actual_train_batch_size
        if self.lora is not None and self.lora_alpha is None:
            self.lora_alpha = self.lora * 2  # heuristic usually works well
        if self.lora_modules is None:
            self.lora_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
            ]
        self.model.eval()

    @property
    def actual_train_batch_size(self):
        return self.train_batch_size // self.gradient_accumulation_steps

    def save(self, path:ez.filelike):
        path = ez.File(path).path
        self.model.save_pretrained(path)

    def save_checkpoint(self, path: ez.filelike = None):
        if path is None:
            path = 'ex/scratch/checkpoint'
        path = ez.File(path).path
        self.acclerator.save_state(path)

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
                    output_ids = output_ids[:self.max_sequence_length - len(input_ids)]
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
        dataset = Dataset.from_list(datalist)
        return dataset

    def training(self, inputs=None, outputs=None, yield_every_x_epochs=1):
        if outputs is None:
            try:
                inputs, outputs = zip(*inputs)
            except ValueError:
                inputs, outputs = outputs, inputs
        dataset = self.preprocess(inputs, outputs)
        if self.lora is not None:
            if self.lora_merge_on_load:
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
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, return_tensors='pt', pad_to_multiple_of=8
        )
        dataloader = DataLoader(
            dataset, batch_size=self.actual_train_batch_size, collate_fn=collator, shuffle=True
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
        display = tqdm(total=self.epochs * len(dataset), desc='Training')
        epoch_checkpoint_progress = 0.0
        epoch_yield_progress = 0.0
        for self.epoch in range(self.epochs):
            epoch_checkpoint_progress = float(round(epoch_checkpoint_progress))
            epoch_yield_progress = float(round(epoch_yield_progress))
            nlls = []
            for self.step, batch in enumerate(dataloader):
                epoch_checkpoint_progress += 1.0 / len(dataloader)
                epoch_yield_progress += 1.0 / len(dataloader)
                with self.acclerator.accumulate(model):
                    loss = model(**batch).loss
                    self.acclerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                num_tokens = batch['labels'].ne(loss_mask).sum().item()
                nlls.append((loss.item() * num_tokens, num_tokens))
                display.update(len(batch.data['input_ids']))
                if (
                    self.checkpoint_after_every_x_epochs and
                    epoch_checkpoint_progress >= self.checkpoint_after_every_x_epochs
                ):
                    self.save_checkpoint()
                    epoch_checkpoint_progress = 0.0
                if epoch_yield_progress >= yield_every_x_epochs:
                    total_nll = sum(nll for nll, _ in nlls)
                    total_tokens = sum(num_tokens for _, num_tokens in nlls)
                    perplexity = math.exp(total_nll / total_tokens)
                    nlls = []
                    epoch_yield_progress = 0.0
                    yield perplexity
        if self.checkpoint_clean_up_after_train:
            shutil.rmtree('ex/scratch/checkpoint', ignore_errors=True)

    def train(self, inputs=None, outputs=None):
        return list(self.training(inputs=inputs, outputs=outputs))

    def perplexity(self, inputs, outputs=None):
        if outputs is None:
            if inputs and not isinstance(next(iter(inputs)), str):
                inputs, outputs = zip(*inputs)
            elif not inputs:
                inputs, outputs = outputs, inputs
        dataset = self.preprocess(inputs, outputs)
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, return_tensors='pt', pad_to_multiple_of=8
        )
        dataloader = DataLoader(
            dataset, batch_size=self.actual_train_batch_size, collate_fn=collator, shuffle=True
        )
        dataloader, model = self.acclerator.prepare(dataloader, self.model)  # noqa
        display = tqdm(total=self.epochs * len(dataloader), desc='Calculating Perplexity')
        nlls = []
        for step, batch in enumerate(dataloader):
            loss = model(**batch).loss
            num_tokens = batch['labels'].ne(loss_mask).sum().item()
            nlls.append((loss.item() * num_tokens, num_tokens))
            display.update(len(batch.data['input_ids']))
        total_nll = sum(nll for nll, _ in nlls)
        total_tokens = sum(num_tokens for _, num_tokens in nlls)
        ppl = math.exp(total_nll / total_tokens)
        self.model.eval()
        return ppl

    def generate(self, prompt):
        with ez.shush():
            single = False
            if isinstance(prompt, str):
                prompt = [prompt]
                single = True
            dataset = self.preprocess(inputs=prompt)
            collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, return_tensors='pt', pad_to_multiple_of=8
            )
            dataloader = DataLoader(
                dataset, batch_size=self.gen_batch_size, collate_fn=collator, shuffle=False
            )
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
            input_lens = [len(case['input_ids']) for case in dataset]
            encoded_gens = []
            for step, batch in enumerate(dataloader):
                batch = {k: v.to('cuda') for k, v in batch.items()}
                gen = self.model.generate(
                    **batch, generation_config=config, pad_token_id=self.tokenizer.eos_token_id
                )
                encoded_gens.extend(gen)
            decoded_gens = []
            for gen, input_len in zip(encoded_gens, input_lens):
                generated = self.tokenizer.decode(gen[input_len:], skip_special_tokens=True)
                decoded_gens.append(generated)

            return decoded_gens[0] if single else decoded_gens


def main():
    import ezpyzy as ez

    from test.check_lm import models, eval, data_capital_langs

    model_name = 'Llama'
    Model = Llama

    with ez.check("Load LoRA"):
        model = Model(
            f'ex/test/{model_name}/lora_capital_langs',
            lora_merge_on_load=False
        )
        print('Perplexity:', model.perplexity(data_capital_langs.items()))
        eval(
            model, {
                "Kyiv": "Ukrainian",
                "London": "English",
                "Washington D.C.": "English",
                "Hanoi": "Vietnamese"
            }.items()
        )


if __name__ == '__main__':
    main()

