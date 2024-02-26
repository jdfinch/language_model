import peft
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    get_constant_schedule_with_warmup as warmup_scheduler,
    Adafactor
)
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig, TaskType

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
    base_model = T5ForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model = peft.PeftModel.from_pretrained(base_model, lora_path)
    merged = model.merge_and_unload()
    if merged_path is None:
        merged_path = lora_path.parent / f"{name}.MERGED"
    merged.save_pretrained(merged_path, safe_serialization=False, save_peft_format=False)
    return merged_path


@settings
class T5Hyperparameters(ez.Settings):
    base: ez.ColStr = ez.Def('t5-{param_magnitude}')
    param_magnitude: ez.ColStr = ez.Def('small')
    format: ez.ColStr = ez.Def('question-answering') # ': ' gets dynamically added
    train_on_s2s_inputs: ez.ColBool = ez.Def(False)
    quantize: ez.ColStr = ez.Def('bf16')
    checkpoint: ez.ColStr = None
    checkpoint_after_every_x_epochs: ez.ColFloat = ez.Def(1.0)
    checkpoint_clean_up_after_train: ez.ColBool = ez.Def(True)
    epoch: ez.ColInt = ez.Def(0)
    step: ez.ColInt = ez.Def(0)
    epochs: ez.ColInt = ez.Def(1)
    max_sequence_length: ez.ColInt = ez.Def(512)
    train_batch_size: ez.ColInt = ez.Def(1)
    gradient_accumulation_steps: ez.ColInt = ez.Def(1)
    optimizer: ez.ColStr = ez.Def('adamw_bnb_8bit')
    learning_rate: ez.ColFloat = ez.Def(1e-4)
    weight_decay: ez.ColFloat = ez.Def(0.0)
    warmup_steps: ez.ColInt = ez.Def(0)
    lr_scheduler_type: ez.ColStr = ez.Def('constant')
    lora: ez.ColInt = ez.Def(8)
    lora_alpha: ez.ColInt = None
    lora_dropout: ez.ColFloat = ez.Def(0.1)
    lora_modules: ez.Column[list[str]] | list[str] | None = None
    lora_merge_on_load: ez.ColBool = ez.Def(True)
    gradient_checkpointing: ez.ColBool = ez.Def(True)
    max_output_length: ez.ColInt = ez.Def(512)
    repetition_penalty: ez.ColFloat = ez.Def(2.0)
    num_beams: ez.ColInt = ez.Def(1)
    temperature: ez.ColFloat = ez.Def(0.6)
    sampled_generation: ez.ColBool = ez.Def(False)
    top_p: ez.ColFloat = ez.Def(0.9)
    top_k: ez.ColInt = ez.Def(50)
    gen_batch_size: ez.ColInt = ez.Def(1)
    ppl_batch_size: ez.ColInt = ez.Def(1)
    dynamic_tokenization: ez.ColBool = ez.Def(True)
    use_local_files: ez.ColBool = ez.Def(False)

    def actual_train_batch_size(self):
        return self.train_batch_size // self.gradient_accumulation_steps

    def __post_init__(self):
        if '{param_magnitude}' in self.base:
            self.base = self.base.replace('{param_magnitude}', str(self.param_magnitude))



class T5(T5Hyperparameters):
    def __post_init__(self):
        T5Hyperparameters.__post_init__(self)
        if pl.Path(self.base).exists() and (pl.Path(self.base)/'hyperparameters.json').exists():
            loaded_hyperparams:dict = ez.File(pl.Path(self.base)/'hyperparameters.json').load()
            specified_hyperparameters = vars(self).pop('settings')
            hyperparameters = {**loaded_hyperparams, **specified_hyperparameters}
            vars(self).update(hyperparameters)
        self.hyperparameters: dict = dict(vars(self))
        tokenizer_reponame = "t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_reponame,
            padding_side='right',
            truncation_side='right',
            model_max_length=1000000, # high number; truncation handled manually
            trust_remote_code=True,
            local_files_only=self.use_local_files
        )
        quant_kwargs = {'device_map': 'auto'}
        if self.quantize is not None:
            if self.quantize == 'nf4':
                quant_kwargs.update(dict(
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False
                    ),
                    torch_dtype=torch.bfloat16,
                ))
            elif self.quantize == 'int8':
                quant_kwargs.update(dict(
                    load_in_8bit=True,
                ))
            elif self.quantize == 'bf16':
                quant_kwargs.update(dict(
                    torch_dtype=torch.bfloat16,
                ))
            else:
                raise ValueError(
                    f"Invalid quantization level: {self.quantize}.\n"
                    f"Supported quantizations are: 'nf4', 'int8', 'bf16', None"
                )

        load_path = pl.Path(self.base)
        delete_merge_path = None
        if load_path.exists() and (load_path / 'adapter_config.json').exists() and self.lora_merge_on_load:
            merged_path = load_path.parent / f"{load_path.name}.MERGED"
            delete_after = not merged_path.exists()
            ez.subproc(load_merge_and_save_lora, load_path)
            if delete_after:
                delete_merge_path = merged_path
            load_path = merged_path
        else:
            load_path = self.base
        self.model = T5ForConditionalGeneration.from_pretrained(
            load_path,
            local_files_only=self.use_local_files, 
            **quant_kwargs
        )
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
                'q', 'v'
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
            raise Exception("You must pass inputs to T5 model!")
        data = zip(inputs, outputs)
        if self.format != '':
            input_pre_format = self.format
            pre_format_tokens = self.tokenizer(input_pre_format + ': ', add_special_tokens=False, padding=False, truncation=False)['input_ids']
        else:
            pre_format_tokens = []
        num_format_tokens = len(pre_format_tokens)
        datalist = []
        for input, output in data:
            input_tokens = self.tokenizer(
                input, add_special_tokens=False, padding=False, truncation=False
            )['input_ids']
            overflow = num_format_tokens + len(input_tokens) + 1 - self.max_sequence_length
            if overflow > 0:
                input_tokens = input_tokens[overflow:]
            all_input_tokens = (
                pre_format_tokens + input_tokens +
                [self.tokenizer.eos_token_id]
            )
            output_tokens = self.tokenizer(
                text_target=output, add_special_tokens=True, padding=False, max_length=self.max_output_length, truncation=True
            )['input_ids'] if output else []
            labels = output_tokens
            attention_mask = [1] * len(all_input_tokens)
            assert len(all_input_tokens) <= self.max_sequence_length, \
                f"Token length {len(all_input_tokens)} exceeds max sequence length {self.max_sequence_length}"
            datalist.append(dict(input_ids=all_input_tokens, attention_mask=attention_mask, labels=labels))
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
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False
                )
                self.model = peft.get_peft_model(self.model, peft_parameters) # noqa
                self.model.base_model.model.enable_input_require_grads()
        self.model.train()
        dataloader = self._set_up_dataloader(
            inputs, outputs, self.actual_train_batch_size(), shuffle=True
        )
        if self.optimizer == 'adamw_bnb_8bit':
            print('Using AdamW8bit optimizer...')
            optimizer = AdamW8bit(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'adafactor':
            print('Using Adafactor optimizer...')
            optimizer = Adafactor(
                self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, relative_step=False
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
                top_k=self.top_k
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
                    pad_token_id=self.tokenizer.pad_token_id
                )
            encoded_gens.extend(gen)
            display.update(len(batch.data['input_ids']))
        with ez.shush():
            decoded_gens = []
            for gen in encoded_gens:
                generated = self.tokenizer.decode(gen, skip_special_tokens=True)
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




