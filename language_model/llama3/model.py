import tqdm
from transformers import AutoModelForCausalLM

import ezpyzy as ez
import dataclasses as dc
import functools as ft
import itertools as it
import json
import gc
import pathlib as pl
import math

from language_model.tokens.template import Template
from language_model.tokens.token_sequences import TokenSequence, TokenSequences
from language_model.language_model_config import LanguageModelConfig
from language_model.lora import LoRA

import transformers as hf
import torch as pt

import language_model.llama3.templates as lt

import typing as T


@dc.dataclass
class Llama3Config(LanguageModelConfig):
    model_base: str = "meta-llama/Llama-3.2-1B-Instruct"
    template_tokenizer: lt.Llama3TemplateTokenizerConfig = lt.Llama3TemplateTokenizerConfig()

    def __post_init__(self):
        super().__post_init__()
        if self.generation and self.training and not self.generation.configured.has.batch_size:
            self.generation.batch_size = self.training.physical_batch_size

@dc.dataclass
class Llama3(ez.ImplementsConfig, Llama3Config):
    template_tokenizer: lt.Llama3TemplateTokenizer = lt.Llama3TemplateTokenizerConfig()

    def __post_init__(self):
        super().__post_init__()
        self.reload_model()

    def reload_model(self):
        assert isinstance(self.model_base, str), \
            "The base model repository ID must be a string."
        attn = dict(attn_implementation='flash_attention_2') if self.device != 'cpu' else {}
        AutoModelForCausalLM = ft.partial(hf.AutoModelForCausalLM.from_pretrained,
            **attn, torch_dtype=pt.bfloat16)
        model = AutoModelForCausalLM(self.model_base, # noqa
            local_files_only=self.load_locally_saved_models_only,
            load_in_4bit='nf4' == self.quantization,
            load_in_8bit='int8' == self.quantization,
            device_map={'': self.device})
        self.model: hf.LlamaForCausalLM = model # noqa
        if self.adapters:
            for name, adapter in self.adapters:
                if isinstance(adapter, LoRA):
                    if adapter.trained:
                        self.model.load_adapter(adapter.repo_id, adapter_name=name, device_map=self.device)
                        if adapter.lora_merge_on_load:
                            self.activate_adapter(name)
                            self.merge_adapter()
                            self.activate_adapter(None)
            self.activate_adapter(self.active_adapter)

    def delete(self):
        self.model = None
        if self.training:
            self.training.optimizer.optimizer = None
            self.training.scheduler.scheduler = None
        gc.collect()
        pt.cuda.empty_cache()

    def save(self, path):
        path_str = path
        path = pl.Path(path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        if self.adapter:
            self.adapter.repo_id = str(path_str)
        self.configured.save(path/'language_model_config.json')
        with ez.shush: self.model.save_pretrained(path)
        if self.training and self.training.optimizer.optimizer:
            pt.save(self.training.optimizer.optimizer.state_dict(), path/'optimizer.pt')
        if self.training and self.training.scheduler.scheduler:
            pt.save(self.training.scheduler.scheduler.state_dict(), path/'scheduler.pt')

    @property
    def adapter(self) -> LoRA|None:
        return getattr(self.adapters, self.active_adapter, None)

    def activate_adapter(self, name: str | None):
        self.active_adapter = name
        if self.active_adapter and self.adapter.trained:
            self.model.set_adapter(name)
        else:
            try: self.model.disable_adapters()
            except ValueError: pass

    def deactivate_adapter(self):
        self.activate_adapter(None)

    def merge_adapter(self):
        raise NotImplementedError

    def unmerge_adapter(self):
        assert getattr(self.adapters, self.adapters.active).repo_id is not None, \
            f"Unmerging adapter {self.adapters.active} requires re-loading the adapter and the base model separately, but the this active adapter does not have a repo_id field. Save the adapter to disk before calling unmerge_adapter() so it can be re-loaded from its repo_id path."
        raise NotImplementedError

    def generate(self,
        data: list[Template | dict[str,str]] | T.Iterable[list[Template | dict[str,str]]]
    ) -> list[str|None]:
        return list(self.each_generation(data))

    def each_generation(self,
        data: list[Template | dict[str,str]] | T.Iterable[list[Template | dict[str,str]]]
    ) -> T.Iterable[str|None]:
        self.model.eval()
        generation_config = self.generation.construct_hf_config()
        generation_config.pad_token_id = self.template_tokenizer.tokenizer.pad_token_id
        if isinstance(data, list) and data and isinstance(data[0], (dict, Template)):
            data = [data]
        generation_groups = {}
        response_queue = {}
        waiting_for_response_index = 0
        data_iterator = enumerate(data)
        data_iteration = next(data_iterator, None)
        generation_group_iterator = None
        progress_total = len(data) if hasattr(data, '__len__') else None
        progress = tqdm.tqdm(total=progress_total, desc="Generating")
        while data_iteration or generation_groups:
            if data_iteration:
                i, data_item = data_iteration
                data_iteration = next(data_iterator, None)
                gen_slot_info = self.template_tokenizer.find_gen_slot(data_item)
                if gen_slot_info is None:
                    response_queue[i] = None
                    continue
                i_gen_segment, gen_slot, n_expected_out = gen_slot_info
                eos_token_id = self.template_tokenizer.template_slots_eos[
                    gen_slot.template.name, gen_slot.slot_index]
                prompt_tokens = self.template_tokenizer._tokenize_sequence(data_item, gen_slot_info)
                max_out = self.template_tokenizer.max_length - len(prompt_tokens)
                if n_expected_out is not None and n_expected_out < max_out: max_out = n_expected_out
                generation_group_key = (max_out, eos_token_id)
                generation_group = generation_groups.setdefault(generation_group_key, [])
                generation_group.append((i, data_item, gen_slot_info, prompt_tokens))
                if len(generation_group) < self.generation.batch_size:
                    continue
                else:
                    del generation_groups[generation_group_key]
            elif generation_group_iterator is None:
                generation_group_iterator = iter(generation_groups.items())
                continue
            else:
                generation_group_iteration = next(generation_group_iterator, None)
                if generation_group_iteration is None:
                    return
                (max_out, eos_token_id), generation_group = generation_group_iteration
            data_item_indices, data_batch, gen_slot_infos, prompts_tokens = zip(*generation_group)
            tokenized_batch = TokenSequences(prompts_tokens,
                pad_to_same_length=True,
                pad_to_multiple_of=self.template_tokenizer.pad_to_multiple_of,
                pad_side=self.template_tokenizer.pad_side,
                tokenizer=self.template_tokenizer.tokenizer)
            tokens_batch = tokenized_batch.dict(with_labels=False,
                seq_type=ft.partial(pt.tensor, dtype=pt.long, device=self.device))
            input_length = tokens_batch['input_ids'].shape[1]
            generation_config.eos_token_id = eos_token_id
            generation_config.max_new_tokens = max_out
            response_batch = self.model.generate(**tokens_batch, generation_config=generation_config)
            for data_item_index, response_tokens, gen_slot_info, data_item in zip(
                data_item_indices, response_batch, gen_slot_infos, data_batch
            ):
                response_tokens = response_tokens[input_length:]
                if eos_token_id is not None:
                    eos_token_indices = (response_tokens == eos_token_id).nonzero() # noqa
                    if eos_token_indices.numel():
                        response_tokens = response_tokens[:eos_token_indices[0].item()]
                response_text = self.template_tokenizer.tokenizer.decode(response_tokens)
                response_queue[data_item_index] = response_text
                segment_index, slot, _ = gen_slot_info
                segment = data_item[segment_index]
                if isinstance(segment, dict):
                    segment[slot.name] = response_text
                else:
                    setattr(segment, slot.name, response_text)
                while waiting_for_response_index in response_queue:
                    response = response_queue.pop(waiting_for_response_index)
                    progress.update(1)
                    yield response
                    waiting_for_response_index += 1
        while waiting_for_response_index in response_queue:
            response = response_queue.pop(waiting_for_response_index)
            progress.update(1)
            yield response
            waiting_for_response_index += 1
        progress.close()

    def start_training(self):
        if self.adapter and not self.adapter.trained:
            self.model.add_adapter(self.adapter.get_peft_config(), self.active_adapter)
            self.adapter.trained = True
        self.model.train()
        if self.training.optimizer.optimizer is None or not self.training.resume_previous_training:
            if self.training.optimizer.optimizer is None         and self.loaded_model_path:
                self.training.optimizer.optimize(self.model)
                self.training.scheduler.schedule(self.training.optimizer)
                if self.training.resume_previous_training:
                    path = pl.Path(self.loaded_model_path).expanduser()
                    if (optpath:=(path/'optimizer.pt')).exists():
                        self.training.optimizer.optimizer.load_state_dict(pt.load(optpath)) # noqa
                    if (schpath:=(path/'scheduler.pt')).exists():
                        self.training.scheduler.scheduler.load_state_dict(pt.load(schpath))
            else:
                self.training.optimizer.optimize(self.model)
                self.training.scheduler.schedule(self.training.optimizer)
        if not self.training.resume_previous_training:
            self.training.current_epoch = 0
            self.training.current_step = 0
        self.training.optimizer.optimizer.zero_grad()
        return self.training.optimizer.optimizer, self.training.scheduler.scheduler

    def train_each_epoch(self, data: T.Iterable[list[Template|dict[str,str]]]):
        self.start_training()
        if self.training.shuffle_data:
            data = list(data)
        progress_total = (len(data)//self.training.batch_size * self.training.batch_size
            ) * self.training.epochs if hasattr(data, '__len__') else None
        progress = tqdm.tqdm(total=progress_total, desc='Training')
        starting_epoch = self.training.current_epoch + 1
        for self.training.current_epoch in range(1, self.training.epochs+1):
            if self.training.shuffle_data:
                data = list(data)
                self.rng.shuffle(data)
            if self.training.current_epoch < starting_epoch: continue
            samples_trained = 0
            nlls, nums_tokens = [], []
            batch_iter = iter(ez.batching(data, size=self.training.physical_batch_size))
            item_ff = 0
            while item_ff // self.training.batch_size < self.training.current_step:
                item_ff += len(next(batch_iter))
            for physical_step, batch in enumerate(batch_iter, start=1):
                tokens = self.template_tokenizer.tokenize(batch).dict(
                    with_labels=True,
                    seq_type=ft.partial(pt.tensor, dtype=pt.long, device=self.device))
                num_tokens = tokens['input_ids'].ne(-100).sum().item()
                nums_tokens.append(num_tokens)
                loss = self.model(**tokens).loss / self.training.gradient_accumulation_steps
                nll = loss.item()
                nlls.append(nll)
                loss.backward()
                if physical_step % self.training.gradient_accumulation_steps == 0:
                    self.training.current_step += 1
                    self.training.optimizer.optimizer.step()
                    self.training.scheduler.scheduler.step()
                    self.training.optimizer.optimizer.zero_grad()
                    progress.update(self.training.batch_size)
                samples_trained += len(batch)
            epoch_nll, epoch_n_tokens = sum(nll * n for nll, n in zip(nlls, nums_tokens)), sum(nums_tokens)
            ppl = math.exp(epoch_nll / epoch_n_tokens)
            self.training.current_step = 0
            yield ppl
        self.training.optimizer.optimizer.zero_grad()
        progress.close()

    def train_each_step(self, data: T.Iterable[list[Template|dict[str,str]]]):
        self.start_training()
        starting_epoch = self.training.current_epoch + 1
        if self.training.shuffle_data:
            data = list(data)
        progress_total = (len(data)//self.training.batch_size * self.training.batch_size
            ) * self.training.epochs if hasattr(data, '__len__') else None
        progress = tqdm.tqdm(total=progress_total, desc='Training')
        for self.training.current_epoch in range(1, self.training.epochs + 1):
            if self.training.shuffle_data:
                self.rng.shuffle(data)
            if self.training.current_epoch < starting_epoch: continue
            samples_trained = 0
            nlls, nums_tokens = [], []
            batch_iter = iter(ez.batching(data, size=self.training.physical_batch_size))
            item_ff = 0
            while item_ff // self.training.batch_size < self.training.current_step:
                item_ff += len(next(batch_iter))
            for physical_step, batch in enumerate(batch_iter, start=1):
                tokens = self.template_tokenizer.tokenize(batch).dict(
                    with_labels=True,
                    seq_type=ft.partial(pt.tensor, dtype=pt.long, device=self.device))
                num_tokens = tokens['input_ids'].ne(-100).sum().item()
                nums_tokens.append(num_tokens)
                loss = self.model(**tokens).loss / self.training.gradient_accumulation_steps
                nll = loss.item()
                nlls.append(nll)
                loss.backward()
                if physical_step % self.training.gradient_accumulation_steps == 0:
                    self.training.current_step += 1
                    self.training.optimizer.optimizer.step()
                    self.training.scheduler.scheduler.step()
                    self.training.optimizer.optimizer.zero_grad()
                    step_nll, step_n_tokens = sum(nll * n for nll, n in zip(nlls, nums_tokens)), sum(nums_tokens)
                    ppl = math.exp(step_nll / step_n_tokens)
                    nlls, nums_tokens = [], []
                    progress.update(self.training.batch_size)
                    yield ppl
                samples_trained += len(batch)
            self.training.current_step = 0
        self.training.optimizer.optimizer.zero_grad()
        progress.close()

    def train_each_step_each_epoch(self, data: T.Iterable[list[Template|dict[str,str]]]):
        self.start_training()
        if self.training.shuffle_data:
            data = list(data)
        progress_total = (len(data)//self.training.batch_size * self.training.batch_size
            ) * self.training.epochs if hasattr(data, '__len__') else None
        progress = tqdm.tqdm(total=progress_total, desc='Training')
        starting_epoch = self.training.current_epoch + 1
        for self.training.current_epoch in range(1, self.training.epochs + 1):
            if self.training.shuffle_data:
                self.rng.shuffle(data)
            if self.training.current_epoch < starting_epoch: continue
            def train_epoch_each_step(data=data):
                samples_trained = 0
                nlls, nums_tokens = [], []
                batch_iter = iter(ez.batching(data, size=self.training.physical_batch_size))
                item_ff = 0
                while item_ff // self.training.batch_size < self.training.current_step:
                    item_ff += len(next(batch_iter))
                for physical_step, batch in enumerate(batch_iter, start=1):
                    tokens = self.template_tokenizer.tokenize(batch).dict(with_labels=True,
                        seq_type=ft.partial(pt.tensor, dtype=pt.long, device=self.device))
                    num_tokens = tokens['input_ids'].ne(-100).sum().item()
                    nums_tokens.append(num_tokens)
                    loss = self.model(**tokens).loss / self.training.gradient_accumulation_steps
                    nll = loss.item()
                    nlls.append(nll)
                    loss.backward()
                    if physical_step % self.training.gradient_accumulation_steps == 0:
                        self.training.current_step += 1
                        self.training.optimizer.optimizer.step()
                        self.training.scheduler.scheduler.step()
                        self.training.optimizer.optimizer.zero_grad()
                        step_nll, step_n_tokens = sum(nll*n for nll, n in zip(nlls, nums_tokens)), sum(nums_tokens)
                        ppl = math.exp(step_nll / step_n_tokens)
                        nlls, nums_tokens = [], []
                        progress.update(self.training.batch_size)
                        yield ppl
                    samples_trained += len(batch)
            yield train_epoch_each_step(data=data)
            self.training.current_step = 0
        self.training.optimizer.optimizer.zero_grad()
        progress.close()



if __name__ == '__main__':

    config = Llama3Config()

    with ez.Timer('Initializing'):
        model = Llama3()

    dialogue = [
        lt.System("You are a pirate."),
        lt.User("Where were you yesterday??"),
        lt.Assistant(...),
    ]

    with ez.Timer('Generating'):
        response, = model.generate(dialogue)
        print(response)

    print(f"\nGPU allocated {pt.cuda.max_memory_allocated() / 1e9:.2f} GB")

