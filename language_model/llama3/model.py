
import ezpyzy as ez
import dataclasses as dc
import functools as ft
import itertools as it

from language_model.tokens.template import Template
from language_model.tokens.token_sequences import TokenSequence, TokenSequences
from language_model.language_model_config import LanguageModelConfig

import transformers as hf
import torch as pt
with ez.shush:
    import unsloth as us

import language_model.llama3.templates as lt

import typing as T


@dc.dataclass
class Llama3Config(LanguageModelConfig):
    model_base: str = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    template_tokenizer: lt.Llama3TemplateTokenizer = lt.Llama3TemplateTokenizerConfig()

    def __post_init__(self):
        super().__post_init__()
        if self.generation and self.training and not self.generation.configured.has.batch_size:
            self.generation.batch_size = self.training.physical_batch_size


@dc.dataclass
class Llama3(ez.ImplementsConfig, Llama3Config):
    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.model_base, str), \
            "The base model repository ID must be a string."
        with ez.shush:
            model, _ = us.FastLanguageModel.from_pretrained(self.model_base,
                local_files_only=self.load_locally_saved_models_only)
        del _
        self.model: hf.LlamaForCausalLM = model

    def generate(self,
        data: list[Template | dict[str,str]] | T.Iterable[list[Template | dict[str,str]]]
    ) -> list[str|None]:
        return list(self.each_generation(data))

    def each_generation(self,
        data: list[Template | dict[str,str]] | T.Iterable[list[Template | dict[str,str]]]
    ) -> T.Iterable[str|None]:
        us.FastLanguageModel.for_inference(self.model)
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
                max_out = min(self.template_tokenizer.max_length - len(prompt_tokens), n_expected_out)
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
                _, gen_slot, _ = gen_slot_info
                trunc_text_suffix = gen_slot.trunc_text
                if eos_token_id is not None:
                    eos_token_indices = (response_tokens == eos_token_id).nonzero() # noqa
                    if eos_token_indices.numel():
                        response_tokens = response_tokens[:eos_token_indices[0].item()]
                        trunc_text_suffix = ''
                response_text = self.template_tokenizer.tokenizer.decode(response_tokens)
                if trunc_text_suffix:
                    response_text += trunc_text_suffix
                response_queue[data_item_index] = response_text
                segment_index, slot, _ = gen_slot_info
                setattr(data_item[segment_index], slot.name, response_text)
                while waiting_for_response_index in response_queue:
                    response = response_queue.pop(waiting_for_response_index)
                    yield response
                    waiting_for_response_index += 1
        while waiting_for_response_index in response_queue:
            response = response_queue.pop(waiting_for_response_index)
            yield response
            waiting_for_response_index += 1

    def train_each_epoch(self, data: T.Iterable[list[Template|dict[str,str]]]):
        for epoch, steps in enumerate(self.train_each_step_each_epoch(data)):
            for step, ppl in enumerate(steps):
                ...
            yield ...

    def train_each_step(self, data: T.Iterable[list[Template|dict[str,str]]]):
        for epoch, steps in enumerate(self.train_each_step_each_epoch(data)):
            for step, ppl in enumerate(steps):
                yield ppl

    def train_each_step_each_epoch(self, data: T.Iterable[list[Template|dict[str,str]]]):
        us.FastLanguageModel.for_training(self.model)
        if self.training.optimizer.optimizer is None:
            self.training.optimizer.optimizer = self.training.optimizer.construct_optimizer(self.model)
        scheduler = self.training.scheduler.construct_scheduler(self.training.optimizer.optimizer)
        for epoch in range(self.training.epochs):
            def train_epoch_each_step():
                samples_trained = 0
                nlls = []
                for physical_step, batch in enumerate(ez.batching(data, size=self.training.physical_batch_size)):
                    tokens = self.template_tokenizer.tokenize(batch).dict(with_labels=True,
                        seq_type=ft.partial(pt.tensor, dtype=pt.long, device=self.device))
                    num_tokens = tokens['input_ids'].ne(-100).sum().item()
                    loss = self.model(**tokens).loss
                    nlls.append(loss.item() / num_tokens)
                    loss.backward()
                    if physical_step % self.training.gradient_accumulation_steps == 0:
                        self.training.optimizer.optimizer.step()
                        scheduler.step()
                        self.training.optimizer.optimizer.zero_grad()
                        perplexity = pt.exp(pt.tensor(nlls).mean())
                        yield perplexity
                        nlls = []
                    samples_trained += len(batch)
            yield train_epoch_each_step()


if __name__ == '__main__':

    config = Llama3Config()

    with ez.Timer('Initializing'):
        model = Llama3()

    dialogue = [
        lt.System("You are a pirate."),
        lt.User("Where were you yesterday??"),
        lt.Response(...),
    ]

    with ez.Timer('Generating'):
        response, = model.generate(dialogue)
        print(response)

    print(f"\nGPU allocated {pt.cuda.max_memory_allocated() / 1e9:.2f} GB")

