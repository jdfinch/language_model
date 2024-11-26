from unsloth import FastLanguageModel

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
    model_base: str = "unsloth/Llama-3.2-1B-Vision-Instruct"
    tokenizer_templates: lt.Llama3TemplateTokenizer = lt.Llama3TemplateTokenizerConfig()


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
        segments_values: list[Template|dict[str,str]]|T.Iterable[list[Template|dict[str,str]]]
    ) -> str|list[str]:
        us.FastLanguageModel.for_inference(model.model)
        tokenized = self.tokenizer_templates.fill(segments_values)
        tokens = tokenized.dict(with_labels=False,
            seq_type=ft.partial(pt.tensor, dtype=pt.int, device=self.device))
        input_length = tokens['input_ids'].shape[1]
        responses_tokens = self.model.generate(**tokens,
            pad_token_id=self.tokenizer_templates.tokenizer.pad_token_id)
        responses = []
        for response_tokens in responses_tokens:
            response = self.tokenizer_templates.tokenizer.decode(response_tokens[input_length:-1])
            responses.append(response)
        return responses

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
        FastLanguageModel.for_training(self.model)
        if self.training.optimizer.optimizer is None:
            self.training.optimizer.optimizer = self.training.optimizer.construct_optimizer(self.model)
        scheduler = self.training.scheduler.construct_scheduler(self.training.optimizer.optimizer)
        for epoch in range(self.training.epochs):
            def train_epoch_each_step():
                samples_trained = 0
                nlls = []
                for physical_step, batch in enumerate(ez.batching(data, size=self.training.physical_batch_size)):
                    tokens = self.tokenizer_templates.fill(batch).dict(with_labels=True,
                        seq_type=ft.partial(pt.tensor, dtype=pt.int, device=self.device))
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

    model = Llama3()

    dialogue = [
        lt.System("You are a pirate."),
        lt.User("Where were you yesterday??"),
        lt.Response(...),
    ]

    response, = model.generate(dialogue)

    print(response)

    print(f"\nGPU allocated {pt.cuda.max_memory_allocated() / 1e9:.2f} GB")

