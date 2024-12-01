
from __future__ import annotations

import dataclasses as dc
import sys
import ezpyzy as ez

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dc.dataclass
class Tokenizer(ez.Config):
    slot_affix_replacements: dict[str, str] = {}
    slot_lead_pattern: str = r''
    slot_trail_pattern: str = r''
    pad_token_id: int = 0

    def __post_init__(self):
        super().__post_init__()

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError


@dc.dataclass
class HuggingfaceTokenizerConfig(Tokenizer):
    repo_id: str = None
    slot_lead_pattern: str = r" ?"

    def __post_init__(self):
        super().__post_init__()

@dc.dataclass
class HuggingfaceTokenizer(ez.ImplementsConfig, HuggingfaceTokenizerConfig):
    tokenizer: ... = None

    def __post_init__(self):
        super().__post_init__()
        try:
            import transformers as hf
        except ImportError as e:
            print('''Could not import huggingface transformers-- make sure it is installed in your python environment.''', file=sys.stderr)
            raise e
        self.tokenizer: hf.PreTrainedTokenizer
        if isinstance(self.repo_id, str):
            self.tokenizer = hf.AutoTokenizer.from_pretrained(self.repo_id)
        self.tokenizer.pad_token_id = self.tokenizer.encode('-')[0]
        self.tokenizer.pad_token = '-'
        self.pad_token_id = self.tokenizer.pad_token_id
        discovered_affix_replacements = dict(
            bos=self.tokenizer.bos_token,
            eos=self.tokenizer.eos_token,
            pad=self.tokenizer.pad_token)
        self.slot_affix_replacements = {
            **discovered_affix_replacements, **(self.slot_affix_replacements or {})}

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def __str__(self):
        fields = {f.name: getattr(self, f.name) for f in dc.fields(self.__config_implemented__)} # noqa
        return f"{self.__class__.__name__}({', '.join(k+': '+repr(v) for k, v in fields.items())})"
    __repr__ = __str__


if __name__ == '__main__':

    llama3tokenzier = HuggingfaceTokenizer(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct')
    encoded = llama3tokenzier.encode("Hello, this is a test!")
    print(encoded)
    for token in encoded:
        print(token, llama3tokenzier.decode([token]))


