
from __future__ import annotations

import dataclasses as dc

import ezpyzy as ez


@dc.dataclass
class Tokenizer(ez.Config):
    ...

    def __post_init__(self):
        super().__post_init__()
        assert type(self) is not Tokenizer, "Use a HuggingfaceTokenizer, not Tokenizer itself."


@dc.dataclass
class HuggingfaceTokenizerConfig(ez.Config):
    repo_id: str = None

    def __post_init__(self):
        super().__post_init__()
        assert self.repo_id is not None, "A repo_id must be provided to HuggingfaceTokenizerConfig."

@dc.dataclass
class HuggingfaceTokenizer(ez.Implementation, HuggingfaceTokenizerConfig):
    ...