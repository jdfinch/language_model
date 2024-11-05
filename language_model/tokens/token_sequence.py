
from __future__ import annotations

import copy as cp
import dataclasses as dc
import ezpyzy.ansi as ansi

from language_model.tokens.tokenizer import Tokenizer


class TokenSequence:
    tokenizer: Tokenizer = None

    def __init__(self,
        sequence: str|'TokenSequence' = '',
        is_attended: bool = True,
        is_label:bool = False,
        tokenizer: Tokenizer = None,
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        assert self.tokenizer is not None, "A tokenizer must be provided to TokenSequence."
        if isinstance(sequence, str) and sequence:
            self.token_ids = self.tokenizer.encode(sequence)
            self.is_attendeds = [is_attended] * len(self.token_ids)
            self.is_labels = [is_label] * len(self.token_ids)
        elif isinstance(sequence, TokenSequence):
            self.token_ids = list(sequence.token_ids)
            self.is_attendeds = list(sequence.is_attendeds)
            self.is_labels = list(sequence.is_labels)
        elif isinstance(sequence, str):
            self.token_ids: list[int] = []
            self.is_attendeds: list[bool] = []
            self.is_labels: list[bool] = []
        else: raise ValueError(f"Invalid sequence type in constructor: {type(sequence)}")

    def extend(self, sequence: str|'TokenSequence', is_attended: bool = True, is_label:bool = False):
        if isinstance(sequence, str):
            token_ids = self.tokenizer.encode(sequence)
            self.token_ids.extend(token_ids)
            self.is_attendeds.extend([is_attended] * len(self.token_ids))
            self.is_labels.extend([is_label] * len(self.token_ids))
        elif isinstance(sequence, TokenSequence):
            self.token_ids.extend(sequence.token_ids)
            self.is_attendeds.extend(sequence.is_attendeds)
            self.is_labels.extend(sequence.is_labels)
        else: raise ValueError(f"Invalid concatenating sequence type: {type(sequence)}")
        return self
    __iadd__ = extend

    def dict(self, seq_type: type|callable = list):
        if seq_type is list:
            return dict(
                input_ids=self.token_ids,
                attention_mask=self.is_attendeds,
                labels=self.is_labels,)
        else:
            return dict(
                input_ids=seq_type(self.token_ids),
                attention_mask=seq_type(self.is_attendeds),
                labels=seq_type(self.is_labels),)

    def text(self):
        return self.tokenizer.decode(self.token_ids)

    def tokens(self):
        return [self.tokenizer.decode([token_id]) for token_id in self.token_ids]

    def display(self):
        return f"{ansi.bold}{self.__class__.__name__} with {len(self)} tokens:{ansi.reset}\n{self.ansi()}"

    def __add__(self, other):
        copy = cp.deepcopy(self)
        copy.extend(other)
        return copy

    def __iter__(self):
        return iter(Token(self.tokenizer.decode([token_id]), token_id, is_attended, is_label)
                for token_id, is_attended, is_label in zip(self.token_ids, self.is_attendeds, self.is_labels))

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            copy = cp.copy(self)
            copy.token_ids = self.token_ids[index]
            copy.is_attendeds = self.is_attendeds[index]
            copy.is_labels = self.is_labels[index]
            return copy
        else:
            return self.token_ids[index]

    def __setitem__(self, index, value):
        self.token_ids[index] = value

    def __str__(self):
        if len(self) > 10:
            return f'<TokenSequence len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) for t in self[:10])}|...>' # noqa
        else:
            return f'<TokenSequence len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) for t in self)}>' # noqa

    def __repr__(self):
        return f"TokenSequence({repr(self.token_ids)})"


@dc.dataclass
class Token:
    text: str
    token_id: int
    is_attended: bool = True
    is_label: bool = False

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Token({self.text}, {self.token_id}, is_attended={self.is_attended}, is_label={self.is_label})"
