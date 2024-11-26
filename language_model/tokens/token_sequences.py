
from __future__ import annotations

from language_model.tokens.token_sequence import TokenSequence
from language_model.tokens.tokenizer import Tokenizer

import typing as T


class TokenSequences:
    tokenizer = None

    def __init__(self,
        sequences: T.Iterable[str|TokenSequence|T.Iterable[tuple[int, bool, bool]]],
        pad_to_same_length: bool = True,
        pad_to_multiple_of: int = 8,
        pad_side: str = 'L',
        tokenizer: Tokenizer = None,
    ):
        self.sequences: list[TokenSequence] = []
        self.tokenizer = type(self).tokenizer if tokenizer is None else tokenizer
        assert self.tokenizer is not None, "A tokenizer must be provided to TokenSequences."
        self.pad_to_same_length = pad_to_same_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_side = pad_side
        self.add(sequences)

    def add(self,
        sequences: T.Iterable[str|TokenSequence|T.Iterable[tuple[int, bool, bool]]],
        is_attended: bool = True,
        is_label: bool = False
    ):
        """Add sequences to the list."""
        to_add = []
        for sequence in sequences:
            if isinstance(sequence, TokenSequence):
                to_add.append(sequence)
            else:
                to_add.append(TokenSequence(
                    sequence, is_attended=is_attended, is_label=is_label, tokenizer=self.tokenizer))
        if self.pad_to_same_length:
            prev_max_len = max(len(seq) for seq in self.sequences) if self.sequences else 0
            max_len = max(max(len(seq) for seq in to_add), prev_max_len)
            if max_len > prev_max_len:
                self.sequences.extend(to_add) # noqa
                self.pad(max_len)
            else:
                to_add = TokenSequences(to_add, tokenizer=self.tokenizer, pad_to_same_length=False,
                    pad_to_multiple_of=self.pad_to_multiple_of, pad_side=self.pad_side)
                to_add.pad(max_len)
                self.sequences.extend(to_add) # noqa
        else:
            self.sequences.extend(to_add)
        return self

    def pad(self, max_length: int = None):
        """Pads all token sequences to the same length, if required."""
        if self.pad_to_same_length and self.sequences:
            if max_length is None:
                max_length = max(len(seq) for seq in self.sequences)
            if self.pad_to_multiple_of:
                max_length = ((max_length + self.pad_to_multiple_of - 1) //
                              self.pad_to_multiple_of * self.pad_to_multiple_of)
            for seq in self.sequences:
                padding_length = max_length - len(seq)
                if padding_length > 0:
                    padding = [self.tokenizer.pad_token_id] * padding_length
                    if self.pad_side == 'L':
                        seq.token_ids = padding + seq.token_ids
                        seq.is_attendeds = [False] * padding_length + seq.is_attendeds
                        seq.is_labels = [False] * padding_length + seq.is_labels
                    else:
                        seq.token_ids += padding
                        seq.is_attendeds += [False] * padding_length
                        seq.is_labels += [False] * padding_length

    def dict(self, seq_type: type|callable = list, with_labels=True):
        """Returns the input_ids, attention_mask, and labels for all sequences."""
        if with_labels:
            return dict(
                input_ids=seq_type([seq.token_ids for seq in self.sequences]),
                attention_mask=seq_type([seq.is_attendeds for seq in self.sequences]),
                labels=seq_type([seq.is_labels for seq in self.sequences]),)
        else:
            return dict(
                input_ids=seq_type([seq.token_ids for seq in self.sequences]),
                attention_mask=seq_type([seq.is_attendeds for seq in self.sequences]),)

    def __iter__(self):
        return iter(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

    def __setitem__(self, index, value):
        if isinstance(value, TokenSequence):
            self.sequences[index] = value
        else:
            new_sequence = TokenSequence(value, tokenizer=self.tokenizer)
            self.sequences[index] = new_sequence

    def __str__(self):
        if len(self) > 3:
            return f'<TokenSequences len {len(self)}: {self[0]};  {self[1]};  ...;  {self[-1]}>'
        else:
            return f'<TokenSequences len {len(self)}: {";  ".join(str(seq) for seq in self)}>'

    def __repr__(self):
        return f"TokenSequences({repr(self.sequences)})"