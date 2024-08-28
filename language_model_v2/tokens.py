
from __future__ import annotations

import dataclasses as dc
import re
import collections as coll
import itertools as it
import functools as ft
from language_model_v2.utils.config import config, Config
from language_model_v2.utils import ansi

# black magic type hinting of config as dataclass
from dataclasses import dataclass; vars().update(dataclass=config)

from transformers import PreTrainedTokenizer, AutoTokenizer

import typing as T


def _imports(): pass


default: T.Any = object()


class Tokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizer|str):
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if tokenizer.pad_token_id is None:
            pad_token = '-'
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            tokenizer.pad_token = pad_token
            tokenizer.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
            
    def templatize(self, 
        *sequence: str | 'TokenTemplate' | 'TokSlot' | T.Iterable[tuple[int, bool, bool] | 'TokSlot'],
        is_attended: bool = default,
        is_label: bool = default,
    ) -> 'TokenTemplate':
        return TokenTemplate(*sequence, is_attended=is_attended, is_label=is_label, tokenizer=self.tokenizer)
    
    def tokenize(self, 
        *sequences: str | T.Iterable[tuple[int, bool, bool]] | T.Iterable[str | T.Iterable[tuple[int, bool, bool]]],
        is_attended: bool = default,
        is_label: bool = default,
    ) -> 'TokenSequence' | list['TokenSequence']:
        for sequence in sequences:
            if not hasattr(sequence, '__len__'):
                sequence = list(sequence)
            if sequence and (
                isinstance(sequence[0], str) or
                isinstance(sequence[0], tuple) and len(sequence[0]) == 3 and isinstance(sequence[0][0], int)
            ):
                return TokenSequence(*sequences, is_attended=is_attended, is_label=is_label, tokenizer=self.tokenizer)
        else:
            return [TokenSequence(seq, is_attended=is_attended, is_label=is_label, tokenizer=self.tokenizer) 
                for seqs in sequences for seq in seqs]
        
        
class _DisplaySettings:
    _display_width = 80
    _display_token_colors = ((25, 20, 65), (0, 30, 60), (0, 40, 50))
    _display_padding_color = ('black',)
    _display_foreground_color = (200, 200, 200)
    _display_label_color = (255, 255, 255)
    _display_label_style = ansi.bold
    _display_slot_color = (80, 60, 30)

def display_tokens(seq: TokenSequence | TokenTemplate):
    num_slots = len(seq.slots) if hasattr(seq, 'slots') else 0
    num_tokens = len(seq) - num_slots
    if num_slots > 0:
        print(f"{seq.__class__.__name__} with {num_tokens} tokens and {num_slots} slots:")
    else:
        print(f"{seq.__class__.__name__} with {num_tokens} tokens:")
    display_tokens = []
    for token, token_background_color in zip(seq, it.cycle(seq._display_token_colors)):
        if isinstance(token, TokSlot):
            token_text = f'#[{token.name}]#'
            token = (-1, True, bool(isinstance(token, TokOut)))
            token_text = f"{ansi.color(*seq._display_slot_color).bg}{token_text}{ansi.reset}"
        else:
            token_text = seq.tokenizer.decode(token[0])
            newlinestripped = token_text.rstrip('\n')
            num_newlines = len(token_text) - len(newlinestripped)
            if num_newlines > 0:
                token_text = ''.join((newlinestripped, "\\n" * num_newlines, ansi.reset, '\n'*num_newlines))
        display_tokens.append(ansi.color(*token_background_color).bg)
        if token[2]:
            display_tokens.append(ansi.color(*seq._display_label_color).fg)
            display_tokens.append(seq._display_label_style)
        elif token[1]:
            display_tokens.append(ansi.color(*seq._display_foreground_color).fg)
        else:
            display_tokens.append(ansi.color(*seq._display_padding_color).fg)
        display_tokens.append(token_text)
        display_tokens.append(ansi.reset)
    print(''.join(display_tokens), end='\n\n')


class TokenSequence(_DisplaySettings, list):

    def __init__(
        self,
        *sequence: str | T.Iterable[tuple[int, bool, bool]],
        is_attended: bool = default,
        is_label: bool = default,
        tokenizer: PreTrainedTokenizer = None,
    ):
        self.tokenizer = tokenizer
        list.__init__(self)
        if is_attended is default: is_attended = True
        if is_label is default: is_label = False
        for sequence in sequence:
            if isinstance(sequence, str):
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                list.extend(self, ((token_id, is_attended, is_label) for token_id in token_ids))
            else:
                list.extend(self, sequence)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(list.__getitem__(self, item))
        else:
            return list.__getitem__(self, item)

    def text(self):
        return self.tokenizer.decode([t[0] for t in self])

    def dict(self, seq_type:type=list):
        return dict(
            input_ids=seq_type([t[0] for t in self]),
            attention_mask=seq_type([t[1] for t in self]),
            labels=seq_type([t[0] if t[2] else -100 for t in self]))

    def __str__(self):
        if len(self) > 10:
            return f'<TokenSequence len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) if isinstance(t, tuple) else t.as_text() for t in self[:10])}|...>' # noqa
        else:
            return f'<TokenSequence len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) if isinstance(t, tuple) else t.as_text() for t in self)}>' # noqa

    def __repr__(self):
        return list.__repr__(self)

    def display(self):
        return display_tokens(self)


class TokenSequenceBatch(list):

    def __init__(self,
        *sequences: T.Iterable[TokenSequence|str],
        tokenizer: PreTrainedTokenizer = None,
        pad_to_same_length: bool = True,
    ):
        self.tokenizer = tokenizer
        list.__init__(self)
        seqs = []
        for sequence in sequences:
            if isinstance(sequence, TokenSequence):
                seqs.append(sequence)
            elif isinstance(sequence, str):
                seqs.append(TokenSequence(sequence, tokenizer=self.tokenizer))
            else:
                seqs.extend(seq if isinstance(seq, TokenSequence) else TokenSequence(seq, tokenizer=self.tokenizer) 
                    for seq in sequence)
        if pad_to_same_length:
            max_len = max(len(seq) for seq in seqs)
            for seq in seqs:
                if len(seq) < max_len:
                    pad = [(self.tokenizer.pad_token_id, False, False)]
                    padding = pad * (max_len - len(seq))
                    seq = TokenSequence(padding + seq, tokenizer=self.tokenizer)
                list.append(self, seq)
        else:
            list.extend(self, seqs)

    def append(self, __object):
        raise NotImplementedError("TokenSequenceBatch should not be added to after construction.")

    def extend(self, __iterable):
        raise NotImplementedError("TokenSequenceBatch should not be added to after construction.")

    def __setitem__(self, key, value):
        raise NotImplementedError("TokenSequenceBatch should not be added to after construction.")

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(list.__getitem__(self, item))
        else:
            return list.__getitem__(self, item)

    def dict(self, seq_type:type=list):
        return dict(
            input_ids=seq_type([[t[0] for t in s] for s in self]),
            attention_mask=seq_type([[t[1] for t in s] for s in self]),
            labels=seq_type([[t[0] if t[2] else -100 for t in s] for s in self]))

    def display(self):
        for seq in self:
            seq.display()


class TokenTemplate(_DisplaySettings, list):
    _slot_pattern = re.compile(r"#\[([a-z_A-Z0-9]+=[^,\]]*(?:, ?[a-z_A-Z0-9]+=[^,\]]*)*)]#")

    def __init__(self,
        *sequence: str | 'TokenTemplate' | 'TokSlot' | T.Iterable[tuple[int, bool, bool] | TokSlot],
        is_attended: bool = default,
        is_label: bool = default,
        tokenizer: PreTrainedTokenizer = None,
    ):
        self.tokenizer = tokenizer
        list.__init__(self)
        """Tokens as (id, str, is_attended, is_label) tuples. Input/OutputSequence objects represent slots to fill in the sequence with input/output text."""
        self.slots: dict[str, TokSlot] = {}
        for sequence in sequence:
            if isinstance(sequence, TokenTemplate):
                list.extend(self, sequence)
                self.slots.update(sequence.slots)
            elif isinstance(sequence, str):
                while sequence:
                    match = self._slot_pattern.search(sequence)
                    if match is None:
                        prefix, sequence = sequence, ''
                    else:
                        prefix = sequence[:match.start()]
                        sequence = sequence[match.end():]
                    if prefix:
                        token_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                        list.extend(self, (
                            (
                                token_id,
                                True if is_attended is default else is_attended,
                                False if is_label is default else is_label
                            )
                            for token_id in token_ids))
                    if match:
                        arguments = dict(
                            [x.strip() for x in argument.split('=', 1)]
                            for argument in match.group(1).split(','))
                        if 'input' in arguments:
                            arguments['name'] = arguments.pop('input') # noqa
                            slot = TokIn(**arguments, index=len(self))
                        elif 'output' in arguments:
                            arguments['name'] = arguments.pop('output')
                            slot = TokOut(**arguments, index=len(self))
                        else:
                            raise ValueError(f"Slot be named using input= or output=, but got {arguments}")
                        assert slot.name not in self.slots, f"Duplicate slot name {slot.name} detected when constructing TokenSequence."
                        self.slots[slot.name] = slot
                        list.append(self, slot)
            elif isinstance(sequence, TokSlot):
                list.append(self, sequence)
            else:
                list.extend(self, sequence)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(list.__getitem__(self, item))
        else:
            return list.__getitem__(self, item)

    def fill(self,
        slots: dict[str, str | 'TokenSequence'] | T.Iterable[dict[str, str | 'TokenSequence']],
        max_length: int = None,
        min_length: int = None,
        pad_to_same_length: bool = True,
    ) -> T.Union['TokenSequence', 'TokenSequenceBatch']:
        if isinstance(slots, dict):
            return self._fill_single(slots, max_length, min_length)
        else:
            return self._fill_batch(slots, max_length, min_length, pad_to_same_length)

    def _fill_single(self,
        slots: dict[str, str | 'TokenSequence'],
        max_length: int = None,
        min_length: int = None,
    ):
        assert all(slot in self.slots for slot in slots), \
            f"Slots {set(slots) - set(self.slots)} not found in TokenSequence."
        slot_subseqs = []
        filled = []
        previous_splitter = 0
        for slot_name, text in slots.items():
            slot = self.slots[slot_name]
            if isinstance(text, str):
                text = TokenSequence(text, is_label=slot.is_label, tokenizer=self.tokenizer)
            if slot.eos is None:
                eos = ((self.tokenizer.eos_token_id, True, slot.is_label),)
            elif slot.eos:
                eos = TokenSequence(slot.eos, is_label=slot.is_label, tokenizer=self.tokenizer)
            else:
                eos = ()
            text.extend(eos)
            slot_subseqs.append([slot, text])
        for i, slot_subseq in enumerate(slot_subseqs):
            slot, subseq = slot_subseq
            if slot.max and len(subseq) > slot.max:
                if slot.trunc_side == 'L':
                    slot_subseqs[i][1] = subseq[-slot.max:]
                else:
                    slot_subseqs[i][1] = subseq[:slot.max]
        template_length = len(self) - len(self.slots)
        current_length = sum(len(value) for _, value in slot_subseqs) + template_length
        if max_length is not None and current_length > max_length:
            slot_trunc_candidates = iter(sorted(enumerate(slot_subseqs), key=lambda x: x[1][0].trunc_rank))
            for slot_subseq in slot_trunc_candidates:
                i, (slot, subseq) = slot_subseq
                amount_to_truncate = max(min(current_length - max_length, len(subseq) - slot.min, len(subseq)), 0)
                if amount_to_truncate:
                    if slot.trunc_side == 'L':
                        slot_subseqs[i][1] = subseq[amount_to_truncate:]
                    else:
                        slot_subseqs[i][1] = subseq[:-amount_to_truncate]
                    current_length = sum(len(value) for _, value in slot_subseqs) + template_length
                    if current_length <= max_length:
                        break
            else: # nobreak
                raise ValueError(f"Could not truncate slot text to fit within max_length {max_length} (max truncation was reached after sequence was cut down to {current_length} tokens).")
        for slot, value in slot_subseqs:
            splitter = slot.index
            prefix = self[previous_splitter:splitter]
            filled.extend(prefix)
            filled.extend(value)
            previous_splitter = splitter + 1
        filled.extend(self[previous_splitter:])
        if min_length is not None and len(filled) < min_length:
            pad_length = min_length - len(filled)
            padding = [(self.tokenizer.pad_token_id, False, False)] * pad_length
            filled = padding + filled
        return TokenSequence(filled, tokenizer=self.tokenizer)

    def _fill_batch(self,
        slots: T.Iterable[dict[str, str | 'TokenSequence']],
        max_length: int = None,
        min_length: int = None,
        pad_to_same_length: bool = True,
    ):
        return TokenSequenceBatch(
            [self._fill_single(slots_, max_length, min_length) for slots_ in slots],
            tokenizer=self.tokenizer,
            pad_to_same_length=pad_to_same_length)

    def text(self):
        return ''.join(self.tokenizer.decode(t[0]) if isinstance(t, tuple) else t.as_text() for t in self)

    def __str__(self):
        if len(self) > 10:
            return f'<TokenTemplate len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) if isinstance(t, tuple) else t.as_text() for t in self[:10])}|...>'
        else:
            return f'<TokenTemplate len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) if isinstance(t, tuple) else t.as_text() for t in self)}>'

    def __repr__(self):
        return str(self)

    def display(self):
        return display_tokens(self)


def _tokenclasses(): pass


@dataclass
class TokSlot(Config):
    name: str
    max: int = None
    min: int = 0
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    index: int = None
    is_label: bool = False
    eos: str | None = ''

    def __post_init__(self):
        self.max = None if self.max in (None, 'None') else int(self.max)
        self.min = 0 if self.min in (None, 'None') else int(self.min)
        trunc_side = self.trunc_side.upper()
        if 'LEFT'.startswith(trunc_side):
            self.trunc_side = 'L'
        elif 'RIGHT'.startswith(trunc_side):
            self.trunc_side = 'R'
        else:
            raise ValueError(f"trunc_side must be a prefix of 'LEFT' or 'RIGHT', not {trunc_side}")
        self.trunc_rank = float(self.trunc_rank)
        self.is_label = (self.is_label and isinstance(self.is_label, bool)
                         or not 'false'.startswith(str(self.is_label).lower()))
        self.eos = None if self.eos in (None, 'None') else self.eos

    def as_text(self):
        return f"#[{self.name}]#"

    def __str__(self):
        if isinstance(self, TokIn):
            kind = 'input'
        elif isinstance(self, TokOut):
            kind = 'output'
        else:
            kind = 'name'
        fields = ''.join(f", {k}={v}" for k, v in self.__dict__.items() if k not in ('name', 'index'))
        return f"#[{kind}={self.name}{fields}]#"

    def __repr__(self):
        return str(self)


@dataclass
class TokIn(TokSlot):
    name: str = 'input'

@dataclass
class TokOut(TokSlot):
    name: str = 'output'
    trunc_side: str = 'R'
    trunc_rank: float = 0.0
    is_label: bool = True
    eos: str | None = None



def main():
    import textwrap as tw
    template = tw.dedent(f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    {TokIn('my_input')}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {TokOut('my_output')}
    """).strip()

    print(template, '\n\n')

    data = [
        dict(my_input="What is the capital of France?"*100, my_output="The capital of France is Paris."),
        dict(my_input="What is the capital of the United States of America (USA)?", my_output="The capital of the United States of America is Washington, D.C."*100)
    ]

    import time
    import contextlib as cl

    @cl.contextmanager
    def timer(label):
        t1 = time.perf_counter()
        yield
        t2 = time.perf_counter()
        print(f"{label} took {t2 - t1:.3f} seconds.\n")

    import torch as pt
    llama_tokenizer = Tokenizer('meta-llama/Meta-Llama-3.1-8B-Instruct')
    template_sequence = llama_tokenizer.templatize(template)
    template_sequence.display()
    batch_size = 128
    data = data * (batch_size // len(data))
    num_batches = 100
    with timer("Filling token sequences"):
        for _ in range(num_batches):
            filled_sequence = template_sequence.fill(data, max_length=1024)
            input_to_llm = filled_sequence.dict(seq_type=pt.LongTensor)
    filled_sequence[:2].display()
    return


if __name__ == '__main__':
    import cProfile
    # cProfile.run('main()', sort='cumtime')
    main()





































