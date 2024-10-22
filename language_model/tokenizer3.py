from __future__ import annotations

import dataclasses as dc
import inspect
import re
import collections as coll
import itertools as it
import functools as ft
import textwrap as tw
import abc
import copy as cp
from language_model.utils.config import config, Config
from language_model.utils.peek import peek
from language_model.utils import ansi

# black magic type hinting of base as dataclass
from dataclasses import dataclass; vars().update(dataclass=config) # noqa

from transformers import PreTrainedTokenizer, AutoTokenizer

import typing as T


default: T.Any = object()


class TokenSequence:
    tokenizer = None

    def __init__(self,
        *sequence: str|T.Iterable[tuple[int, bool, bool]]|T.Iterable[int],
        is_attended: bool = True,
        is_label:bool = False,
        tokenizer: PreTrainedTokenizer = None,
    ):
        self.sequence = []
        self.tokenizer = type(self).tokenizer if tokenizer is None else tokenizer
        assert self.tokenizer is not None, "A tokenizer must be provided to TokenSequence."
        for sequence in sequence:
            self.extend(sequence, is_attended=is_attended, is_label=is_label)

    def extend(self,
        sequence: str|T.Iterable[tuple[int, bool, bool]]|T.Iterable[int],
        is_attended: bool = True,
        is_label:bool = False,
    ):
        if isinstance(sequence, str):
            token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
            self.sequence.extend((token_id, is_attended, is_label) for token_id in token_ids)
        else:
            first, sequence = peek(sequence)
            if isinstance(first, int):
                self.sequence.extend((token_id, is_attended, is_label) for token_id in sequence)
            else:
                self.sequence.extend(sequence)
        return self
    __iadd__ = extend

    def text(self):
        return self.tokenizer.decode([t[0] for t in self], clean_up_tokenization_spaces=True)

    def dict(self, seq_type: type|callable = list):
        return dict(
            input_ids=seq_type([t[0] for t in self]),
            attention_mask=seq_type([t[1] for t in self]),
            labels=seq_type([t[0] if t[2] else -100 for t in self]))

    def tokens(self, strip=False):
        tokens = [self.tokenizer.decode(t[0], clean_up_tokenization_spaces=strip) for t in self]
        if strip:
            stripped = [t.strip() for t in tokens]
            return [x or y for x, y in zip(stripped, tokens)]
        else:
            return tokens

    def ansi(self):
        return TokenPrinter().ansi(self)

    def display(self):
        return f"{ansi.bold}{self.__class__.__name__} with {len(self)} tokens:{ansi.reset}\n{self.ansi()}"

    def __add__(self, other):
        copy = cp.copy(self)
        copy.sequence = copy.sequence + other.sequence
        return copy

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        if isinstance(index, slice):
            copy = cp.copy(self)
            copy.sequence = self.sequence[index]
            return copy
        else:
            return self.sequence[index]

    def __setitem__(self, index, value):
        self.sequence[index] = value

    def __str__(self):
        if len(self) > 10:
            return f'<TokenSequence len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) for t in self[:10])}|...>' # noqa
        else:
            return f'<TokenSequence len {len(self)}: {"|".join(self.tokenizer.decode(t[0]) for t in self)}>' # noqa

    def __repr__(self):
        return f"TokenSequence({repr(self.sequence)})"


class TokenSequences:
    tokenizer = None

    def __init__(self,
        sequences: T.Iterable[str|TokenSequence|T.Iterable[tuple[int, bool, bool]]],
        pad_to_same_length: bool = True,
        pad_to_multiple_of: int = 8,
        pad_side: str = 'L',
        tokenizer: PreTrainedTokenizer = None,
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
            self.sequences.extend(to_add)
            self.pad(max_len)
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
                    padding = [(self.tokenizer.pad_token_id, False, False)] * padding_length
                    if self.pad_side == 'L':
                        seq.sequence = padding + seq.sequence
                    else:
                        seq.sequence += padding

    def text(self) -> list[str]:
        """Returns the decoded text for all sequences."""
        return [seq.text() for seq in self.sequences]

    def ansi(self):
        return TokenPrinter().ansi(self)

    def display(self):
        return f"{ansi.bold}{self.__class__.__name__} with {len(self)} sequences:{ansi.reset}\n{self.ansi()}"

    def dict(self, seq_type: type|callable = list):
        """Returns the input_ids, attention_mask, and labels for all sequences."""
        return dict(
            input_ids=seq_type([seq.dict(seq_type)["input_ids"] for seq in self.sequences]),
            attention_mask=seq_type([seq.dict(seq_type)["attention_mask"] for seq in self.sequences]),
            labels=seq_type([seq.dict(seq_type)["labels"] for seq in self.sequences]),)

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


EOS = object()

@dc.dataclass
class TokenSlot:
    name: str = 'text'
    is_label: bool = False
    max: int = None
    min: int = 0
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str|TokenSequence = '...'
    min_out: int = 0
    prefix: str|TokenSequence = ''
    suffix: str|TokenSequence = ''

    def __post_init__(self):
        self.index = None

Slot = str | TokenSlot

@dc.dataclass
class InputSlot(TokenSlot):
    name: str = 'input'
    is_label: bool = False
    max: int = None
    min: int = 0
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str|TokenSequence = '...'
    min_out: int = 0
    prefix: str|TokenSequence = ''
    suffix: str|TokenSequence = ''

@dc.dataclass
class OutputSlot(TokenSlot):
    name: str = 'output'
    is_label: bool = True
    max: int = None
    min: int = 0
    trunc_side: str = 'R'
    trunc_rank: float = 1.0
    trunc_text: str|TokenSequence = ''
    min_out: int = 0
    prefix: str|TokenSequence = ''
    suffix: str|TokenSequence = '{eos}'


class TemplateMeta(type):
    template: str = None
    slots: dict[str, TokenSlot]
    def __new__(typ, name, bases, attrs):
        cls = super().__new__(typ, name, bases, attrs)
        setattr(cls, 'slots', {})
        for base in bases:
            if getattr(base, 'slots', None):
                cls.slots.update(base.slots)
        if Template in bases:
            assert isinstance(attrs.get('template'), str), \
                f"Class {name} must define a class attribute 'template' with a template string."
        for slot_name, value in attrs.items():
            if isinstance(value, TokenSlot):
                assert '{'+slot_name+'}' in cls.template, \
                    f"Slot {slot_name} was defined as a class field of {name} but not in template text:  {cls.template}"
                value.name = name
                cls.slots[name] = value
        return cls

class Template(metaclass=TemplateMeta):
    template: str
    slots: dict[str, TokenSlot]


class Foo(Template):
    template = 'hello'

foo = Foo()

TC = T.TypeVar('TC', bound=Template)

@dc.dataclass
class TokenTemplate(T.Generic[TC]):
    tokenizer = None
    template: TC
    """A custom dataclass object with a class attribute 'template' that defines a template string, and TokenSlot objects as fields for each slot in the template::
    
    @dataclass
    class MyTemplate(Template):
        template = 'This is a {adjective} {noun}.'
        adjective: Slot = InputSlot() 
        noun: Slot = InputSlot()
    """
    is_attended: bool = True
    is_label: bool = False
    trunc_content: bool = True
    trunc_segment: bool = False
    trunc_segment_side: str = 'L'
    max_length: int = None
    pad: bool = True
    pad_to_same_length: bool = True
    pad_to_multiple_of: int = 1
    pad_side: str = 'L'

    def __post_init__(self):
        if hasattr(self, 'slots') and hasattr(self, 'tokens'): return
        self.slots: list[TokenSlot] = []
        template_text = self.template.template
        slot_pattern = re.compile(r"(?P<leading_space> ?)\{(?P<slot_name>[a-zA-Z_][a-zA-Z_0-9]*})")
        previous_end = 0
        template_parts = []
        slots = []
        for slot_match in slot_pattern.finditer(template_text):
            slot_name = slot_match.group('slot_name')
            slot_leading_space = slot_match.group('leading_space')
            if slot_name in self.template.slots:
                slot = self.template.slots[slot_name]
                start, end = slot_match.span()
                template_parts.append(template_text[previous_end:start])
                slot_clone = cp.copy(slot)
                slot_clone.prefix = slot_leading_space + slot.prefix
                slots.append(slot_clone)
                previous_end = end
        template_suffix = template_text[previous_end:]
        self.tokens = TokenSequence(
            '', is_attended=self.is_attended, is_label=self.is_label, tokenizer=self.tokenizer)
        for template_part, slot in zip(template_parts, slots):
            self.tokens += template_part
            slot_clone = cp.copy(slot)
            slot_clone.index = len(self.tokens)
            for name, value in self.tokenizer.slot_affix_replacements.items():
                if value is None: continue
                if isinstance(slot_clone.prefix, str):
                    slot_clone.prefix = slot_clone.prefix.replace(name, value)
                if isinstance(slot_clone.suffix, str):
                    slot_clone.suffix = slot_clone.suffix.replace(name, value)
                if isinstance(slot_clone.trunc_text, str):
                    slot_clone.trunc_text = slot_clone.trunc_text.replace(name, value)
            slot_clone.suffix = TokenSequence(slot_clone.suffix,
                is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer)
            slot_clone.trunc_text = TokenSequence(slot_clone.trunc_text,
                is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer)
            self.slots.append(slot_clone)
        self.tokens += template_suffix

    def __call__(self, *args, **kwargs) -> TC:
        return self.template.__class__(*args, **kwargs)

    def fill(self, values: TC|T.Iterable[TC]) -> TokenSequence|TokenSequences:
        if not isinstance(values, self.template.__class__):
            assert hasattr(values, '__iter__'), \
                f"Values must be an iterable of {self.template.__class__.__name__}"
            for value in values:
                assert isinstance(value, self.template.__class__), \
                    f"Values must be an iterable of {self.template.__class__.__name__}"
            return TokenSequences([self.fill(value) for value in values])
        len_template = len(self.tokens)
        raw_value_texts = {}
        raw_value_seqs = {}
        for slot in self.slots:
            if slot.name in raw_value_texts: continue
            value_text = getattr(values, slot.name)
            prefix = slot.prefix

        return ...


@dc.dataclass
class TokenTemplates:
    tokenizer = None
    max_length: int = None
    pad_to_same_length: bool = True
    pad_to_multiple_of: int = 8
    pad_side: str = 'L'
    max_segments: int | None = None

    def __post_init__(self):
        template_fields = {field.name for field in dc.fields(self)} - token_templates_base_fields
        if not template_fields:
            raise ValueError(f"No templates were defined in {type(self).__qualname__}. Initialize a subclass with at least one TokenTemplate.")
        for field in dc.fields(self):
            if field.name in template_fields:
                annotation = field.type
                assert getattr(annotation, '__origin__', None) == TokenTemplate, \
                    f"Field {field.name} in {type(self).__qualname__} must be of type TokenTemplate."
                assert (args:=getattr(annotation, '__args__', None)) and isinstance(args[0], Template), \
                    f"Field {field.name} in {type(self).__qualname__} must be a TokenTemplate with a type argument that is an instance of a Template dataclass (defining a 'template' string with TokenSlot fields)."
                assert field.default is args[0], \
                    f"Field {field.name} in {type(self).__qualname__} must have a default value that is an instance of the TokenTemplate type argument, like `my_template: TokenTemplate[X] = X`"
                template = TokenTemplate(field.default())
                template.tokenizer = self.tokenizer
                setattr(self, field.name, template)

    def fill(self, values: T.Iterable[Template]) -> TokenSequence:
        ...

    def fill_batch(self, values: T.Iterable[T.Iterable[Template]]) -> TokenSequences:
        ...

    def batch(self, values: T.Iterable[T.Iterable[Template]], batch_size: int) -> list[TokenSequences]:
        ...

token_templates_base_fields = {field.name for field in dc.fields(TokenTemplates)}


@dc.dataclass
class TokenPrinter:
    tokens: T.Iterable = None
    token_colors:tuple = ((55, 45, 120), (30, 70, 130), (20, 90, 110))
    foreground_color: tuple = (200, 200, 200)
    padding_color:tuple = ('black',)
    label_color:tuple = (255, 255, 255)
    label_style: str|None = ansi.bold
    slot_color:tuple = (80, 60, 30)

    def __post_init__(self):
        if self.tokens is not None:
            self.print(self.tokens)

    def ansi(self, tokens: TokenSequence|TokenSequences):
        if isinstance(tokens, TokenSequences):
            return '\n\n'.join(self.ansi(seq) for seq in tokens)
        display = []
        token_color_iter =iter(it.cycle(self.token_colors))
        token_texts = tokens.tokens()
        token_types = [type(token) for token in tokens]
        for token, token_text, token_type in zip(tokens, token_texts, token_types):
            styles = []
            if token_type is tuple:
                token_id, is_attended, is_label = token
                token_color = ansi.color(*next(token_color_iter)).bg
                if not is_attended:
                    padding_color = ansi.color(*self.padding_color).fg
                    styles.append(padding_color)
            elif issubclass(token_type, TokenSlot):
                is_label = token.is_label
                token_color = ansi.color(*self.slot_color).bg
            else:
                raise ValueError(f"Token type {token_type} for displaying token {token} is not recognized.")
            styles.append(token_color)
            if is_label:
                styles.extend((ansi.color(*self.label_color).fg, self.label_style))
            ansi_style = ''.join(styles)
            token_display = f'{ansi_style}{token_text}{ansi.reset}'.replace(
                '\n', f'↵{ansi.reset}\n{ansi_style}')
            display.append(token_display)
        return ''.join(display)

    def print(self, tokens):
        print(self.ansi(tokens))


@dc.dataclass
class Foo:
    template = 'describe {x}: {y}'
    x: Slot
    y: Slot

@dc.dataclass
class MyTemplates(TokenTemplates):
    system_prompt: TokenTemplate[Foo] = Foo


templates = MyTemplates(
    system_prompt=TokenTemplate(template=Foo(x=InputSlot(), y=OutputSlot(trunc_text='.....')),
        is_label=False, trunc_content=False)
)
