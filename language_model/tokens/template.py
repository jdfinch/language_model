
from __future__ import annotations

import dataclasses as dc
import functools as ft
import copy as cp
import re

import ezpyzy as ez
from dataclasses import dataclass; vars().update(dataclass=ez.config) # noqa, black magic type hinting

from language_model.tokens.token_sequence import TokenSequence
from language_model.tokens.token_sequences import TokenSequences
from language_model.tokens.tokenizer import Tokenizer

import typing as T


def fields(cls_or_instance) -> list[dc.Field]: return dc.fields(cls_or_instance) # noqa


@dataclass
class TokenSlot(ez.Config):
    name: str = 'text'
    is_label: bool = False
    max: int = None
    min: int = 0
    truncatable: bool = True
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str|TokenSequence = '...'
    min_out: int = 0
    prefix: str|TokenSequence = ''
    suffix: str|TokenSequence = ''
    index: int = 0


Slot = str | TokenSlot

@dataclass
class InputSlot(TokenSlot):
    name: str = 'input'
    is_label: bool = False
    max: int = None
    min: int = 0
    truncatable: bool = True
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str|TokenSequence = '...'
    min_out: int = 0
    prefix: str|TokenSequence = ''
    suffix: str|TokenSequence = ''

def Input(
    name: str = 'input',
    is_label: bool = False,
    max: int = None,
    min: int = 0,
    truncatable: bool = True,
    trunc_side: str = 'L',
    trunc_rank: float = 1.0,
    trunc_text: str|TokenSequence = '...',
    min_out: int = 0,
    prefix: str|TokenSequence = '',
    suffix: str|TokenSequence = '',
) -> str|InputSlot:
    return ...
vars().update(Input=InputSlot)

@dataclass
class OutputSlot(TokenSlot):
    name: str = 'output'
    is_label: bool = True
    max: int = None
    min: int = 0
    truncatable: bool = True
    trunc_side: str = 'R'
    trunc_rank: float = 1.0
    trunc_text: str|TokenSequence = ''
    min_out: int = 0
    prefix: str|TokenSequence = ''
    suffix: str|TokenSequence = '{eos}'

def Output(
    name: str = 'output',
    is_label: bool = True,
    max: int = None,
    min: int = 0,
    truncatable: bool = True,
    trunc_side: str = 'R',
    trunc_rank: float = 1.0,
    trunc_text: str|TokenSequence = '',
    min_out: int = 0,
    prefix: str|TokenSequence = '',
    suffix: str|TokenSequence = '{eos}',
) -> str|OutputSlot:
    return ...
vars().update(Output=OutputSlot)


class TemplateMeta(type):
    template: str = None
    __template_slots__: dict[str, TokenSlot]
    def __new__(typ, name, bases, attrs):
        cls = super().__new__(typ, name, bases, attrs)
        setattr(cls, '__template_slots__', {})
        for base in bases:
            if getattr(base, '__template_slots__', None):
                cls.__template_slots__.update(base.__template_slots__)
        if len(bases) > 1:
            assert isinstance(attrs.get('template'), str), \
                f"Class {name} must define a class attribute 'template' with a template string."
        cls.template = str(cls.template)
        for slot_name, value in attrs.items():
            if isinstance(value, TokenSlot):
                assert '<'+slot_name+'>' in cls.template, \
                    f"Slot {slot_name} was defined as a class field of {name} but not in template text:  {cls.template}"
                value = TokenSlot(value, name=slot_name)
                cls.__template_slots__[slot_name] = value
                setattr(cls, slot_name, dc.field(default_factory=ft.partial(cp.copy, value)))
        return cls

    def __iter__(self) -> T.Iterator[TokenSlot]:
        return iter(self.__template_slots__.values())

class Template(metaclass=TemplateMeta):
    template: str
    __template_slots__: dict[str, TokenSlot]

    def __iter__(self):
        return iter(self.__template_slots__)

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self):
        text = self.template
        for slot_name, slot in self.__template_slots__.items():
            value = getattr(self, slot_name)
            if isinstance(value, TokenSlot):
                value = f"<{value.name}>"
            text = text.replace(f"<{slot_name}>", value)
        return text


TT = T.TypeVar('TT', bound=Template)


@dataclass
class TemplateConfig(ez.Config, T.Generic[TT]):
    template: TT = None
    """A custom dataclass object with a class attribute 'template' that defines a template string, and TokenSlot objects as fields for each slot in the template::

    @dataclass
    class MyTemplate(Template):
        template = 'This is a <adjective> <noun>.'
        adjective: Slot = InputSlot() 
        noun: Slot = InputSlot()
    """
    is_attended: bool = True
    is_label: bool = False
    trunc_content: bool = True
    trunc_segment: bool = False
    trunc_segment_with_no_content: bool = True
    trunc_segment_rank: float = 1.0
    trunc_segment_side: str = 'L'
    max_length: int = None
    pad: bool = True
    pad_to_same_length: bool = True
    pad_to_multiple_of: int = 1
    pad_side: str = 'L'
    tokenizer: Tokenizer = None

    def __post_init__(self):
        assert isinstance(self.template, Template), \
            f"TemplateConfig must be initialized with a Template object, but got {self.template}."
        if self.tokenizer is None:
            self.tokenizer = self.template.tokenizer # noqa
        if hasattr(self, '__template_slots__') and hasattr(self, 'tokens'): return
        self.slots: list[TokenSlot] = []
        template_text = self.template.template
        slot_pattern = re.compile(
            f"(?P<slot_lead>{self.tokenizer.slot_lead_pattern})" +
            r"<(?P<slot_name>[a-zA-Z_][a-zA-Z_0-9]*)>" +
            f"(?P<slot_trail>{self.tokenizer.slot_trail_pattern})"
        )
        previous_end = 0
        template_parts = []
        slots = []
        for slot_match in slot_pattern.finditer(template_text):
            slot_name = slot_match.group('slot_name')
            slot_lead = slot_match.group('slot_lead')
            slot_trail = slot_match.group('slot_trail')
            if slot_name in self.template.__template_slots__:
                slot = self.template.__template_slots__[slot_name]
                slot_config = getattr(self.template, slot_name)
                assert isinstance(slot_config, TokenSlot), \
                    f"Slot {slot_name} in template {self.template.__class__.__name__} must be a TokenSlot object when configuring a template with TemplateConfig, but got {slot_config}."
                slot = slot | slot_config
                start, end = slot_match.span()
                template_parts.append(template_text[previous_end:start])
                slot |= TokenSlot(prefix=slot_lead + slot.prefix, suffix=slot.suffix + slot_trail)
                slots.append(slot)
                previous_end = end
        template_suffix = template_text[previous_end:]
        self.tokens = TokenSequence(
            '', is_attended=self.is_attended, is_label=self.is_label, tokenizer=self.tokenizer)
        for template_part, slot in zip(template_parts, slots):
            self.tokens += template_part
            slot.index = len(self.tokens)
            for name, value in self.tokenizer.slot_affix_replacements.items():
                if value is None: continue
                if isinstance(slot.prefix, str):
                    slot.prefix = slot.prefix.replace(name, value)
                if isinstance(slot.suffix, str):
                    slot.suffix = slot.suffix.replace(name, value)
                if isinstance(slot.trunc_text, str):
                    slot.trunc_text = slot.trunc_text.replace(name, value)
            slot.trunc_text = TokenSequence(
                    slot.trunc_text, is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer)
            if isinstance(slot.max, float):
                assert slot.max > slot.min + len(slot.trunc_text), \
                    f"Slot {slot.name} has a max value length shorter than the sum of its min value length (plus length of truncation_text tokens such as '...')."
            self.slots.append(slot)
        self.tokens += template_suffix

    def __call__(self, *args, **kwargs) -> TT:
        return self.template.__class__(*args, **kwargs)

    def fill(self, values: TT | T.Iterable[TT]) -> TokenSequence | TokenSequences:
        if not isinstance(values, type(self.template)):
            assert hasattr(values, '__iter__'), \
                f"Values must be an iterable of {self.template.__class__.__name__}"
            for value in values:
                assert isinstance(value, type(self.template)), \
                    f"Values must be an iterable of {self.template.__class__.__name__}"
            return TokenSequences(
                [self.fill(value) for value in values],
                pad_to_same_length=self.pad_to_same_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_side=self.pad_side,
                tokenizer=self.tokenizer
            )
        value_seqs, num_expected_out_tokens, template = self._tokenize_value_sequences(values)
        seq_with_values = self._fill(value_seqs, num_expected_out_tokens, template)
        return seq_with_values

    def _tokenize_value_sequences(self, values):
        template = self.tokens
        num_expected_out_tokens = 0
        unique_value_seqs = {}
        for slot in self.slots:
            if slot.name in unique_value_seqs: continue
            value = getattr(values, slot.name)
            if value is Ellipsis:
                template = self.tokens[:slot.index]
                num_expected_out_tokens = slot.min_out
                break
            value_text = ''.join((slot.prefix, getattr(values, slot.name), slot.suffix))
            value_seq = TokenSequence(
                value_text, is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer
            )
            unique_value_seqs[slot.name] = value_seq
        value_seqs = {slot: unique_value_seqs[slot.name] for slot in self.slots}
        return value_seqs, num_expected_out_tokens, template

    def _fill(self, value_seqs, num_expected_out_tokens, template):
        total_len = sum(
            (len(template), num_expected_out_tokens,
            sum(len(value_seq) for value_seq in value_seqs.values()))
        )
        trunc_amounts = {}
        for slot, value_seq in value_seqs.items():
            if slot.truncatable and slot.max and len(value_seq) > slot.max:
                amount_to_trunc_for_this_slot = len(value_seq) - slot.max
                trunc_amounts[slot] = amount_to_trunc_for_this_slot + len(slot.trunc_text)
                total_len -= amount_to_trunc_for_this_slot
        if self.max_length is not None and total_len > self.max_length:
            slots_in_trunc_order = sorted(trunc_amounts, key=lambda slot: slot.trunc_rank)
            for slot in slots_in_trunc_order:
                amount_to_trunc = self.max_length - total_len
                value_seq = value_seqs[slot]
                amount_possible_trunc = max(
                    0,
                    len(value_seq) - slot.min - trunc_amounts.get(slot, len(slot.trunc_text))
                )
                amount_to_trunc_for_this_slot = min(amount_to_trunc, amount_possible_trunc)
                if amount_to_trunc_for_this_slot:
                    trunc_amounts[slot] = amount_to_trunc_for_this_slot + trunc_amounts.get(
                        slot, len(slot.trunc_text)
                    )
                    total_len -= amount_to_trunc_for_this_slot
                if total_len <= self.max_length: break
        for slot, trunc_amount in trunc_amounts.items():
            value_seq = value_seqs[slot]
            if slot.trunc_side == 'L':
                truncated = slot.trunc_text[:]
                truncated.extend(value_seq[trunc_amount:])
            else:
                truncated = value_seq[:-trunc_amount]
                truncated.extend(slot.trunc_text)
            value_seqs[slot] = truncated
        seq_with_values = TokenSequence(
            is_attended=self.is_attended, is_label=self.is_label, tokenizer=self.tokenizer
        )
        end_index = 0
        for slot, value in value_seqs.items():
            seq_with_values.extend(template[end_index:slot.index])
            seq_with_values.extend(value)
            end_index = slot.index
        seq_with_values.extend(template[end_index:])
        return seq_with_values
