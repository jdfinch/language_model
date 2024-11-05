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
import ezpyzy as ez

# black magic type hinting of base as dataclass
from dataclasses import dataclass; vars().update(dataclass=config) # noqa

import transformers as hf
import tokenizers as tok

import typing as T

default: T.Any = object()

def fields(cls_or_instance) -> list[dc.Field]: return dc.fields(cls_or_instance) # noqa


class Tokenizer(abc.ABC):

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenizes text into token IDs"""
    @abc.abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decodes token IDs into text."""

    @property
    @abc.abstractmethod
    def slot_lead_pattern(self) -> str:
        """Regex pattern for matching chars preceding __template_slots__ that should be included in the slot prefix."""

    @property
    @abc.abstractmethod
    def slot_trail_pattern(self) -> str:
        """Regex pattern for matching chars following __template_slots__ that should be included in the slot suffix."""

    @property
    @abc.abstractmethod
    def slot_affix_replacements(self) -> dict[str, str]:
        """Replacements for special characters in slot prefixes and suffixes."""


class TokenSequence:
    tokenizer: Tokenizer = None

    def __init__(self,
        sequence: str|TokenSequence = '',
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

    def extend(self, sequence: str|TokenSequence, is_attended: bool = True, is_label:bool = False):
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

    def ansi(self):
        return TokenPrinter().ansi(self)

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
                        seq.token_ids = padding + seq.token_ids
                    else:
                        seq.token_ids += padding

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

@dc.dataclass(frozen=True)
class TokenSlot:
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

@dc.dataclass(frozen=True)
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

@dc.dataclass(frozen=True)
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
        for slot_name, value in attrs.items():
            if isinstance(value, TokenSlot):
                assert '{'+slot_name+'}' in cls.template, \
                    f"Slot {slot_name} was defined as a class field of {name} but not in template text:  {cls.template}"
                value = dc.replace(value, name=slot_name)
                cls.__template_slots__[slot_name] = value
                setattr(cls, slot_name, dc.field(default_factory=ft.partial(cp.copy, value)))
        return cls

class Template(metaclass=TemplateMeta):
    template: str
    __template_slots__: dict[str, TokenSlot]

    def __iter__(self):
        return iter(self.__template_slots__)

    def __getitem__(self, item: str) -> str|TokenSlot:
        return getattr(self, item)


class Foo(Template):
    template = 'hello'

foo = Foo()

TT = T.TypeVar('TT', bound=Template)

@dc.dataclass
class TemplateConfig(T.Generic[TT]):
    template: TT
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
    trunc_segment_rank: float = 1.0
    trunc_segment_side: str = 'L'
    max_length: int = None
    pad: bool = True
    pad_to_same_length: bool = True
    pad_to_multiple_of: int = 1
    pad_side: str = 'L'
    tokenizer: Tokenizer = None

    def __post_init__(self):
        if hasattr(self, '__template_slots__') and hasattr(self, 'tokens'): return
        self.slots: list[TokenSlot] = []
        template_text = self.template.template
        slot_pattern = re.compile(
            f"(?P<slot_lead>{self.tokenizer.slot_lead_pattern})" +
            r"\{(?P<slot_name>[a-zA-Z_][a-zA-Z_0-9]*)}" +
            f"(?P<slot_trail>{self.tokenizer.slot_trail_pattern})")
        previous_end = 0
        template_parts = []
        slots = []
        for slot_match in slot_pattern.finditer(template_text):
            slot_name = slot_match.group('slot_name')
            slot_lead = slot_match.group('slot_lead')
            slot_trail = slot_match.group('slot_trail')
            if slot_name in self.template.__template_slots__:
                slot = self.template.__template_slots__[slot_name]
                start, end = slot_match.span()
                template_parts.append(template_text[previous_end:start])
                slot_clone = dc.replace(slot, prefix=slot_lead+slot.prefix, suffix=slot.suffix+slot_trail)
                slots.append(slot_clone)
                previous_end = end
        template_suffix = template_text[previous_end:]
        self.tokens = TokenSequence(
            '', is_attended=self.is_attended, is_label=self.is_label, tokenizer=self.tokenizer)
        for template_part, slot in zip(template_parts, slots):
            self.tokens += template_part
            slot_clone = dc.replace(slot, index=len(self.tokens))
            for name, value in self.tokenizer.slot_affix_replacements.items():
                if value is None: continue
                if isinstance(slot_clone.prefix, str):
                    slot_clone = dc.replace(slot_clone, prefix=slot_clone.prefix.replace(name, value))
                if isinstance(slot_clone.suffix, str):
                    slot_clone = dc.replace(slot_clone, suffix=slot_clone.suffix.replace(name, value))
                if isinstance(slot_clone.trunc_text, str):
                    slot_clone = dc.replace(slot_clone, trunc_text=slot_clone.trunc_text.replace(name, value))
            slot_clone = dc.replace(slot_clone, trunc_text = TokenSequence(slot_clone.trunc_text,
                is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer))
            if isinstance(slot_clone.max, float):
                assert slot_clone.max > slot_clone.min + len(slot_clone.trunc_text), \
                    f"Slot {slot_clone.name} has a max value length shorter than the sum of its min value length (plus length of truncation_text tokens such as '...')."
            self.slots.append(slot_clone)
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
            return TokenSequences([self.fill(value) for value in values],
                pad_to_same_length=self.pad_to_same_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_side=self.pad_side,
                tokenizer=self.tokenizer)
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
                value_text, is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer)
            unique_value_seqs[slot.name] = value_seq
        value_seqs = {slot: unique_value_seqs[slot.name] for slot in self.slots}
        return value_seqs, num_expected_out_tokens, template

    def _fill(self, value_seqs, num_expected_out_tokens, template):
        total_len = sum(
            (len(template), num_expected_out_tokens,
            sum(len(value_seq) for value_seq in value_seqs.values())))
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
                amount_possible_trunc = max(0,
                    len(value_seq) - slot.min - trunc_amounts.get(slot, len(slot.trunc_text)))
                amount_to_trunc_for_this_slot = min(amount_to_trunc, amount_possible_trunc)
                if amount_to_trunc_for_this_slot:
                    trunc_amounts[slot] = amount_to_trunc_for_this_slot + trunc_amounts.get(
                        slot, len(slot.trunc_text))
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
            is_attended=self.is_attended, is_label=self.is_label, tokenizer=self.tokenizer)
        end_index = 0
        for slot, value in value_seqs.items():
            seq_with_values.extend(template[end_index:slot.index])
            seq_with_values.extend(value)
            end_index = slot.index
        seq_with_values.extend(template[end_index:])
        return seq_with_values


@dc.dataclass
class TokenPrinterSettings:
    tokens: T.Iterable = None
    token_colors:tuple = ((55, 45, 120), (30, 70, 130), (20, 90, 110))
    foreground_color: tuple = (200, 200, 200)
    padding_color:tuple = ('black',)
    label_color:tuple = (255, 255, 255)
    label_style: str|None = ansi.bold
    slot_color:tuple = (80, 60, 30)


class Llama3Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = hf.AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
        self.pad_token = '-'
        self.pad_token_id, = self.tokenizer.encode(self.pad_token, add_special_tokens=False)
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
    slot_lead_pattern = r" ?"
    slot_trail_pattern = ""
    slot_affix_replacements = {'{eos}': '<|eot_id|>', '{bos}': '<|begin_of_text|>'}

class Llama3TemplateConfig(TemplateConfig):
    tokenizer = Llama3Tokenizer()

@dc.dataclass
class MyTemplate(Template):
    template = 'This is a {adjective} {noun}.\n{description}'
    adjective: Slot = InputSlot()
    noun: Slot = InputSlot()
    description: Slot = OutputSlot()

# template = Llama3TemplateConfig(MyTemplate())
# seq = template.fill(MyTemplate(
#     adjective='really big', noun='dog in the park', description='A big dog I saw in the park.'))
# print(seq.tokens())


class TokenTemplatesMeta(type):
    def __new__(typ, name, bases, attrs):
        cls = super().__new__(typ, name, bases, attrs)
        for name, attr in attrs.items():
            if isinstance(attr, type) and issubclass(attr, Template) and attr.template is None:
                attr.template = attrs.get('template', None)
        return cls

@dc.dataclass
class TokenTemplates(metaclass=TokenTemplatesMeta):
    tokenizer = None
    max_length: int = None
    pad_to_same_length: bool = True
    pad_to_multiple_of: int = 8
    pad_side: str = 'L'
    max_segments: int | None = None

    def __post_init__(self):
        self.tokenizer: Tokenizer
        self.templates: dict[type[Template], TemplateConfig] = {}
        template_fields = {field.name for field in fields(self)} - token_templates_base_fields
        if self.__class__ is TokenTemplates:
            raise ValueError(f"No templates are defined in the base TokenTemplates class. Initialize an instance of a TokenTemplates subclass with at least one TokenTemplate.")
        for field in fields(self):
            if field.name in template_fields:
                template = getattr(self, field.name, None)
                if isinstance(template, type) and issubclass(template, Template):
                    template = TemplateConfig(template(), tokenizer=self.tokenizer)
                    setattr(self, field.name, template)
                elif template.tokenizer is None:
                    template.tokenizer = self.tokenizer
                self.templates[field.default] = getattr(self, field.name)

    def fill(self, segments_values: list[Template]|T.Iterable[list[Template]]) -> TokenSequence|TokenSequences:
        if not isinstance(segments_values, list) or isinstance(segments_values[0], list):
            return TokenSequences([self.fill(value) for value in segments_values],
                pad_to_same_length=self.pad_to_same_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_side=self.pad_side,
                tokenizer=self.tokenizer)
        for segment_values in segments_values:
            assert type(segment_values) in self.templates, \
                f"Values must be a list of {', '.join(cls.__name__ for cls in self.templates)} Template instances for TokenTemplates {type(self).__qualname__}."

        # original templates and corresponding values
        segments_values: list[Template]
        templates = [self.templates[type(value)] for value in segments_values]

        # find slot for generation
        slot_for_generation = None
        index_of_slot_for_generation = None
        index_of_segment_for_generation = None
        num_expected_out_tokens = 0
        for i, (segment_values, template) in enumerate(zip(segments_values, templates)):
            for j, slot in enumerate(template.slots):
                value = getattr(segment_values, slot.name)
                if value is Ellipsis:
                    slot_for_generation = slot
                    index_of_slot_for_generation = j
                    index_of_segment_for_generation = i
                    num_expected_out_tokens = slot.min_out
                    break
            if index_of_segment_for_generation is not None:
                break

        # truncate segments after segment with output, and slots from segment with output
        if index_of_segment_for_generation is not None:
            segments_values = segments_values[:index_of_segment_for_generation+1]
            templates = templates[:index_of_segment_for_generation+1]
            template_with_generation = templates[index_of_segment_for_generation]
            template_with_generation = cp.deepcopy(template_with_generation)
            template_with_generation.slots = template_with_generation.slots[:index_of_slot_for_generation]
            template_with_generation.tokens = template_with_generation.tokens[:slot_for_generation.index]
            templates[index_of_segment_for_generation] = template_with_generation

        # create segments table where original templates/segments_values indices are IDs
        segments_table = {i: (template, segment_values)
            for i, (template, segment_values) in enumerate(zip(templates, segments_values))}
        _segments_table_template, _segments_table_values = 0, 1

        # sort segments by trunc_rank, trunc_side, and truncability (and filter by truncability)
        seg_trunc_cands = dict(ez.sort([(i, (template, segment_values))
                for i, (template, segment_values) in segments_table.items()
                if template.trunc_segment and i != index_of_segment_for_generation],
            by=[(template.trunc_segment_rank, template.trunc_segment_side)
                for i, (template, _) in segments_table.items()
                if template.trunc_segment and i != index_of_segment_for_generation]))
        _seg_trunc_cands_template, _seg_trunc_cands_values = 0, 1

        # truncate segments based on max sequences
        if self.max_segments and len(segments_table) > self.max_segments:
            for i in it.islice(seg_trunc_cands, len(segments_table) - self.max_segments):
                del segments_table[i]
                del seg_trunc_cands[i]

        # tokenize all value sequences in remaining segments
        segs_value_seqs = {i: (template, list[TokenSequence]()) for i, (template, _) in segments_table.items()}
        for i, (template, segment_values) in segments_table.items():
            for slot in template.slots:
                value_text = ''.join((slot.prefix, getattr(segment_values, slot.name), slot.suffix))
                value_seq = TokenSequence(value_text,
                    is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer)
                segs_value_seqs[i][1].append(value_seq)

        # compute min and max tokens of each slot
        slot_value_table = {}  # (seg_idx, slot_idx) -> (slot, value, min, max)
        for i, (template, seg_value_seqs) in segs_value_seqs.items():
            for j, (slot, seq) in enumerate(zip(template.slots, seg_value_seqs)):
                if slot.truncatable and template.trunc_content:
                    min_tokens = min(slot.min + len(slot.trunc_text), len(seq))
                    max_tokens = max(min(len(seq), len(seq) if slot.max is None else slot.max), min_tokens)
                else:
                    max_tokens = min_tokens = len(seq)
                slot_value_table[(i, j)] = (slot, seq, min_tokens, max_tokens)
        _slot_value_table_slot, _slot_value_table_seq = 0, 1
        _slot_value_table_min, _slot_value_table_max = 2, 3

        # sort slots by trunc_rank, trunc_side, and truncability (and filter by truncability)
        slot_trunc_cands = dict(ez.sort([
            ((i, j), [slot, seq, min_tokens, max_tokens])
            for (i, j), (slot, seq, min_tokens, max_tokens) in slot_value_table.items()
            if min_tokens < max_tokens
        ], by=[
            (slot.trunc_rank, slot.trunc_side)
            for (i, _), (slot, _, min_tokens, max_tokens) in slot_value_table.items()
            if min_tokens < max_tokens
        ]))
        _slot_trunc_cands_slot, _slot_trunc_cands_seq = 0, 1
        _slot_trunc_cands_min, _slot_trunc_cands_max = 2, 3

        # calculate current total length if we truncated and merged all segments as-is
        current_len = (num_expected_out_tokens +
            sum(max_tokens for _, _, _, max_tokens in slot_value_table.values()) +
            sum(len(template.tokens) for template, _ in segs_value_seqs.values()))

        # register truncations for slots and segments
        slots_with_content_trunc = []
        iter_seg_trunc_cands = iter(list(seg_trunc_cands.items()))
        iter_slot_trunc_cands = iter(list(slot_trunc_cands.items()))
        next_seg_to_trunc = next(iter_seg_trunc_cands, None)
        next_slot_to_trunc = next(iter_slot_trunc_cands, None)
        while self.max_length and current_len > self.max_length:
            amount_to_trunc = current_len - self.max_length
            do_trunc_slot = False
            do_trunc_seg = False
            if next_seg_to_trunc is None and next_slot_to_trunc is None:
                raise ValueError(f"Current length {current_len} exceeds max length {self.max_length} but no more truncation candidates are available.")
            elif next_slot_to_trunc is None:
                if len(segments_table) > 1:
                    do_trunc_seg = True
                else:
                    raise ValueError(f"Current length {current_len} exceeds max length {self.max_length} but no more truncation candidates are available.")
            elif next_seg_to_trunc is None:
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                if i in segments_table and amount_to_trunc >= len(slot.trunc_text):
                    do_trunc_slot = True
                else:
                    next_slot_to_trunc = next(iter_slot_trunc_cands, None)
            else:
                (_, (template, seg_value_seqs)) = next_seg_to_trunc
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                if (i in segments_table and amount_to_trunc >= len(slot.trunc_text) and
                  (slot.trunc_rank, slot.trunc_side) > (template.trunc_segment_rank, template.trunc_segment_side)
                ):
                    do_trunc_slot = True
                else:
                    do_trunc_seg = True
            if do_trunc_slot:
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                trunc_amount = min(amount_to_trunc, max_tokens - min_tokens)
                current_len -= trunc_amount
                slot_value_table[(i, j)] = (slot, seq, min_tokens, max_tokens - trunc_amount)
                slots_with_content_trunc.append(((i, j), slot_value_table[(i, j)]))
                next_slot_to_trunc = next(iter_slot_trunc_cands, None)
            elif do_trunc_seg:
                i, (template, seg_value_seqs) = next_seg_to_trunc
                current_len -= len(template.tokens)
                for j in range(len(template.slots)):
                    del slot_trunc_cands[(i, j)]
                    current_len -= slot_value_table[(i, j)][_slot_value_table_max]
                del segments_table[i]
                next_seg_to_trunc = next(iter_seg_trunc_cands, None)

        # recover some tokens from truncation
        for (i, j), (slot, seq, min_tokens, max_tokens) in reversed(slots_with_content_trunc):
            if current_len >= self.max_length: break
            recover_amount = min(slot.max - max_tokens, self.max_length - current_len) # noqa
            current_len += recover_amount
            slot_value_table[(i, j)] = (slot, seq, min_tokens, max_tokens + recover_amount)

        # create final token sequence
        final_seq = TokenSequence(tokenizer=self.tokenizer)
        for i, (template, seg_value_seqs) in segs_value_seqs.items():
            previous_slot_index = 0
            for j, value_seq in enumerate(seg_value_seqs):
                slot, _, min_tokens, max_tokens = slot_value_table[(i, j)]
                final_seq += template.tokens[previous_slot_index:slot.index]
                previous_slot_index = slot.index
                if len(value_seq) > max_tokens:
                    if slot.trunc_side == 'L':
                        final_seq += slot.trunc_text
                        final_seq += value_seq[-max_tokens:]
                    else:
                        final_seq += value_seq[:max_tokens]
                        final_seq += slot.trunc_text
                else:
                    final_seq += value_seq
            final_seq += template.tokens[previous_slot_index:]
        return final_seq




token_templates_base_fields = {field.name for field in fields(TokenTemplates)}




if __name__ == '__main__':

    @dc.dataclass
    class SystemTemplate(Template):
        template = "<|start_header_id|>system<|end_header_id|>\n\n{prompt}\n\n{date}<|eot_id|>"
        prompt: Slot = InputSlot()
        date: Slot = InputSlot()

    @dc.dataclass
    class UserTemplate(Template):
        template = "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>"
        input: Slot = InputSlot()

    @dc.dataclass
    class BotTemplate(Template):
        template = "<|start_header_id|>bot<|end_header_id|>\n\n{output}<|eot_id|>"
        output: Slot = OutputSlot()

    @dc.dataclass
    class LlamaTemplates(TokenTemplates):
        tokenizer = Llama3Tokenizer()
        system: TemplateConfig[SystemTemplate] = SystemTemplate
        user: TemplateConfig[UserTemplate] = UserTemplate
        bot: TemplateConfig[BotTemplate] = BotTemplate


    ##############


    templates = LlamaTemplates(

    )

    chat = [
        LlamaTemplates.system(prompt="You are a helpful assistant.", date="2022-01-01"),
        LlamaTemplates.user(input="What is the weather like today?"),
        LlamaTemplates.bot(output="The weather is sunny!"),
        LlamaTemplates.user("Great! What about tomorrow?"),
    ]

    sequence = templates.fill(chat)
    print(sequence.tokens())



