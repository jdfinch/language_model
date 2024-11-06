
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
        adjective: Slot = Input() 
        noun: Slot = Input()
    """
    is_attended: bool = True
    is_label: bool = False
    trunc_content: bool = True
    trunc_segment: bool = False
    trunc_segment_if_no_content: bool = True
    trunc_segment_rank: float = 1.0
    trunc_segment_side: str = 'L'

    def __post_init__(self):
        assert isinstance(self.template, Template), \
            f"TemplateConfig must be initialized with a Template object, but got {self.template}."
        self.slots: list[TokenSlot] = []
