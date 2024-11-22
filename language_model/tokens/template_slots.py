
from __future__ import annotations


import dataclasses as dc
import functools as ft
import copy as cp

import ezpyzy as ez


@dc.dataclass
class TokenSlot(ez.Config):
    name: str = 'text'
    is_label: bool = False
    max: int = None
    min: int = 0
    truncatable: bool = True
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str = '...'
    min_out: int = 0
    prefix: str = ''
    suffix: str = ''

    def __post_init__(self):
        self.index: int = 0


Slot = str | TokenSlot

@dc.dataclass
class InputSlot(TokenSlot):
    name: str = 'input'
    is_label: bool = False
    max: int = None
    min: int = 0
    truncatable: bool = True
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str = '...'
    min_out: int = 0
    prefix: str = ''
    suffix: str = ''

def Input(
    name: str = 'input',
    is_label: bool = False,
    max: int = None,
    min: int = 0,
    truncatable: bool = True,
    trunc_side: str = 'L',
    trunc_rank: float = 1.0,
    trunc_text: str = '...',
    min_out: int = 0,
    prefix: str = '',
    suffix: str = '',
) -> str|InputSlot:
    return ...
vars().update(Input=InputSlot)

@dc.dataclass
class OutputSlot(TokenSlot):
    name: str = 'output'
    is_label: bool = True
    max: int = None
    min: int = 0
    truncatable: bool = True
    trunc_side: str = 'R'
    trunc_rank: float = 1.0
    trunc_text: str = ''
    min_out: int = 0
    prefix: str = ''
    suffix: str = '{eos}'

def Output(
    name: str = 'output',
    is_label: bool = True,
    max: int = None,
    min: int = 0,
    truncatable: bool = True,
    trunc_side: str = 'R',
    trunc_rank: float = 1.0,
    trunc_text: str = '',
    min_out: int = 0,
    prefix: str = '',
    suffix: str = '{eos}',
) -> str|OutputSlot:
    return ...
vars().update(Output=OutputSlot)




@dc.dataclass
class TemplateSlots(ez.MultiConfig[TokenSlot]):
    pass