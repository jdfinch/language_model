
import dataclasses as dc

import ezpyzy.ansi as ansi

import typing as T


@dc.dataclass
class TokenPrinterSettings:
    tokens: T.Iterable = None
    token_colors:tuple = ((55, 45, 120), (30, 70, 130), (20, 90, 110))
    foreground_color: tuple = (200, 200, 200)
    padding_color:tuple = ('black',)
    label_color:tuple = (255, 255, 255)
    label_style: str|None = ansi.bold
    slot_color:tuple = (80, 60, 30)