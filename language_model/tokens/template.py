
from __future__ import annotations

import dataclasses as dc
import copy as cp
import functools as ft
import re

import ezpyzy as ez

from language_model.tokens.tokenizer import Tokenizer


@dc.dataclass
class TokenSlot(ez.Config):
    name: str = 'text'
    is_label: bool = False
    max: int|None = None
    min: int = 0
    trunc: bool = True
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str = '...'
    prefix: str = ''
    suffix: str = ''

    def __post_init__(self):
        super().__post_init__()
        self.token_index: int = None # noqa
        self.template: 'SegmentTemplate' = None # noqa
        self.slot_index: int = None # noqa


Slot = str | TokenSlot

@dc.dataclass
class InputSlot(TokenSlot):
    name: str = 'input'
    is_label: bool = False
    max: int|None = None
    min: int = 0
    trunc: bool = True
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    trunc_text: str = '...'
    prefix: str = ''
    suffix: str = ''

def Input(
    name: str = 'input',
    is_label: bool = False,
    max: int|None = None,
    min: int = 0,
    trunc: bool = True,
    trunc_side: str = 'L',
    trunc_rank: float = 1.0,
    trunc_text: str = '...',
    prefix: str = '',
    suffix: str = '',
) -> str|InputSlot:
    return ...
vars().update(Input=InputSlot)

@dc.dataclass
class OutputSlot(TokenSlot):
    name: str = 'output'
    is_label: bool = True
    max: int|None = None
    min: int = 0
    trunc: bool = True
    trunc_side: str = 'R'
    trunc_rank: float = 0.0
    trunc_text: str = '...{eos}'
    prefix: str = ''
    suffix: str = '{eos}'

def Output(
    name: str = 'output',
    is_label: bool = True,
    max: int|None = None,
    min: int = 0,
    trunc: bool = True,
    trunc_side: str = 'R',
    trunc_rank: float = 0.0,
    trunc_text: str = '...{eos}',
    prefix: str = '',
    suffix: str = '{eos}',
) -> str|OutputSlot:
    return ...
vars().update(Output=OutputSlot)




@dc.dataclass
class TemplateSlots(ez.MultiConfig[TokenSlot]):
    pass


def fields(cls_or_instance) -> list[dc.Field]: return dc.fields(cls_or_instance) # noqa

slot_pattern = re.compile(r"<.*?>")


class TemplateMeta(type):
    def __new__(cls, name, bases, dct):
        if bases:
            assert 'template' in dct and isinstance(dct['template'], (str, Template)), \
                f"Template class {name} must define a template string."
            for attr, value in dct.items():
                if isinstance(value, TokenSlot):
                    value.name = attr
            def get_template_str(tmp):
                if isinstance(tmp, Template):
                    tmp_text = tmp.template
                    for attr, value in vars(tmp).items():
                        if isinstance(value, str):
                            tmp_text = tmp_text.replace(f"<{attr}>", value)
                        elif isinstance(value, Template):
                            tmp_text = tmp_text.replace(f"<{attr}>", get_template_str(value))
                    return tmp_text
                else:
                    return tmp
            template = get_template_str(dct['template'])
            dct['template'] = template
            dct['__template_slots__'] = {}
            for attr, value in dct.items():
                if isinstance(value, TokenSlot):
                    assert f"<{attr}>" in template, \
                        f"Slot <{attr}> was defined as a class field of {name} but not in template text:  {template}"
                    dct[attr] = dc.field(default_factory=ft.partial(cp.deepcopy, value))
                    dct['__template_slots__'][value.name] = value
        return super().__new__(cls, name, bases, dct)

class Template(metaclass=TemplateMeta):
    template: str | Template = None
    __template_slots__: dict[str, TokenSlot]

    def __str__(self):
        template = self.template
        for attr, value in vars(self).items():
            if isinstance(value, str):
                template = template.replace(f"<{attr}>", value)
        return template

@dc.dataclass
class SegmentTemplate(ez.Config):
    template: str|Template = None
    """A custom dataclass object with a class attribute 'template' that defines a template string, and TokenSlot objects as and_unconfigured for each slot in the template::

    @dc.dataclass
    class MyTemplate(Template):
        template = 'This is a <adjective> <noun>.'
        adjective: Slot = Input() 
        noun: Slot = Input()
    """
    slots: TemplateSlots = TemplateSlots()
    name: str = None
    is_attended: bool = True
    is_label: bool = False
    trunc_content: bool = True
    trunc_segment: bool = False
    trunc_segment_if_no_content: bool = True
    trunc_segment_rank: float = 1.0
    trunc_segment_side: str = 'L'
    tokenizer: Tokenizer = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.template, str) and not self.configured.has.slots:
            for match in slot_pattern.findall(self.template):
                slot_name = match[1:-1]
                if slot_name not in self.slots:
                    if slot_name == 'output':
                        self.slots[slot_name] = OutputSlot(name=slot_name)
                    else:
                        self.slots[slot_name] = InputSlot(name=slot_name)
        else:
            template = self.template
            self.template = template.template
            for slot_name, slot in vars(template).items():
                if isinstance(slot, TokenSlot):
                    assert f"<{slot_name}>" in self.template, \
                        f"Slot {slot_name} was defined as a class field of {template.__class__.__name__} but not in template text:  {self.template}"
                    slot_copy = cp.deepcopy(slot)
                    self.slots[slot_name] = slot_copy
                    slot_copy.name = slot_name
            self.name = template.__class__.__name__
        assert isinstance(self.name, str), \
            "SegmentTemplate must have a name attribute, either as a string or from the template class name."


@dc.dataclass
class Templates(ez.MultiConfig[SegmentTemplate]):
    def __post_init__(self):
        super().__post_init__()
        for name, template in self:
            if isinstance(template, type) and issubclass(template, Template):
                is_configured = name in self.configured
                template = template()
                segment_template = SegmentTemplate(template=template)
                segment_template.configured.set('template', template, configured=is_configured)
                setattr(self, name, segment_template)
                self.configured.set(name, segment_template, configured=is_configured)
            elif isinstance(template, Template):
                is_configured = name in self.configured
                segment_template = SegmentTemplate(template=template)
                segment_template.configured.set('template', template, configured=is_configured)
                setattr(self, name, segment_template)
                self.configured.set(name, segment_template, configured=is_configured)


if __name__ == '__main__':

    @dc.dataclass
    class MyTemplate(Template):
        template = 'This is a <adjective> <noun>. The <noun> is <phrase>!'
        adjective: Slot = Input()
        noun: Slot = Input()
        phrase: Slot = Output()


    template = MyTemplate()
    print(template)

    filled = MyTemplate(adjective='big', noun='dog', phrase='happy')
    print(filled, '\n')


    template = SegmentTemplate(template=MyTemplate(
        adjective=Input(max=24, min=16)
    ), trunc_content=False, trunc_segment_rank=1.5)

    print(template.configured.json())
























