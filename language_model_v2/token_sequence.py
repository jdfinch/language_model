
from __future__ import annotations

import dataclasses as dc
import re
import collections as coll
import itertools as it
from language_model_v2.utils.config import config, Config
from language_model_v2.utils import ansi

# black magic type hinting of config as dataclass
from dataclasses import dataclass; vars().update(dataclass=config)

from transformers import PreTrainedTokenizer

import typing as T


def _imports(): pass


default: T.Any = object()


def WrapTokenizer(tokenizer: PreTrainedTokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '-'
        tokenizer.pad_token_id, = tokenizer.encode('-', add_special_tokens=False)
    class TokenSequence(TokenSequenceBase): pass
    TokenSequence.tokenizer = tokenizer
    class TokenTemplate(TokenTemplateBase): pass
    TokenTemplate.tokenizer = tokenizer
    TokenTemplate.TokenSequence = TokenSequence
    return TokenSequence, TokenTemplate



class TokenSequenceBase:
    tokenizer: PreTrainedTokenizer = None
    _display_width = 80
    _display_token_colors = ((40, 25, 65), (65, 25, 35), (25, 45, 65))
    _display_padding_color = ('black',)
    _display_foreground_color = (150, 150, 150)
    _display_label_color = (255, 255, 255)
    _display_slot_color = (80, 60, 30)

    def __init__(
        self,
        *sequence: str | T.Iterable[Token],
        is_attended: bool = default,
        is_label: bool = default,
    ):
        self.tokens: list[Token] = []
        """Tokens as (id, str, is_attended, is_label) tuples."""
        for sequence in sequence:
            if isinstance(sequence, TokenSequenceBase):
                self.tokens.extend(sequence.tokens)
            elif isinstance(sequence, str):
                token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
                self.tokens.extend(
                    Token(
                        token_id,
                        self.tokenizer.decode(token_id),
                        True if is_attended is default else is_attended,
                        False if is_label is default else is_label
                    )
                    for token_id in token_ids)
            else:
                self.tokens.extend(sequence)

    def text(self):
        return ''.join(t.text if isinstance(t, Token) else t.as_text() for t in self) # noqa

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]

    def __str__(self):
        if len(self) > 10:
            return f'<TokenSequence len {len(self)}: {"|".join(t.text if isinstance(t, Token) else t.as_text() for t in self.tokens[:10])}|...>' # noqa
        else:
            return f'<TokenSequence len {len(self)}: {"|".join(t.text if isinstance(t, Token) else t.as_text() for t in self.tokens)}>' # noqa

    def __repr__(self):
        return str(self)

    def display(self):
        return display_tokens(self)


class TokenTemplateBase:
    _slot_pattern = re.compile(r"#\[([a-z_A-Z0-9]+=[^,\]]*(?:, ?[a-z_A-Z0-9]+=[^,\]]*)*)]#")
    _display_width = 80
    _display_token_colors = ((40, 25, 65), (65, 25, 35), (25, 45, 65))
    _display_padding_color = ('black',)
    _display_foreground_color = (150, 150, 150)
    _display_label_color = (255, 255, 255)
    _display_slot_color = (80, 60, 30)
    tokenizer: PreTrainedTokenizer = None
    TokenSequence = TokenSequenceBase

    def __init__(self,
        *sequence: str | 'TokenTemplateBase' | 'TokenSlot' | T.Iterable[Token | TokenSlot],
        is_attended: bool = default,
        is_label: bool = default,
    ):
        self.tokens: list[TokenSlot | Token] = []
        """Tokens as (id, str, is_attended, is_label) tuples. Input/OutputSequence objects represent slots to fill in the sequence with input/output text."""
        self.slots: dict[str, TokenSlot] = {}
        for sequence in sequence:
            if isinstance(sequence, TokenTemplateBase):
                self.tokens.extend(sequence.tokens)
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
                        self.tokens.extend(
                            Token(
                                token_id,
                                self.tokenizer.decode(token_id),
                                True if is_attended is default else is_attended,
                                False if is_label is default else is_label
                            )
                            for token_id in token_ids)
                    if match:
                        arguments = dict(
                            [x.strip() for x in argument.split('=', 1)]
                            for argument in match.group(1).split(','))
                        if 'input' in arguments:
                            arguments['name'] = arguments.pop('input')
                            slot = InputTokenSlot(**arguments, index=len(self.tokens))
                        elif 'output' in arguments:
                            arguments['name'] = arguments.pop('output')
                            slot = OutputTokenSlot(**arguments, index=len(self.tokens))
                        else:
                            raise ValueError(f"Slot be named using input= or output=, but got {arguments}")
                        assert slot.name not in self.slots, f"Duplicate slot name {slot.name} detected when constructing TokenSequence."
                        self.slots[slot.name] = slot
                        self.tokens.append(slot)
            elif isinstance(sequence, TokenSlot):
                self.tokens.append(sequence)
            else:
                self.tokens.extend(sequence)

    def fill(self,
        slots: dict[str, str | 'TokenTemplateBase'] | T.Iterable[str, str | 'TokenTemplateBase'],
        max_length: int = None,
        min_length: int = None,
    ):
        assert all(slot in self.slots for slot in slots), \
            f"Slots {set(slots) - set(self.slots)} not found in TokenSequence."
        slot_subseqs: list[tuple[TokenSlot, TokenTemplateBase]] = []
        filled = []
        previous_splitter = 0
        for slot_name, text in slots.items():
            slot = self.slots[slot_name]
            if isinstance(text, str):
                text = type(self)(text, is_label=slot.is_label)
            slot_subseqs.append((slot, text))
        for slot, subseq in slot_subseqs:
            if slot.max and len(subseq) > slot.max:
                if slot.trunc_side == 'L':
                    subseq.tokens = subseq.tokens[-slot.max:]
                else:
                    subseq.tokens = subseq.tokens[:slot.max]
        template_length = len(self) - len(self.slots)
        current_length = sum(len(value) for _, value in slot_subseqs) + template_length
        if max_length is not None and current_length > max_length:
            slot_trunc_candidates = iter(sorted(slot_subseqs, key=lambda x: x[0].trunc_rank))
            for slot, subseq in slot_trunc_candidates:
                amount_to_truncate = max(min(current_length - max_length, len(subseq) - slot.min, len(subseq)), 0)
                if amount_to_truncate:
                    if slot.trunc_side == 'L':
                        subseq.tokens = subseq.tokens[amount_to_truncate:]
                    else:
                        subseq.tokens = subseq.tokens[:-amount_to_truncate]
                    current_length = sum(len(value) for _, value in slot_subseqs) + template_length
                    if current_length <= max_length:
                        break
            else: # nobreak
                raise ValueError(f"Could not truncate slot text to fit within max_length {max_length} (max truncation was reached after sequence was cut down to {current_length} tokens).")
        for slot, value in slot_subseqs:
            splitter = slot.index
            prefix = self.tokens[previous_splitter:splitter]
            filled.extend(prefix)
            filled.extend(value.tokens)
            previous_splitter = splitter + 1
        filled.extend(self.tokens[previous_splitter:])
        if min_length is not None and len(filled) < min_length:
            pad_length = min_length - len(filled)
            padding = [Token(self.tokenizer.pad_token_id, self.tokenizer.pad_token, False, False)] * pad_length
            filled = padding + filled
        return self.TokenSequence(filled)

    def text(self):
        return ''.join(t.text if isinstance(t, Token) else t.as_text() for t in self)

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]

    def __str__(self):
        if len(self) > 10:
            return f'<TokenTemplate len {len(self)}: {"|".join(t.text if isinstance(t, Token) else t.as_text() for t in self.tokens[:10])}|...>'
        else:
            return f'<TokenTemplate len {len(self)}: {"|".join(t.text if isinstance(t, Token) else t.as_text() for t in self.tokens)}>'

    def __repr__(self):
        return str(self)

    def display(self):
        return display_tokens(self)


def display_tokens(seq: TokenSequenceBase | TokenTemplateBase):
    num_slots = len(seq.slots) if hasattr(seq, 'slots') else 0
    num_tokens = len(seq) - num_slots
    if num_slots > 0:
        print(f"{seq.__class__.__name__} with {num_tokens} tokens and {num_slots} slots:")
    else:
        print(f"{seq.__class__.__name__} with {num_tokens} tokens:")
    display_tokens = []
    for token, token_background_color in zip(seq, it.cycle(seq._display_token_colors)):
        if isinstance(token, TokenSlot):
            token_text = f'#[{token.name}]#'
            token = Token(-1, token_text, True, bool(isinstance(token, OutputTokenSlot)))
            token_text = f"{ansi.color(*seq._display_slot_color).bg}{token_text}{ansi.reset}"
        else:
            token_text = token.text
            newlinestripped = token_text.rstrip('\n')
            num_newlines = len(token_text) - len(newlinestripped)
            if num_newlines > 0:
                token_text = ''.join((newlinestripped, "\\n" * num_newlines, ansi.reset, '\n'*num_newlines))
        display_tokens.append(ansi.color(*token_background_color).bg)
        if token.is_label:
            display_tokens.append(ansi.color(*seq._display_label_color).fg)
        elif token.is_attended:
            display_tokens.append(ansi.color(*seq._display_foreground_color).fg)
        else:
            display_tokens.append(ansi.color(*seq._display_padding_color).fg)
        display_tokens.append(token_text)
    display_tokens.append(ansi.reset)
    print(''.join(display_tokens), end='\n\n')


def _tokenclasses(): pass


Token = coll.namedtuple('Token', 'id text is_attended is_label')

@dataclass
class TokenSlot(Config):
    name: str
    max: int = None
    min: int = 0
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    index: int = None
    is_label: bool = False

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
        self.is_label = bool(self.is_label)

    def as_text(self):
        return f"#[{self.name}]#"


@dataclass
class InputTokenSlot(TokenSlot):
    pass

@dataclass
class OutputTokenSlot(TokenSlot):
    def __post_init__(self):
        super().__post_init__()
        if 'trunc_side' not in self.__config__: self.trunc_side = 'R'
        if 'trunc_rank' not in self.__config__: self.trunc_rank = 0.0
        if 'is_label' not in self.__config__: self.is_label = True



def main():
    import textwrap as tw
    template = tw.dedent("""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 26 Jul 2024

    You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

    #[input=my_input]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    #[output=my_output, trunc_side=left, max=7, min=3]#<|eot_id|>
    """).strip()

    from transformers import AutoTokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    LlamaTokenSequence, LlamaTokenTemplate = WrapTokenizer(llama_tokenizer)
    template_sequence = LlamaTokenTemplate(template)
    template_sequence.display()
    filled_sequence = template_sequence.fill(
        dict(my_input="What is the capital of France?", my_output="The capital of France is Paris."),
        max_length=50,
    )
    filled_sequence.display()


if __name__ == '__main__':
    main()






































