

from __future__ import annotations

import dataclasses as dc
import inspect
import re
import collections as coll
import itertools as it
import functools as ft
import textwrap as tw
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
    min_out: int = 0
    prefix: str = ''
    suffix: str|type[EOS] = ''

    def __post_init__(self):
        self.index = None
        if self.name.lstrip().startswith('#[') and self.name.endswith(']#'):
            space = self.name.startswith(' ')
            if space:
                self.name = self.name[1:]
            args = [arg.strip() for arg in self.name[2:-2].split(', ')]
            if args and '=' not in args[0]:
                type = args.pop(0)
            elif args and args[0].startswith('type='):
                type = args.pop(0)[5:]
            else:
                type = 'InputSlot'
            if type.lower() == 'input':
                self.__class__ = InputSlot
                vars(self).update(vars(InputSlot()))
            elif type.lower() == 'output':
                self.__class__ = OutputSlot
                vars(self).update(vars(OutputSlot()))
            args = {var.strip(): val.strip() for var, val in (arg.split('=') for arg in args)}
            vars(self).update(args)
            if space:
                self.prefix = ' ' + self.prefix
        if isinstance(self.is_label, str):
            self.is_label = self.is_label.lower() in ('true', 'yes', '1', 't', 'y')
        if isinstance(self.max, str):
            self.max = int(self.max)
        if isinstance(self.min, str):
            self.min = int(self.min)
        if isinstance(self.trunc_rank, str):
            self.trunc_rank = float(self.trunc_rank)
        if isinstance(self.min_out, str):
            self.min_out = int(self.min_out)
        if self.suffix == 'EOS':
            self.suffix = EOS

    def text(self):
        return str(self)

    def __str__(self):
        params = [self.name] + [f'{f.name}={getattr(self, f.name)}' for f in dc.fields(self)[1:]
            if getattr(self, f.name) != f.default and (f.name != 'prefix' or self.prefix not in ' ')
           or f.name == 'name']
        return f"#[{', '.join(params)}]#"

    def __repr__(self):
        params = [self.name]+[f'{f.name}={getattr(self, f.name)}' for f in dc.fields(self)[1:]
            if getattr(self, f.name) != f.default and (f.name != 'prefix' or self.prefix not in ' ')
            or f.name == 'name']
        return f"TokenSlot({', '.join(params)})"

@dc.dataclass
class InputSlot(TokenSlot):
    name: str = 'input'
    is_label: bool = False
    max: int = None
    min: int = 0
    trunc_side: str = 'L'
    trunc_rank: float = 1.0
    min_out: int = 0
    prefix: str = ''
    suffix: str|type[EOS] = ''

@dc.dataclass
class OutputSlot(TokenSlot):
    name: str = 'output'
    is_label: bool = True
    max: int = None
    min: int = 0
    trunc_side: str = 'R'
    trunc_rank: float = 1.0
    min_out: int = 0
    prefix: str = ''
    suffix: str|type[EOS] = EOS

def Input(
    name: str = 'input',
    is_label: bool = False,
    max: int = None,
    min: int = 0,
    trunc_side: str = 'L',
    trunc_rank: float = 1.0,
    min_out: int = 0,
    prefix: str = '',
    suffix: str|type[EOS] = '',
) -> str:
    return InputSlot(name, is_label, max, min, trunc_side, trunc_rank, min_out, prefix, suffix) # noqa

def Output(
    name: str = 'output',
    is_label: bool = True,
    max: int = None,
    min: int = 0,
    trunc_side: str = 'R',
    trunc_rank: float = 1.0,
    min_out: int = 0,
    prefix: str = '',
    suffix: str|type[EOS] = EOS,
) -> str:
    return OutputSlot(name, is_label, max, min, trunc_side, trunc_rank, min_out, prefix, suffix) # noqa


slot_regex = re.compile(r" ?#\[(.*?)]#")


class TokenTemplate:
    tokenizer = None

    def __init__(self,
        *sequence: str | 'TokenTemplate' | 'TokenSlot' | T.Iterable[tuple[int, bool, bool]|'TokenSlot'],
        is_attended: bool = True,
        is_label: bool = False,
        max_length: int = None,
        pad_to_same_length: bool = True,
        pad_to_multiple_of: int = 8,
        pad_side: str = 'L',
        trunc_segment: bool = True,
        trunc_content: bool = True,
        tokenizer: PreTrainedTokenizer = None,
    ):
        self.sequence:list[tuple[int, bool, bool]|TokenSlot] = []
        self.slots:dict[str, TokenSlot] = {}
        self.tokenizer = type(self).tokenizer if tokenizer is None else tokenizer
        assert self.tokenizer is not None, "A tokenizer must be provided to TokenTemplate."
        self.pad_to_same_length = pad_to_same_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_side = pad_side
        self.trunc_segment = trunc_segment
        self.trunc_content = trunc_content
        self.max_length = max_length
        for sequence in sequence:
            self.extend(sequence, is_attended=is_attended, is_label=is_label)

    def extend(self,
        sequence: str | 'TokenTemplate' | 'TokenSlot' | T.Iterable[tuple[int, bool, bool]|'TokenSlot'],
        is_attended: bool = True,
        is_label: bool = False,
    ):
        extension = []
        added_slots = {}
        if isinstance(sequence, str):
            slots = slot_regex.finditer(sequence)
            i = 0
            for slot_args in slots:
                template_text = sequence[i:slot_args.start()]
                if template_text:
                    extension.extend(TokenSequence(
                        template_text, is_attended=is_attended, is_label=is_label, tokenizer=self.tokenizer))
                slot = TokenSlot(slot_args.group(0))
                added_slots[slot.name] = (slot, len(self)+len(extension))
                extension.append(slot)
                i = slot_args.end()
            template_text = sequence[i:]
            if template_text:
                extension.extend(TokenSequence(
                    template_text, is_attended=is_attended, is_label=is_label, tokenizer=self.tokenizer))
        elif isinstance(sequence, TokenSlot):
            added_slots[sequence.name] = (sequence, len(self)+len(extension))
            extension.append(sequence)
        else:
            extension.extend(sequence)
            added_slots.update({
                slot.name: (slot, len(self)+j)
                for j, slot in enumerate(sequence) if isinstance(slot, TokenSlot)})
        for slot_name, (slot, index) in added_slots.items():
            if slot_name in self.slots:
                raise ValueError(f"Slot {slot_name} already exists in TokenTemplate.")
            if slot.index is None:
                slot.index = index
            else:
                slot = cp.copy(slot)
                slot.index = index
                added_slots[slot_name] = (slot, index)
        self.slots.update(((slot_name, slot) for slot_name, (slot, _) in added_slots.items()))
        self.sequence.extend(extension)
        return self
    __iadd__ = extend

    def fill(self, slots: dict[str, str|TokenSequence]|T.Iterable[dict[str, str|TokenSequence]]):
        if not isinstance(slots, dict):
            return TokenSequences([self.fill(slot) for slot in slots],
                pad_to_same_length=self.pad_to_same_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_side=self.pad_side,
                tokenizer=self.tokenizer)
        assert all(slot in self.slots for slot in slots), \
            f"Slots {set(slots) - set(self.slots)} not found in TokenSequence."
        filled = []
        # get the prefix of this self template that contains slots about to be filled with given values
        end_of_filled = None
        prompt_slots = set()
        for slot_name, slot in self.slots.items():
            if slots.get(slot_name, object) in (None, Ellipsis):
                end_of_filled = slot.index
                break
            elif slot_name not in slots:
                raise ValueError(f"Filling {self} failed to provide value for slot {slot_name}.")
            else:
                prompt_slots.add(slot_name)
        template_prompt = self[slice(0, end_of_filled)]
        # convert slot values to TokenSequence object, keyed by slot name
        slot_subseqs = {}  # slot name, value seq
        template_prompt_slots = {}
        for slot_name, text_seq in slots.items():
            slot = self.slots[slot_name]
            if isinstance(text_seq, str):
                slot_suffix = slot.suffix if slot.suffix is not EOS else self.tokenizer.eos_token
                text_seq = ''.join((slot.prefix, text_seq, slot_suffix))
                text_seq = TokenSequence(text_seq, is_label=slot.is_label, tokenizer=self.tokenizer)
            else:
                prefix_seq = TokenSequence(slot.prefix, is_label=slot.is_label, tokenizer=self.tokenizer)
                prefix_seq.extend(text_seq)
                text_seq = prefix_seq
                if slot.suffix is EOS:
                    eos = ((self.tokenizer.eos_token_id, True, slot.is_label),)
                elif slot.suffix:
                    eos = TokenSequence(slot.suffix, is_label=slot.is_label, tokenizer=self.tokenizer)
                else:
                    eos = ()
                text_seq.extend(eos)
            slot_subseqs[slot_name] = text_seq
            if slot_name in prompt_slots:
                template_prompt_slots[slot_name] = slot
        # apply per-slot independent truncation rules based on per-slot max length
        for slot_name, subseq in slot_subseqs.items():
            slot = self.slots[slot_name]
            if slot.max and len(subseq) > slot.max:
                if slot.trunc_side == 'L':
                    slot_subseqs[slot_name] = subseq[-slot.max:]
                else:
                    slot_subseqs[slot_name] = subseq[:slot.max]
        # calculate whether/how much we need to truncate
        if end_of_filled is None:
            template_length = self.template_length()
        else:
            template_length = end_of_filled + self[end_of_filled].min_out
        current_length = sum(len(text_seq) for text_seq in slot_subseqs.values()) + template_length
        if self.max_length is not None and current_length > self.max_length:
            # truncate each slot value as much as possible (per-slot min is a floor) in order of trunc_rank until fit
            slot_trunc_candidates = sorted(slot_subseqs, key=lambda x: self.slots[x].trunc_rank)
            for candidate_slot_name in slot_trunc_candidates:
                subseq = slot_subseqs[candidate_slot_name]
                slot = self.slots[candidate_slot_name]
                amount_to_truncate = max(
                    min(current_length - self.max_length, len(subseq) - slot.min, len(subseq)), 0)
                if amount_to_truncate:
                    if slot.trunc_side == 'L':
                        slot_subseqs[candidate_slot_name] = subseq[amount_to_truncate:]
                    else:
                        slot_subseqs[candidate_slot_name] = subseq[:-amount_to_truncate]
                    current_length = sum(len(text_seq) for text_seq in slot_subseqs.values()) + template_length
                    if current_length <= self.max_length:
                        break
            else:  # nobreak
                raise ValueError(
                    f"Could not truncate slot text to fit within max_length {self.max_length} (max truncation was reached after sequence was cut down to {current_length} tokens).")
        # join together the final sequence
        previous_splitter = 0
        for slot_name, subseq in slot_subseqs.items():
            if slot_name in template_prompt_slots:
                splitter = self.slots[slot_name].index
                segment = template_prompt[previous_splitter:splitter]
                filled.extend(segment)
                filled.extend(subseq)
                previous_splitter = splitter + 1
        filled.extend(template_prompt[previous_splitter:])
        return TokenSequence(filled, tokenizer=self.tokenizer)

    def template_length(self):
        return len(self) - len(self.slots)

    def tokens(self):
        return [str(t) if isinstance(t, TokenSlot) else self.tokenizer.decode(t[0]) for t in self]

    def ansi(self):
        return TokenPrinter().ansi(self)

    def display(self):
        return (f"{ansi.bold}{self.__class__.__name__} "
                f"with {self.template_length()} tokens "
                f"and {len(self.slots)} slots:{ansi.reset}\n{self.ansi()}")

    def __iter__(self):
        return iter(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        if isinstance(item, slice):
            copy = cp.copy(self)
            copy.sequence = self.sequence[item]
            return copy
        else:
            return self.sequence[item]

    def __str__(self):
        tokens = (str(t) if isinstance(t, TokenSlot) else self.tokenizer.decode(t[0]) for t in self)
        if len(self) > 10:
            return f'<TokenTemplate len {len(self)}: {"|".join(it.islice(tokens, 10))}|...>' # noqa
        else:
            return f'<TokenTemplate len {len(self)}: {"|".join(tokens)}>' # noqa

    def __repr__(self):
        return f"TokenTemplate({repr(self.sequence)})"


class TokenTemplates:
    tokenizer = None

    def __init__(self,
        max_length: int|None = None,
        pad_to_same_length: bool = True,
        pad_to_multiple_of: int = 8,
        pad_side: str = 'L',
        trunc_segments_side: str = 'L',
        max_segments: int | None = None,
        tokenizer: str | PreTrainedTokenizer = None,
    ):
        self.templates: dict[str, TokenTemplate] = {}
        self.max_length = max_length
        self.pad_to_same_length = pad_to_same_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_side = pad_side
        self.trunc_segments_side = trunc_segments_side
        self.max_segments = max_segments
        self.tokenizer = type(self).tokenizer if tokenizer is None else tokenizer
        assert self.tokenizer is not None, "A tokenizer must be provided to TokenTemplates."

    def add(self, templates=None, /, **templates_):
        templates = (templates or {}) | templates_
        for name, template in templates.items():
            assert name not in self.templates, f"Template name {name} already exists in TokenTemplates."
            self.templates[name] = template

    def fill(self,
        segments: T.Iterable[T.Iterable[dict[str, str|TokenSequence]]]
                  |T.Iterable[dict[str, str|TokenSequence]]
    ):
        first, segments = peek(segments)
        if not isinstance(first, dict):
            return TokenSequences([self.fill(seg) for seg in segments],
                pad_to_same_length=self.pad_to_same_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_side=self.pad_side,
                tokenizer=self.tokenizer)
        assert all('temp' in seg for seg in segments), \
            "All segments must have a 'temp' key to define which template to use."
        # Get the template corresponding to each segment
        try: templates = [self.templates[seg['temp']] for seg in segments]
        except KeyError as e: raise ValueError(f"Template {e} not found in TokenTemplates.")
        # Truncate entire segments based on max_segments
        if self.max_segments is not None and len(templates) > self.max_segments:
            num_segs_need_to_delete = len(templates) - self.max_segments
            segs_to_delete = set()
            if self.trunc_segments_side == 'R':
                seg_iter = reversed(list(enumerate(templates)))
            else:
                seg_iter = enumerate(templates)
            for i, template in seg_iter:
                if template.trunc_segment:
                    segs_to_delete.add(i)
                    num_segs_need_to_delete -= 1
                if num_segs_need_to_delete == 0:
                    break
            else:
                raise ValueError(f"Could not truncate any segment to fit within max_segments {self.max_segments}.")
            templates = [template for i, template in enumerate(templates) if i not in segs_to_delete]
            segments = [segment for i, segment in enumerate(segments) if i not in segs_to_delete]
        # Convert all segment values to TokenSequence objects
        segments_no_temp = []
        for segment in segments:
            seg_no_temp = {}
            for slot_name, slot_value in segment.items():
                if slot_name != 'temp' and slot_value not in (None, Ellipsis):
                    seg_no_temp[slot_name] = TokenSequence(slot_value, tokenizer=self.tokenizer)
            segments_no_temp.append(seg_no_temp)
        segments = segments_no_temp
        # Caclulate the minimum possible length of each segment after filling its template
        template_min_lengths = []
        for i, (template, segment_values) in enumerate(zip(templates, segments)):
            template_length = template.template_length()
            values_length = 0
            for j, (slot_name, slot) in enumerate(template.slots.items()):
                if slot_name in segment_values:
                    if template.trunc_content:
                        values_length += min(len(segment_values[slot_name]), slot.min or 0)
                    else:
                        values_length += len(segment_values[slot_name]) + len(slot.eos_tokens or ())
                else:
                    template_length = slot.index - j
                    template_min_lengths.append(template_length + values_length + slot.min_out)
                    break
            else:
                template_min_lengths.append(template_length + values_length)
                continue
            break
        # Find which segments must be removed to satisfy the max_length constraint
        total_min_length = sum(template_min_lengths)
        if self.max_length is not None and total_min_length > self.max_length:
            deleted = set()
            if self.trunc_segments_side == 'R':
                templates_values_minlens = reversed(list(enumerate(zip(templates, segments, template_min_lengths))))
            else:
                templates_values_minlens = enumerate(zip(templates, segments, template_min_lengths))
            for i, (template, segment_values, template_min_length) in templates_values_minlens:
                if self.max_length is None or total_min_length < self.max_length:
                    break
                if template.trunc_segment:
                    deleted.add(i)
                    total_min_length -= template_min_length
            else:
                raise ValueError(f"Could not truncate any segment to fit within max_length {self.max_length}.")
            templates_and_values = [
                (template, segment_values)
                for i, (template, segment_values, _) in enumerate(zip(templates, segments, template_min_lengths))
                if i not in deleted]
        else:
            templates_and_values = list(zip(templates, segments))
        # Build a megatemplate by appending the segment index to each slot name
        total_template = TokenTemplate(
            max_length=self.max_length,
            pad_to_same_length=False,
            pad_to_multiple_of=1,
            tokenizer=self.tokenizer)
        total_values = {}
        for i, (template, values) in enumerate(templates_and_values):
            offset = len(total_template)
            total_template += template
            for slot_name, slot in template.slots.items():
                i_slot_name = f"{slot_name}__{i}"
                total_template.slots[i_slot_name] = dc.replace(slot,
                    name=i_slot_name,
                    index=offset + slot.index,
                    trunc_rank=slot.trunc_rank if template.trunc_content else float('+inf'))
                if slot_name in values:
                    total_values[i_slot_name] = values[slot_name]
        # Fill the megatemplate with the segment values
        total_filled = total_template.fill(total_values)
        return total_filled

    def __str__(self):
        return f'<TokenTemplates with {len(self.templates)} templates>'

    def __repr__(self):
        return f"TokenTemplates({repr(self.templates)})"

    def display(self):
        return '\n\n'.join(
            f"{ansi.bold}{name}{ansi.reset}:\n{template.display()}"
            for name, template in self.templates.items())


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

    def ansi(self, tokens: TokenSequence|TokenSequences|TokenTemplate):
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


############################################################################################



'''Library'''

Slot = str | TokenSlot

@dc.dataclass
class Template:
    trunc_content = True
    trunc_segment = False

    def __call__(self, *args, **kwargs):
        return self.__class__(*args, **kwargs) # noqa

@dc.dataclass
class Templates:
    max_length: int = None
    pad_to_same_length: bool = True
    pad_to_multiple_of: int = 8
    pad_side: str = 'L'
    trunc_segments_side: str = 'L'
    max_segments: int | None = None

    def __post_init__(self):
        for name, field in dc.asdict(self):
            if issubclass(field, Template):
                setattr(self, name, field())
            if isinstance(field, Template):
                for slot_name, slot in dc.asdict(field):
                    assert isinstance(slot, Slot), \
                    f"When constructing a Templates collection, every template must be a Template object with Slot objects for all fields. However, got {slot} for slot {slot_name} in template {name}: {field}."

    def fill(self, values: T.Iterable[Template]) -> TokenSequence:
        ...

    def fill_batch(self, values: T.Iterable[T.Iterable[Template]]) -> TokenSequences:
        ...

    def batch(self, values: T.Iterable[T.Iterable[Template]], batch_size: int) -> list[TokenSequences]:
        ...


@dc.dataclass
class SystemTemplate(Template):
    template = "<|start_header_id|>system<|end_header_id|>\n\n{prompt}\n\n{date}<|eot_id|>"
    prompt: Slot = InputSlot(); date: Slot = InputSlot()

@dc.dataclass
class UserTemplate:
    template = "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>"
    input: Slot = InputSlot()

@dc.dataclass
class LlamaTemplates(Templates):
    system: SystemTemplate = SystemTemplate
    user: UserTemplate = UserTemplate


'''Approach'''

@dc.dataclass
class MyDSTTemplate:
    template = UserTemplate(input="{slot}: {description}")
    slot: Slot = InputSlot(); description: Slot = InputSlot(max=128)

@dc.dataclass
class MyLlamaTemplates(LlamaTemplates):
    dst: MyDSTTemplate = MyDSTTemplate

def experiment():
    my_templates = MyLlamaTemplates(
        system=SystemTemplate(prompt=InputSlot(min=64, max=128)),
        dst=MyDSTTemplate(description=InputSlot(max=64)),
        max_length=512,
        pad_to_multiple_of=16,
    )

    my_dialogue = [
        MyLlamaTemplates.system('track the dialogue', date='oct 8'),
        MyLlamaTemplates.user('I went to the station at 5'),
        MyLlamaTemplates.dst(slot='time', description='the time'),
        my_templates.system(prompt='track the above dialogue please')
    ]

    context_tokens = my_templates.fill(my_dialogue)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
    tokenizer.pad_token = '-'
    tokenizer.pad_token_id, = tokenizer.encode('-', add_special_tokens=False)

    class LlamaTemplate(TokenTemplate): tokenizer = tokenizer

    template = LlamaTemplate(
        f"<|start_header_id|>system<|end_header_id|>\n\n{InputSlot('instruction', max=128)}<|eot_id|>")

    print(template.display(), '\n')
    tokens = template.fill([
        dict(input='sentence with many words', output='OK!'),
        dict(input='short', output='short'),
        dict(input='a really really long sentence with many words', output='OK!'),
    ])
    for token in tokens:
        print(token.display(), '\n')