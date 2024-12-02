from __future__ import annotations

import dataclasses as dc
import itertools as it
import copy as cp
import re

import ezpyzy as ez

from language_model.tokens.tokenizer import Tokenizer
from language_model.tokens.token_sequence import TokenSequence
from language_model.tokens.token_sequences import TokenSequences
from language_model.tokens.template import (
    Slot, Input, Output, TokenSlot, OutputSlot, Template, SegmentTemplate, Templates)

import typing as T

default: T.Any = object()

def fields(cls_or_instance) -> list[dc.Field]: return dc.fields(cls_or_instance) # noqa


@dc.dataclass
class TemplateTokenizerConfig(ez.Config):
    templates: Templates = Templates()
    """Config of a Templates subclass that defines the templates that can be used for tokenization."""
    max_length: int | None = None
    """The maximum token length of sequences the model trains on or can be fed as input for generation."""
    pad_to_same_length: bool = True
    """Whether to pad sequences to the same length within each batch."""
    pad_side: str = 'L'
    """The side to pad sequences on. 'L' for left, 'R' for right."""
    pad_to_multiple_of: int = 8
    """Pads sequences so that total token_ids lengths are a multiple of this value, which can improve performance on GPU."""
    max_segments: int | None = None
    """The maximum number of segments to keep in a token_ids. If None, no segment pruning will be performed. The primary use case for setting max_segments is to trim extremely long sequences by a number-of-segments threshold BEFORE any tokenization is performed, which can improve preprocessing efficiency."""
    sequence_prefix: str = '{bos}'
    """The prefix to add to the beginning of each sequence that gets tokenized."""
    sequence_suffix: str = ''
    """The suffix to add to the end of each sequence that gets tokenized."""
    tokenizer: Tokenizer = None
    """The tokenizer to use for tokenization. This should be an instance of a class inheriting from Tokenizer, such as a HuggingfaceTokenizer."""
    max_out: int | None = None
    """Global setting for the maximum number of tokens that are expected to be generated."""


@dc.dataclass
class TemplateTokenizer(ez.ImplementsConfig, TemplateTokenizerConfig):

    def __post_init__(self):
        super().__post_init__()
        with self.configured.not_configuring():
            for special_symbol, replacement in self.tokenizer.slot_affix_replacements.items():
                self.sequence_prefix = self.sequence_prefix.replace(f'{{{special_symbol}}}', replacement)
                self.sequence_suffix = self.sequence_suffix.replace(f'{{{special_symbol}}}', replacement)
        self.sequence_prefix_tokens = TokenSequence(self.sequence_prefix, tokenizer=self.tokenizer)
        self.sequence_suffix_tokens = TokenSequence(self.sequence_suffix, tokenizer=self.tokenizer)
        self.templates_tokens: dict[str, TokenSequence] = {}
        self.slot_trunc_text_tokens: dict[tuple[str, str], TokenSequence] = {}
        self.template_name_map: dict[str, SegmentTemplate] = {}
        self.template_slots: dict[str, list[TokenSlot]] = {}
        self.template_slots_eos: dict[tuple[str, int], int] = {}
        for _, template in self.templates:
            if not isinstance(template, SegmentTemplate):
                continue
            template_name = template.name
            template_text = template.template
            for special_symbol, replacement in self.tokenizer.slot_affix_replacements.items():
                template_text = template_text.replace(f'{{{special_symbol}}}', replacement)
            slot_pattern = re.compile(
                f"(?P<slot_lead>{self.tokenizer.slot_lead_pattern})" +
                r"<(?P<slot_name>[a-zA-Z_][a-zA-Z_0-9]*)>" +
                f"(?P<slot_trail>{self.tokenizer.slot_trail_pattern})")
            previous_end = 0
            template_parts = []
            slots = []
            for slot_match in slot_pattern.finditer(template_text):
                slot_name = slot_match.group('slot_name')
                slot_lead = slot_match.group('slot_lead')
                slot_trail = slot_match.group('slot_trail')
                if slot_name in template.slots and isinstance(slot:=getattr(template.slots, slot_name), TokenSlot):
                    start, end = slot_match.span()
                    template_parts.append(template_text[previous_end:start])
                    with slot.configured.not_configuring():
                        slot.prefix = slot_lead + slot.prefix
                        slot.suffix = slot.suffix + slot_trail
                        for special_symbol, replacement in self.tokenizer.slot_affix_replacements.items():
                            slot.prefix = slot.prefix.replace(f'{{{special_symbol}}}', replacement)
                            slot.suffix = slot.suffix.replace(f'{{{special_symbol}}}', replacement)
                            slot.trunc_text = slot.trunc_text.replace(f'{{{special_symbol}}}', replacement)
                    slots.append(slot)
                    previous_end = end
            template_suffix = template_text[previous_end:]
            template_tokens = TokenSequence(
                '', is_attended=template.is_attended, is_label=template.is_label, tokenizer=self.tokenizer)
            self.template_slots[template_name] = []
            for i, (template_part, slot) in enumerate(zip(template_parts, slots)):
                template_tokens += template_part
                slot = +slot
                slot.token_index = len(template_tokens)
                slot.slot_index = i
                slot.template = template
                slot_trunc_tokens = TokenSequence(slot.trunc_text,
                    is_attended=template.is_attended, is_label=False, tokenizer=self.tokenizer)
                self.slot_trunc_text_tokens[(template_name, slot.name)] = slot_trunc_tokens
                if isinstance(slot.max, float):
                    assert slot.max > slot.min + len(slot.trunc_text), \
                        f"Slot {slot.name} has a max value length {slot.max} shorter than the sum of its min value length {slot.min} (plus length of truncation_text tokens - {repr(slot.trunc_text)})."
                slot_eos = None
                if slot.suffix:
                    slot_suffix_tokens = self.tokenizer.encode(slot.suffix)
                    if slot_suffix_tokens:
                        slot_eos = slot_suffix_tokens[-1]
                self.template_slots_eos[template_name, len(self.template_slots[template_name])] = slot_eos
                self.template_slots[template_name].append(slot)
            template_tokens += template_suffix
            self.templates_tokens[template_name] = template_tokens
            self.template_name_map[template_name] = template

    def find_gen_slot(self,
        data_item: list[Template|dict[str,str]]
    ) -> tuple[int, TokenSlot, int] | None:
        """
        Find the first slot in the input sequence that is marked for generation.

        template_name, segment_index, slot_index, slot
        """
        for i, segment in enumerate(data_item):
            if isinstance(segment, dict):
                values = segment.values()
            else:
                values = segment.__dict__.values()
            for j, value in enumerate(values):
                if value is Ellipsis:
                    temp = segment['temp'] if isinstance(segment, dict) else type(segment).__name__
                    slot_for_generation = self.template_slots[temp][j]
                    if self.max_out is None and slot_for_generation.max is None:
                        num_expected_out_tokens = None
                    elif self.max_out is None:
                        num_expected_out_tokens = slot_for_generation.max
                    elif slot_for_generation.max is None:
                        num_expected_out_tokens = self.max_out
                    else:
                        num_expected_out_tokens = min(self.max_out, slot_for_generation.max)
                    return i, self.template_slots[temp][j], num_expected_out_tokens


    def _tokenize_sequence(self,
        data: list[Template | dict[str, str]],
        gen_slot: tuple[int, TokenSlot, int]|None
    ) -> TokenSequences:

        # truncate based on slot-for-generation
        slot_for_generation = None
        index_of_slot_for_generation = None
        index_of_segment_for_generation = None
        num_expected_out_tokens = 0
        if gen_slot is not None:
            index_of_segment_for_generation, slot_for_generation, num_expected_out_tokens = gen_slot
            index_of_slot_for_generation = slot_for_generation.slot_index

        # get templates with corresponding values
        templates: list[SegmentTemplate] = []
        segs_slots: list[list[TokenSlot]] = []
        segs_value_dicts: list[dict[str, str]] = []
        templates_tokens: list[TokenSequence] = []

        for i, segment_values in enumerate(data):
            if isinstance(segment_values, dict):
                temp_name = segment_values['temp']
                segment_values = {k: v for k, v in segment_values.items() if k != 'temp'}
            else:
                temp_name = type(segment_values).__name__
                segment_values = dict(segment_values.__dict__)
            assert temp_name in self.templates_tokens, \
                f"Got a template named {temp_name} but it is not a template in {self.templates}"
            template = self.template_name_map[temp_name]
            segment_slots = self.template_slots[temp_name]
            if i == index_of_segment_for_generation:
                segment_slots = segment_slots[:index_of_slot_for_generation]
                temp_prefix = self.templates_tokens[temp_name][:slot_for_generation.token_index] # noqa
                templates_tokens.append(temp_prefix)
            else:
                templates_tokens.append(self.templates_tokens[temp_name])
            assert set(segment_values).issuperset({slot.name for slot in segment_slots}), \
                f"Segment values {set(segment_values)} for template {temp_name} do not match the template slots {set(slot_name for slot_name, _ in template.slots)}"
            templates.append(template)
            segs_slots.append(segment_slots)
            segs_value_dicts.append(segment_values) # noqa
            if i == index_of_segment_for_generation:
                break

        # create segments table where original templates/segs_value_dicts indices are IDs
        segments_table = {i: (template, segment_values, segment_slots)
            for i, (template, segment_values, segment_slots)
            in enumerate(zip(templates, segs_value_dicts, segs_slots))}
        (_segments_table_template, _segments_table_values, _segments_table_slots
        ) = 0, 1, 2

        # sort segments by trunc_rank, trunc_side, and truncability (and filter by truncability)
        seg_trunc_cands = dict(ez.sort([(i, (template, segment_values, segment_slots))
                for i, (template, segment_values, segment_slots) in segments_table.items()
                if template.trunc_segment and i != index_of_segment_for_generation],
            by=[(-template.trunc_segment_rank, i if template.trunc_segment_side == 'L' else -i)
                for i, (template, _, _) in segments_table.items()
                if template.trunc_segment and i != index_of_segment_for_generation]))
        (_seg_trunc_cands_template, _seg_trunc_cands_values, _seg_trunc_cands_slots
        ) = 0, 1, 2

        # truncate segments based on max sequences
        if self.max_segments and len(segments_table) > self.max_segments:
            for i in it.islice(seg_trunc_cands, len(segments_table) - self.max_segments):
                del segments_table[i]
                del seg_trunc_cands[i]

        # tokenize all value sequences in remaining segments
        segs_value_seqs: dict[int, tuple[SegmentTemplate, list[TokenSequence], list[TokenSlot]]] = {}
        for i, (template, segment_values, segment_slots) in segments_table.items():
            seg_value_seqs = []
            for slot in segment_slots:
                value_text = ''.join((slot.prefix, segment_values[slot.name], slot.suffix))
                value_seq = TokenSequence(value_text,
                    is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer)
                seg_value_seqs.append(value_seq)
            segs_value_seqs[i] = (template, seg_value_seqs, segment_slots)
        (_segs_val_seqs_template, _segs_val_seqs_seqs, _segs_val_seqs_slots
        ) = 0, 1, 2

        # compute min and max tokens of each slot
        slot_value_table = {}  # (seg_idx, slot_idx) -> (slot, value, min, max)
        for i, (template, seg_value_seqs, seg_slots) in segs_value_seqs.items():
            for j, (slot, seq) in enumerate(zip(seg_slots, seg_value_seqs)):
                if slot.trunc and template.trunc_content:
                    trunc_text_len = len(self.slot_trunc_text_tokens[(template.name, slot.name)])
                    min_tokens = min(slot.min + trunc_text_len, len(seq))
                    max_tokens = max(
                        min(len(seq), len(seq) if slot.max is None else slot.max), min_tokens)
                else:
                    max_tokens = len(seq)
                    min_tokens = len(seq)
                slot_value_table[(i, j)] = (slot, seq, min_tokens, max_tokens)
        _slot_value_table_slot, _slot_value_table_seq = 0, 1
        _slot_value_table_min, _slot_value_table_max = 2, 3

        # sort slots by trunc_rank, trunc_side, and truncability (and filter by truncability)
        slot_trunc_cands = dict(
            ez.sort(
                [
                    ((i, j), [slot, seq, min_tokens, max_tokens])
                    for (i, j), (slot, seq, min_tokens, max_tokens) in slot_value_table.items()
                    if min_tokens < max_tokens
                ], by=[
                    (
                        -slot.trunc_rank,
                        (i if slot.template.trunc_segment_side == 'L' else -i),
                        (j if slot.trunc_side == 'L' else -j)
                    )
                    for (i, j), (slot, _, min_tokens, max_tokens) in slot_value_table.items()
                    if min_tokens < max_tokens
                ]
            )
        )
        _slot_trunc_cands_slot, _slot_trunc_cands_seq = 0, 1
        _slot_trunc_cands_min, _slot_trunc_cands_max = 2, 3

        # calculate current total length if we truncated and merged all segments as-is
        current_len = ((num_expected_out_tokens or 0)
                       + len(self.sequence_prefix_tokens)
                       + (len(self.sequence_suffix_tokens) if slot_for_generation is None else 0)
                       + sum(max_tokens for _, _, _, max_tokens in slot_value_table.values())
                       + sum(len(temp) for temp in templates_tokens)
                       )
        # register truncations for slots and segments
        slots_with_content_trunc = []
        iter_seg_trunc_cands = iter(list(seg_trunc_cands.items()))
        iter_slot_trunc_cands = iter(list(slot_trunc_cands.items()))  # noqa
        next_seg_to_trunc = next(iter_seg_trunc_cands, None)
        next_slot_to_trunc = next(iter_slot_trunc_cands, None)
        while self.max_length and current_len > self.max_length:
            amount_to_trunc = current_len - self.max_length
            do_trunc_slot = False
            do_trunc_seg = False
            if next_seg_to_trunc is None and next_slot_to_trunc is None:
                raise ValueError(f"Current length {current_len} exceeds max length {self.max_length} but no more truncation candidates are available.")
            elif next_slot_to_trunc is None:
                do_trunc_seg = True
            elif next_seg_to_trunc is None:
                do_trunc_slot = True
            else:
                (_, (template, _, _)) = next_seg_to_trunc
                (_, _), (slot, _, _, _) = next_slot_to_trunc
                if slot.trunc_rank >= template.trunc_segment_rank:
                    do_trunc_slot = True
                else:
                    do_trunc_seg = True
            if do_trunc_slot:
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                trunc_amount = min(amount_to_trunc, max_tokens - min_tokens)
                trunc_text_len = len(self.slot_trunc_text_tokens[slot.template.name, slot.name])
                if i in segs_value_seqs:
                    current_len -= trunc_amount
                    new_value_len = max_tokens - trunc_amount
                    slot_value_table[(i, j)] = (slot, seq, min_tokens, new_value_len)
                    slots_with_content_trunc.append(((i, j), slot_value_table[(i, j)]))  # noqa
                    if (new_value_len <= trunc_text_len and slot.template.trunc_segment_if_no_content
                        and i != index_of_segment_for_generation
                    ):
                        all_slots_in_segment = segs_value_seqs[i][_segs_val_seqs_slots]
                        for other_j, other_slot in enumerate(all_slots_in_segment):
                            max_other_slot = slot_value_table[(i, other_j)][_slot_value_table_max]
                            other_slot_trunc_text_len = len(
                                self.slot_trunc_text_tokens[other_slot.template.name, slot.name])
                            if max_other_slot > other_slot_trunc_text_len:
                                break
                        else: # no break
                            if len(segs_value_seqs) == 1:
                                raise ValueError(f"Current length {current_len} exceeds max length {self.max_length} but no more truncation candidates are available.")
                            current_len -= len(templates_tokens[i])
                            for other_j in range(len(all_slots_in_segment)):
                                current_len -= slot_value_table[(i, other_j)][_slot_value_table_max]
                            del segs_value_seqs[i]
                next_slot_to_trunc = next(iter_slot_trunc_cands, None)
            elif do_trunc_seg:
                i, (template, seg_value_seqs, _) = next_seg_to_trunc
                if i in segs_value_seqs and len(segs_value_seqs) > 1 and i != index_of_segment_for_generation:
                    current_len -= len(templates_tokens[i])
                    for j in range(len(seg_value_seqs)):
                        current_len -= slot_value_table[(i, j)][_slot_value_table_max]
                    del segs_value_seqs[i]
                next_seg_to_trunc = next(iter_seg_trunc_cands, None)

        # recover some tokens from truncation
        for (i, j), (slot, seq, min_tokens, max_tokens) in reversed(slots_with_content_trunc):
            if current_len >= self.max_length: break
            if i not in segs_value_seqs: continue
            left_to_recover = self.max_length - current_len
            slot_can_recover = len(seq) - max_tokens
            if slot.max is not None:
                slot_can_recover = min(slot_can_recover, slot.max - max_tokens)
            recover_amount = min(left_to_recover, slot_can_recover)
            current_len += recover_amount
            slot_value_table[(i, j)] = (slot, seq, min_tokens, max_tokens + recover_amount)

        # create final token sequence
        final_seq = TokenSequence(self.sequence_prefix_tokens or '', tokenizer=self.tokenizer)
        for i, (template, seg_value_seqs, _) in segs_value_seqs.items():
            previous_slot_index = 0
            for j, value_seq in enumerate(seg_value_seqs):
                slot, _, min_tokens, max_tokens = slot_value_table[(i, j)]
                final_seq += templates_tokens[i][previous_slot_index:slot.token_index]
                previous_slot_index = slot.token_index
                if len(value_seq) > max_tokens:
                    trunc_tokens = self.slot_trunc_text_tokens[(template.name, slot.name)]
                    if slot.trunc_side == 'L':
                        final_seq += trunc_tokens
                        trunc_index = len(trunc_tokens) - max_tokens
                        if trunc_index < 0:
                            final_seq += value_seq[trunc_index:]
                    else:
                        final_seq += value_seq[:max_tokens - len(trunc_tokens)]
                        final_seq += self.slot_trunc_text_tokens[(template.name, slot.name)]
                else:
                    final_seq += value_seq
            final_seq += templates_tokens[i][previous_slot_index:]
        if self.sequence_suffix_tokens and slot_for_generation is not None:
            final_seq += self.sequence_suffix_tokens
        return final_seq

    def _tokenize_sequences(self,
        data: T.Iterable[list[Template | dict[str, str]]],
        gen_slots: list[tuple[int, TokenSlot, int]|None]
    ) -> TokenSequences:
        tokenized_sequences = []
        for sequence, gen_slot in zip(data, gen_slots):
            tokenized_sequence = self._tokenize_sequence(sequence, gen_slot)
            tokenized_sequences.append(tokenized_sequence)
        tokenized_sequences = TokenSequences(
            tokenized_sequences,
            pad_to_same_length=self.pad_to_same_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            pad_side=self.pad_side,
            tokenizer=self.tokenizer)
        return tokenized_sequences

    def tokenize(self,
        data: list[Template | dict[str, str]] | T.Iterable[list[Template | dict[str, str]]]
    ) -> TokenSequence|TokenSequences:
        if not isinstance(data, list) or isinstance(data[0], list):
            gen_slots = [self.find_gen_slot(data_item) for data_item in data]
            token_sequences = self._tokenize_sequences(data, gen_slots)
            return token_sequences
        else:
            gen_slot = self.find_gen_slot(data)
            token_sequence = self._tokenize_sequence(data, gen_slot)
            return token_sequence




if __name__ == '__main__':

    from language_model.tokens.tokenizer import HuggingfaceTokenizer

    @dc.dataclass
    class MyTemplate(Template):
        template = 'This is a <adjective> <noun>. The <noun> is <phrase>!\n\n'
        adjective: Slot = Input()
        noun: Slot = Input()
        phrase: Slot = Output()
        
    @dc.dataclass
    class MyTurn(Template):
        template = '<speaker> says, "<quote>"'
        speaker: Slot = Input()
        quote: Slot = Input()
        

    @dc.dataclass
    class MyTemplates(Templates):
        my_template: SegmentTemplate = SegmentTemplate(template=MyTemplate())
        my_turn: SegmentTemplate = SegmentTemplate(template=MyTurn())

    ########################################################################################.
    
    tokenizer = TemplateTokenizer(
        templates=MyTemplates(
            my_template=SegmentTemplate(
                template=MyTemplate(adjective=Input(max=24, min=16), phrase=Input()),
                trunc_segment_rank=3.0
            ),
            my_turn=SegmentTemplate(
                template=MyTurn(speaker=Input(), quote=Output()),
                trunc_segment_rank=1.5
            )
        ),
        max_length=128,
        pad_to_same_length=True,
        tokenizer=HuggingfaceTokenizer(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct')
    )
    
    data = [
        [
            MyTemplate(adjective='big', noun='dog', phrase='happy'),
            MyTurn(speaker='Alice', quote=...)
        ],
        [
            MyTemplate(adjective='small', noun='cat', phrase='sad'),
            MyTurn(speaker='Bob', quote='Goodbye, this is a test!')
        ]
    ]

    # print(tokenizer.configured.json())

    seqs = tokenizer.tokenize(data)

    for seq in seqs:
        print('|'.join(seq.tokens()))
        print('\n\n')




