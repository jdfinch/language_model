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
    Slot, Input, Output, TokenSlot, OutputSlot, Template, SegmentTemplate)

import typing as T

default: T.Any = object()

def fields(cls_or_instance) -> list[dc.Field]: return dc.fields(cls_or_instance) # noqa


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
                    slots.append(slot)
                    previous_end = end
            template_suffix = template_text[previous_end:]
            template_tokens = TokenSequence(
                '', is_attended=template.is_attended, is_label=template.is_label, tokenizer=self.tokenizer)
            self.template_slots[template_name] = []
            for i, (template_part, slot) in enumerate(zip(template_parts, slots)):
                template_tokens += template_part
                slot = cp.deepcopy(slot)
                slot.token_index = len(template_tokens)
                slot.slot_index = i
                slot.template = template
                slot_trunc_text = TokenSequence(slot.trunc_text,
                    is_attended=template.is_attended, is_label=slot.is_label, tokenizer=self.tokenizer)
                self.slot_trunc_text_tokens[(template_name, slot.name)] = slot_trunc_text
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
    ) -> tuple[str, int, int, TokenSlot] | None:
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
                    return temp, i, j, self.template_slots[temp][j]

    def _tokenize_sequence(self,
        data: list[Template | dict[str, str]],
        gen_slot: tuple[str, int, int, TokenSlot]|None
    ) -> TokenSequences:

        # get templates with corresponding values
        template_segments_values: list[tuple[SegmentTemplate, dict[str, str]]] = []
        for segment_values in data:
            if isinstance(segment_values, dict):
                temp_name = segment_values['temp']
                segment_values = {k: v for k, v in segment_values.items() if k != 'temp'}
                template = self.template_name_map[temp_name]
            else:
                temp_name = type(segment_values).__name__
                segment_values = dict(segment_values.__dict__)
                template = self.template_name_map[temp_name]
            assert set(segment_values) == {slot.name for slot in self.template_slots[temp_name]}, \
                f"Segment values {set(segment_values)} for template {temp_name} do not match the template slots {set(slot_name for slot_name, _ in template.slots)}"
            template_segments_values.append((template, segment_values))
        templates: list[SegmentTemplate] = [
            template for template, _ in template_segments_values]
        segs_value_dicts: list[dict[str, str]] = [
            segment_values for _, segment_values in template_segments_values]
        templates_slots: dict[str, list[TokenSlot]] = self.template_slots
        templates_tokens: dict[str, TokenSequence] = self.templates_tokens

        # find slot for generation
        slot_for_generation = None
        index_of_slot_for_generation = None
        index_of_segment_for_generation = None
        num_expected_out_tokens = 0
        if gen_slot is not None:
            (   template_name,
                index_of_segment_for_generation,
                index_of_slot_for_generation,
                slot_for_generation
            ) = gen_slot
            num_expected_out_tokens = slot_for_generation.max

        # truncate segments after segment with output, and slots from segment with output
        if index_of_segment_for_generation is not None:
            segs_value_dicts = segs_value_dicts[:index_of_segment_for_generation + 1]
            templates = templates[:index_of_segment_for_generation + 1]
            template_with_generation = templates[index_of_segment_for_generation]
            templates_slots = cp.copy(templates_slots)
            templates_tokens = cp.copy(templates_tokens)
            gen_tmp_name = template_with_generation.name
            template_slots = templates_slots[gen_tmp_name]
            temp_tokens = templates_tokens[gen_tmp_name]
            templates_tokens[gen_tmp_name] = temp_tokens[:template_slots[index_of_slot_for_generation].token_index]
            templates_slots[gen_tmp_name] = template_slots[:index_of_slot_for_generation]

        # create segments table where original templates/segs_value_dicts indices are IDs
        segments_table = {i: (template, segment_values)
            for i, (template, segment_values) in enumerate(zip(templates, segs_value_dicts))}
        _segments_table_template, _segments_table_values = 0, 1

        # sort segments by trunc_rank, trunc_side, and truncability (and filter by truncability)
        seg_trunc_cands = dict(
            ez.sort(
                [(i, (template, segment_values))
                    for i, (template, segment_values) in segments_table.items()
                    if template.trunc_segment and i != index_of_segment_for_generation],
                by=[(-template.trunc_segment_rank, i if template.trunc_segment_side == 'L' else -i)
                    for i, (template, _) in segments_table.items()
                    if template.trunc_segment and i != index_of_segment_for_generation]
            )
        )
        _seg_trunc_cands_template, _seg_trunc_cands_values = 0, 1

        # truncate segments based on max sequences
        if self.max_segments and len(segments_table) > self.max_segments:
            for i in it.islice(seg_trunc_cands, len(segments_table) - self.max_segments):
                del segments_table[i]
                del seg_trunc_cands[i]

        # tokenize all value sequences in remaining segments
        segs_value_seqs = {i: (template, list[TokenSequence]()) for i, (template, _) in segments_table.items()}
        for i, (template, segment_values) in segments_table.items():
            for slot in templates_slots[template.name]:
                value_text = ''.join((slot.prefix, segment_values[slot.name], slot.suffix))
                value_seq = TokenSequence(
                    value_text,
                    is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer
                )
                segs_value_seqs[i][1].append(value_seq)

        # compute min and max tokens of each slot
        slot_value_table = {}  # (seg_idx, slot_idx) -> (slot, value, min, max)
        for i, (template, seg_value_seqs) in segs_value_seqs.items():
            for j, (slot, seq) in enumerate(zip(templates_slots[template.name], seg_value_seqs)):
                if slot.trunc and template.trunc_content:
                    trunc_text_len = len(self.slot_trunc_text_tokens[(template.name, slot.name)])
                    min_tokens = min(slot.min + trunc_text_len, len(seq))
                    max_tokens = max(min(len(seq), len(seq) if slot.max is None else slot.max), min_tokens)
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
        current_len = (num_expected_out_tokens
                       + len(self.sequence_prefix_tokens)
                       + (len(self.sequence_suffix_tokens) if slot_for_generation is None else 0)
                       + sum(max_tokens for _, _, _, max_tokens in slot_value_table.values())
                       + sum(len(templates_tokens[template.name]) for template, _ in segs_value_seqs.values()))

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
                raise ValueError(
                    f"Current length {current_len} exceeds max length {self.max_length} but no more truncation candidates are available."
                )
            elif next_slot_to_trunc is None:
                if len(segments_table) > 1:
                    do_trunc_seg = True
                else:
                    raise ValueError(
                        f"Current length {current_len} exceeds max length {self.max_length} but no more truncation candidates are available."
                    )
            elif next_seg_to_trunc is None:
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                if i in segs_value_seqs and amount_to_trunc >= len(
                    self.slot_trunc_text_tokens[(templates[i].name, slot.name)]
                ):
                    do_trunc_slot = True
                else:
                    next_slot_to_trunc = next(iter_slot_trunc_cands, None)
            else:
                (_, (template, seg_value_seqs)) = next_seg_to_trunc
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                if (i in segments_table and amount_to_trunc >= len(
                    self.slot_trunc_text_tokens[(templates[i].name, slot.name)]
                ) and
                    slot.trunc_rank >= template.trunc_segment_rank
                ):
                    do_trunc_slot = True
                else:
                    do_trunc_seg = True
            if do_trunc_slot:
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                if i in segs_value_seqs:
                    trunc_amount = min(amount_to_trunc, max_tokens - min_tokens)
                    current_len -= trunc_amount
                    new_value_len = max_tokens - trunc_amount
                    slot_value_table[(i, j)] = (slot, seq, min_tokens, new_value_len)
                    slots_with_content_trunc.append(((i, j), slot_value_table[(i, j)]))  # noqa
                    if new_value_len <= 0 and slot.template.trunc_segment_if_no_content:
                        for j, other_slot in enumerate(templates_slots[j]):
                            max_other_slot = slot_value_table[(i, j)][_slot_value_table_max]
                            if max_other_slot > 0:
                                break
                        else: # no break
                            current_len -= len(templates_tokens[slot.template.name])
                            del segs_value_seqs[i]
                next_slot_to_trunc = next(iter_slot_trunc_cands, None)
            elif do_trunc_seg:
                i, (template, seg_value_seqs) = next_seg_to_trunc
                if i in segs_value_seqs:
                    current_len -= len(templates_tokens[template.name])
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
        for i, (template, seg_value_seqs) in segs_value_seqs.items():
            previous_slot_index = 0
            for j, value_seq in enumerate(seg_value_seqs):
                slot, _, min_tokens, max_tokens = slot_value_table[(i, j)]
                final_seq += templates_tokens[template.name][previous_slot_index:slot.token_index]
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
            final_seq += templates_tokens[template.name][previous_slot_index:]
        if self.sequence_suffix_tokens and slot_for_generation is not None:
            final_seq += self.sequence_suffix_tokens
        return final_seq

    def _tokenize_sequences(self,
        data: T.Iterable[list[Template | dict[str, str]]],
        gen_slots: list[tuple[str, int, int, TokenSlot]|None]
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
        data: list[Template | dict[str, str]] | T.Iterable[list[Template | dict[str, str]]],
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




