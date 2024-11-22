from __future__ import annotations

import dataclasses as dc
import itertools as it
import copy as cp
import re

import ezpyzy as ez

from language_model.tokens.tokenizer import Tokenizer
from language_model.tokens.token_sequence import TokenSequence
from language_model.tokens.token_sequences import TokenSequences
from language_model.tokens.template_slots import Slot, Input, Output, TokenSlot
from language_model.tokens.template import Template, SegmentTemplate

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
                segment_template = SegmentTemplate(template=template())
                setattr(self, name, segment_template)
                self.configured.set(name, segment_template, configured=is_configured)

@dc.dataclass
class TemplateTokenizer(ez.Config):
    templates: Templates = Templates()
    sequence_prefix: str = ''
    sequence_suffix: str = ''
    max_length: int = None
    pad_to_same_length: bool = True
    pad_to_multiple_of: int = 8
    pad_side: str = 'L'
    max_segments: int | None = None
    tokenizer: Tokenizer = None

    def __post_init__(self):
        super().__post_init__()
        self.sequence_prefix_tokens = TokenSequence(self.sequence_prefix, tokenizer=self.tokenizer)
        self.sequence_suffix_tokens = TokenSequence(self.sequence_suffix, tokenizer=self.tokenizer)
        self.templates_tokens: dict[str, TokenSequence] = {}
        self.slot_trunc_text_tokens: dict[tuple[str, str], TokenSequence] = {}
        self.template_name_map: dict[str, SegmentTemplate] = {}
        self.template_slots: dict[str, list[TokenSlot]] = {}
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
                    slots.append(slot)
                    previous_end = end
            template_suffix = template_text[previous_end:]
            template_tokens = TokenSequence(
                '', is_attended=template.is_attended, is_label=template.is_label, tokenizer=self.tokenizer)
            self.template_slots[template_name] = []
            for template_part, slot in zip(template_parts, slots):
                template_tokens += template_part
                slot = cp.deepcopy(slot)
                slot.index = len(template_tokens)
                slot_trunc_text = TokenSequence(slot.trunc_text,
                    is_attended=template.is_attended, is_label=slot.is_label, tokenizer=self.tokenizer)
                self.slot_trunc_text_tokens[(template_name, slot.name)] = slot_trunc_text
                if isinstance(slot.max, float):
                    assert slot.max > slot.min + len(slot.trunc_text), \
                        f"Slot {slot.name} has a max value length {slot.max} shorter than the sum of its min value length {slot.min} (plus length of truncation_text tokens - {repr(slot.trunc_text)})."
                self.template_slots[template_name].append(slot)
            template_tokens += template_suffix
            self.templates_tokens[template_name] = template_tokens
            self.template_name_map[template_name] = template

    def fill(self,
        segments_values: list[Template]|T.Iterable[list[Template]]|dict[str,str]|T.Iterable[dict[str,str]]
    ) -> TokenSequence|TokenSequences:
        if not isinstance(segments_values, list) or isinstance(segments_values[0], list):
            return TokenSequences([self.fill(value) for value in segments_values],
                pad_to_same_length=self.pad_to_same_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                pad_side=self.pad_side,
                tokenizer=self.tokenizer)

        # get templates with corresponding values
        template_segments_values: list[tuple[SegmentTemplate, dict[str, str]]] = []
        for segment_values in segments_values:
            if isinstance(segment_values, dict):
                temp_name = segment_values['temp']
                segment_values = {k:v for k,v in segment_values.items() if k != 'temp'}
                template = self.template_name_map[temp_name]
            else:
                temp_name = segment_values.__class__.__name__
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
        for i, (segment_values, template) in enumerate(zip(segs_value_dicts, templates)):
            for j, slot in enumerate(slot for _, slot in template.slots):
                value = segment_values[slot.name]
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
            segs_value_dicts = segs_value_dicts[:index_of_segment_for_generation+1]
            templates = templates[:index_of_segment_for_generation+1]
            template_with_generation = templates[index_of_segment_for_generation]
            templates_slots = cp.copy(templates_slots)
            templates_tokens = cp.copy(templates_tokens)
            gen_tmp_name = template_with_generation.name
            template_slots = templates_slots[gen_tmp_name]
            temp_tokens = templates_tokens[gen_tmp_name]
            templates_tokens[gen_tmp_name] = temp_tokens[:template_slots[index_of_slot_for_generation].index]
            templates_slots[gen_tmp_name] = template_slots[:index_of_slot_for_generation]

        # create segments table where original templates/segs_value_dicts indices are IDs
        segments_table = {i: (template, segment_values)
            for i, (template, segment_values) in enumerate(zip(templates, segs_value_dicts))}
        _segments_table_template, _segments_table_values = 0, 1

        # sort segments by trunc_rank, trunc_side, and truncability (and filter by truncability)
        seg_trunc_cands = dict(ez.sort([(i, (template, segment_values))
                for i, (template, segment_values) in segments_table.items()
                if template.trunc_segment and i != index_of_segment_for_generation],
            by=[(-template.trunc_segment_rank, template.trunc_segment_side)
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
            for slot in templates_slots[template.name]:
                value_text = ''.join((slot.prefix, segment_values[slot.name], slot.suffix))
                value_seq = TokenSequence(value_text,
                    is_attended=True, is_label=slot.is_label, tokenizer=self.tokenizer)
                segs_value_seqs[i][1].append(value_seq)

        # compute min and max tokens of each slot
        slot_value_table = {}  # (seg_idx, slot_idx) -> (slot, value, min, max)
        for i, (template, seg_value_seqs) in segs_value_seqs.items():
            for j, (slot, seq) in enumerate(zip(templates_slots[template.name], seg_value_seqs)):
                if slot.truncatable and template.trunc_content:
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
        slot_trunc_cands = dict(ez.sort([
            ((i, j), [slot, seq, min_tokens, max_tokens])
            for (i, j), (slot, seq, min_tokens, max_tokens) in slot_value_table.items()
            if min_tokens < max_tokens
        ], by=[
            (-slot.trunc_rank, slot.trunc_side)
            for (i, _), (slot, _, min_tokens, max_tokens) in slot_value_table.items()
            if min_tokens < max_tokens
        ]))
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
                if i in segs_value_seqs and amount_to_trunc >= len(
                    self.slot_trunc_text_tokens[(templates[i].name, slot.name)]):
                    do_trunc_slot = True
                else:
                    next_slot_to_trunc = next(iter_slot_trunc_cands, None)
            else:
                (_, (template, seg_value_seqs)) = next_seg_to_trunc
                (i, j), (slot, seq, min_tokens, max_tokens) = next_slot_to_trunc
                if (i in segments_table and amount_to_trunc >= len(
                    self.slot_trunc_text_tokens[(templates[i].name, slot.name)]) and
                  slot.trunc_rank >= template.trunc_segment_rank
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
                current_len -= len(templates_tokens[template.name])
                for j in range(len(seg_value_seqs)):
                    slot_trunc_cands.pop((i, j), None)
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
                final_seq += templates_tokens[template.name][previous_slot_index:slot.index]
                previous_slot_index = slot.index
                if len(value_seq) > max_tokens:
                    trunc_tokens = self.slot_trunc_text_tokens[(template.name, slot.name)]
                    if slot.trunc_side == 'L':
                        final_seq += trunc_tokens
                        final_seq += value_seq[len(trunc_tokens)-max_tokens:]
                    else:
                        final_seq += value_seq[:max_tokens-len(trunc_tokens)]
                        final_seq += self.slot_trunc_text_tokens[(template.name, slot.name)]
                else:
                    final_seq += value_seq
            final_seq += templates_tokens[template.name][previous_slot_index:]
        if self.sequence_suffix_tokens and slot_for_generation is not None:
            final_seq += self.sequence_suffix_tokens
        return final_seq


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

    seqs = tokenizer.fill(data)

    for seq in seqs:
        print('|'.join(seq.tokens()))
        print('\n\n')




