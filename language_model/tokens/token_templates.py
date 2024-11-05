from __future__ import annotations

import dataclasses as dc
import itertools as it
import copy as cp

import ezpyzy as ez

import transformers as hf

from language_model.tokens.tokenizer import Tokenizer
from language_model.tokens.token_sequence import TokenSequence, Token
from language_model.tokens.token_sequences import TokenSequences
from language_model.tokens.template import (
    Template, Slot, TokenSlot, InputSlot, OutputSlot, TemplateConfig)

import typing as T

default: T.Any = object()

def fields(cls_or_instance) -> list[dc.Field]: return dc.fields(cls_or_instance) # noqa



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
        template = "<|start_header_id|>assistant<|end_header_id|>\n\n{output}"
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
        LlamaTemplates.bot(...)
    ]

    sequence = templates.fill(chat)
    print('|'.join(sequence.tokens()))



