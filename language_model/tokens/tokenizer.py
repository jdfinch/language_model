from __future__ import annotations

import abc




class Tokenizer(abc.ABC):

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Tokenizes text into token IDs"""

    @abc.abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decodes token IDs into text."""

    @property
    @abc.abstractmethod
    def slot_lead_pattern(self) -> str:
        """Regex pattern for matching chars preceding __template_slots__ that should be included in the slot prefix."""

    @property
    @abc.abstractmethod
    def slot_trail_pattern(self) -> str:
        """Regex pattern for matching chars following __template_slots__ that should be included in the slot suffix."""

    @property
    @abc.abstractmethod
    def slot_affix_replacements(self) -> dict[str, str]:
        """Replacements for special characters in slot prefixes and suffixes."""


def HuggingfaceTokenizer(
    tokenizer,
    slot_lead_pattern=r' ?',
    slot_trail_pattern=r'',
    slot_affix_replacements:dict[str, str] = None
) -> Tokenizer:
    import transformers as hf
    if isinstance(tokenizer, str):
        tokenizer = hf.AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = '-'
    tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    discovered_affix_replacements = dict(
        bos=tokenizer.bos_token,
        eos=tokenizer.eos_token,
        pad=tokenizer.pad_token)
    slot_affix_replacements = {**discovered_affix_replacements, **(slot_affix_replacements or {})}
    class HuggingfaceTokenizer(Tokenizer):
        @property
        def tokenizer(self):
            return tokenizer
        def encode(self, text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)
        def decode(self, token_ids: list[int]) -> str:
            return tokenizer.decode(token_ids)
        @property
        def slot_lead_pattern(self) -> str:
            return slot_lead_pattern
        @property
        def slot_trail_pattern(self) -> str:
            return slot_trail_pattern
        @property
        def slot_affix_replacements(self) -> dict[str, str]:
            return slot_affix_replacements
    return HuggingfaceTokenizer()