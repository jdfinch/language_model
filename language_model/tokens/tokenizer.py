
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