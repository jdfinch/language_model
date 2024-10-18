
import ezpyzy as ez
import dataclasses as dc
import pathlib as pl
import json
import transformers as hf
import torch as pt

from language_model.lm_config import get_name_of_subclass_for_field

# black magic type hinting: sneak the "base" decorator into "dataclass" var name
from dataclasses import dataclass; vars().update(dataclass=ez.config)



@dataclass
class Generate(ez.Config):
    max_tokens: int = 512
    """The maximum number of tokens the model will generate."""
    min_tokens: int = 0
    """The minimum number of tokens the model will generate."""
    batch_size: int | None = None
    """The batch size to use for generation. Defaults to the actual train batch size."""
    num_beams: int = 1
    """The number of beams to use for generation."""
    repetition_penalty: float = 1.2
    """The repetition penalty to use for generation."""
    no_repeat_ngram_size: int = 0
    """If set to a positive int, ngrams of that size cannot be repeated in the output."""
    strategy: str = None
    """The decoding strategy to use for generation."""

    def __post_init__(self):
        super().__post_init__()
        get_name_of_subclass_for_field(self, Generate, 'strategy')

    def construct_hf_config(self):
        raise TypeError("A subclass of Generate must be used to specify a generation strategy.")


@dataclass
class Greedy(Generate):
    strategy: str = 'Greedy'
    """The decoding strategy to use for generation."""

    def construct_hf_config(self):
        return hf.GenerationConfig(
            max_new_tokens=self.max_tokens,
            min_new_tokens=self.min_tokens,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            do_sample=False)

@dataclass
class Beam(Generate):
    num_beams: int = 4
    """The number of beams to use for generation."""
    top_k: int|None = 50
    """A filter determining the number of highest-probability vocab tokens to keep."""
    min_p: float = 0.0
    """A filter that keeps only the top tokens whose probability exceeds this value."""
    length_penalty: float = 0.0
    """An exponential penalty used for beam-based generation that punishes longer sequences at length_penalty < 0.0"""
    strategy: str = 'Beam'
    """The decoding strategy to use for generation."""

    def _set_num_return_sequences(self, num_return_sequences):
        assert num_return_sequences <= self.num_beams, \
            f"num_return_sequences ({num_return_sequences}) must be less than or equal to num_beams ({self.num_beams})"
        return num_return_sequences


@dc.dataclass
class Contrastive(Beam):
    """Contrastive decoding is a technique for improving the non-repetition of very long generated sequences."""
    penalty_alpha: float = 0.6
    """The penalty alpha value to use for contrastive decoding."""

@dc.dataclass
class Diverse(Beam):
    """Diverse beam decoding is a tecchnique for improving the diversity of generated beams."""
    groups: int = 4
    """The number of diverse beam groups to use for diverse beam search."""

@dc.dataclass
class Sample(Beam):
    temperature: float = 0.6
    """The temperature to use for generation."""

@dc.dataclass
class ContrastiveSample(Contrastive, Sample): pass

@dc.dataclass
class DiverseSample(Diverse, Sample): pass
