
import ezpyzy as ez
import dataclasses as dc

import transformers as hf

from language_model.utils.get_name_of_subclass import get_name_of_subclass


@dc.dataclass
class Generate(ez.Config):
    max_out: int = 64
    """The maximum number of tokens the model will generate."""
    min_out: int = 0
    """The minimum number of tokens the model will generate."""
    batch_size: int = 1
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
        if not self.configured.has.strategy:
            self.strategy = get_name_of_subclass(self, Generate)

    def construct_hf_config(self) -> hf.GenerationConfig:
        raise TypeError("A subclass of Generate implementing construct_hf_config must be used to specify a generation strategy.")


@dc.dataclass
class Greedy(Generate):
    def construct_hf_config(self):
        return hf.GenerationConfig(
            max_new_tokens=self.max_out,
            min_new_tokens=self.min_out,
            num_beams=1,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            do_sample=False)

@dc.dataclass
class Beam(Generate):
    num_beams: int = 4
    """The number of beams to use for generation."""
    top_k: int|None = 50
    """A filter determining the number of highest-probability vocab tokens to keep."""
    min_p: float = 0.0
    """A filter that keeps only the top tokens whose probability exceeds this value."""
    length_penalty: float = 0.0
    """An exponential penalty used for beam-based generation that punishes longer sequences at length_penalty < 0.0"""
    num_return_sequences: int = 1
    """The number of sequences to return for each input."""

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



if __name__ == '__main__':

    beam = Beam(num_beams=5)
    print(beam.configured.json())
    print(f"{beam.num_return_sequences = }")
    beam.num_return_sequences = 3
    print(f"{beam.num_return_sequences = }")
    try:
        beam.num_return_sequences = 6
    except AssertionError as e:
        print(e)

    diverse_sample = Diverse(groups=2, num_beams=4)

    print(diverse_sample.configured.json())
    print(f"{diverse_sample.num_return_sequences = }")
    print(f"{set(diverse_sample.configured.configured) = }")
    print(f"{set(diverse_sample.configured.unconfigured) = }")
    print(f"{diverse_sample.num_return_sequences = }")
    diverse_sample.num_return_sequences = 3
    print(f"{diverse_sample.num_return_sequences = }")
    try:
        diverse_sample.num_return_sequences = 5
        print(f"{diverse_sample.num_return_sequences = }")
    except AssertionError as e:
        print(e)
