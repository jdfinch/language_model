import json
import sys
import ezpyzy as ez
import dataclasses as dc
import pathlib as pl
import random as rng

import transformers as hf

hf.logging.set_verbosity_error()

import language_model.generate as gen
import language_model.training as tr
import language_model.lora as lora
import language_model.tokens as tok


@dc.dataclass
class LanguageModelConfig(ez.ImmutableConfig):
    """Configuration for a language model"""
    model_to_load: str | None = None
    """Path to a custom model to load. The base will be used instead if this is None."""
    model_base: str | None = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    """The base model repository ID. If no model_to_load is specified, the base will be loaded from this ID."""
    quantization: str | None = 'nf4'
    """Quantization mode to use. If None, no quantization will be used (half-precision)."""
    load_locally_saved_models_only: bool = False
    """Whether to load models only from the local cache, not from the Hugging Face model hub."""
    device: str = 'cuda'
    """The hardware device to use for training and/or generation, such as 'cuda', 'cuda:7', or 'cpu'."""
    template_tokenizer: tok.TemplateTokenizer = None
    """A Config for the tokenizer and templates to use to format sequences passed to the model."""
    adapters: ez.MultiConfig[lora.LoRA]|None = ez.MultiConfig(adapter=lora.LoRA())
    """The LoRA adapters to use for the model (set to None for full fine-tuning)."""
    active_adapter: str|None = 'adapter'
    """Which LoRA adapter is active for training and inference."""
    training: tr.Training|None = tr.Training()
    """Hyperparameters and configuration for training the model."""
    generation: gen.Generate = gen.Greedy()
    """Hyperparameters and configuration for generating text from the model."""
    rng_seed: int | None = None
    """Seed for calls to random number generation, such as shuffling training data."""

    def __post_init__(self):
        # If base is a path to a model folder, load the config from that folder
        self.loaded_model_path: str|None = None
        base_folder = None
        if isinstance(self.base, (str, pl.Path)):
            config_path = pl.Path(self.base)
            if config_path.expanduser().is_dir():
                self.loaded_model_path = str(config_path)
                config_path = config_path / 'language_model_config.json'
            if config_path.exists():
                self.base = str(config_path)
                base_folder = config_path.parent
        # Load the Config
        super().__post_init__()
        # If base_model is actually a LoRA adapter, reorganize the base model and adapter in the config
        if isinstance(base_folder, pl.Path):
            adapter_config_path = base_folder / 'adapter_config.json'
            if adapter_config_path.exists():
                assert self.adapters is not None and self.adapters.adapter is not None, \
                    f"Must have an Adapters config to load an adapter from {adapter_config_path}"
                adapter_config = json.loads(adapter_config_path.read_text())
                if self.model_base is None or not self.configured.has.model_base:
                    self.model_base = adapter_config.get('base_model_name_or_path')
                if self.adapters.adapter.repo_id is None:
                    self.adapters.adapter.repo_id = str(base_folder)
        if self.rng_seed is None:
            self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng = rng.Random(self.rng_seed)