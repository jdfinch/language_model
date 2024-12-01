
import ezpyzy as ez

import dataclasses as dc



@dc.dataclass
class LoRA(ez.ImmutableConfig):
    rank: int = 2
    """The LoRA rank to use. If None, no LoRA adapter be used at all."""
    modules: tuple[str] = None
    """The modules to apply LoRA adapters to. Modules must be names from model architecture and can be found using print(model.model)"""
    alpha: float | None = None
    """The alpha value to use for LoRA. If None, the alpha will be set to 2*rank (highly recommended)."""
    dropout: float = 0.0
    """The dropout rate to use when training LoRA adapter weights."""
    lora_merge_on_load: bool = False
    """Whether to merge the LoRA adapter into the original model weights on load."""


@dc.dataclass
class AdaptersConfig(ez.MultiConfig[LoRA]):
    primary: LoRA = LoRA()



