
import dataclasses as dc

import ezpyzy as ez


@dc.dataclass
class LoRA:
    rank: int = 2
    """The LoRA rank to use. If None, no LoRA adapter be used at all."""
    modules: tuple[str] = (
        'embed_tokens', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head')
    """The modules to apply LoRA adapters to. Modules must be names from model architecture and can be found using print(model.model)"""
    alpha: float | None = None
    """The alpha value to use for LoRA. If None, the alpha will be set to 2*lora (recommended)."""
    dropout: float = 0.0
    """The dropout rate to use when training LoRA adapter weights."""