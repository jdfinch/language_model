
import ezpyzy as ez

import dataclasses as dc

import peft


@dc.dataclass
class LoRA(ez.Config):
    rank: int = 2
    """The LoRA rank to use. If None, no LoRA adapter be used at all."""
    modules: tuple[str] = tuple(
        'q_proj k_proj v_proj o_proj gate_proj up_proj down_proj'.split())
    """The modules to apply LoRA adapters to. Modules must be names from model architecture and can be found using print(model.model)"""
    alpha: float | None = None
    """The alpha value to use for LoRA. If None, the alpha will be set to 2*rank (highly recommended)."""
    dropout: float = 0.0
    """The dropout rate to use when training LoRA adapter weights."""
    lora_merge_on_load: bool = False
    """Whether to merge the LoRA adapter into the original model weights on load."""
    trained: bool = False
    """Whether the LoRA is ready to be used during inference (this should be set when finetuning the LoRA starts)"""
    repo_id: str = None
    """The path on disk or huggingface repository ID to load the LoRA from"""

    def __post_init__(self):
        if (self.repo_id is None and
            isinstance(self.base, str) and not self.base.lstrip().startswith('{')
        ):
            self.repo_id = self.base
        super().__post_init__()
        if self.alpha is None:
            with self.configured.not_configuring():
                self.alpha: float = 2 * self.rank

    def get_peft_config(self):
        config = peft.LoraConfig(
            r=self.rank,
            target_modules=list(self.modules),
            lora_alpha=self.alpha, # noqa
            lora_dropout=self.dropout)
        return config


@dc.dataclass
class LoRAs(ez.MultiConfig[LoRA]):
    active: str|None = 'adapter'
    adapter: LoRA = LoRA()

    @property
    def active_adapter(self) -> LoRA:
        return getattr(self, self.active, None)

    @property
    def number(self):
        return len([1 for _, lora in self if isinstance(lora, LoRA)])



