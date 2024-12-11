
import ezpyzy as ez
import pathlib as pl
import dataclasses as dc
import peft


@dc.dataclass
class LoRA(ez.Config):
    rank: int = 1
    """The LoRA rank to use. If None, no LoRA adapter be used at all."""
    modules: tuple[str, ...] = tuple(
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
            path = pl.Path(self.base)
            self.repo_id = str(path.parent if path.expanduser().is_file() else path)
        super().__post_init__()

    def _set_alpha(self, alpha):
        if alpha is None:
            return self.rank * 2
        else:
            return alpha

    def _set_rank(self, rank):
        if not self.configured.has.alpha:
            with self.configured.configuring():
                self.alpha = rank * 2
        return rank

    def get_peft_config(self):
        config = peft.LoraConfig(
            r=self.rank,
            target_modules=list(self.modules),
            lora_alpha=self.alpha, # noqa
            lora_dropout=self.dropout)
        return config


