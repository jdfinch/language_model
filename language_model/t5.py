import peft
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    get_constant_schedule_with_warmup as warmup_scheduler
)
from bitsandbytes.optim import AdamW8bit
from peft import LoraConfig

from accelerate import Accelerator
from tqdm import tqdm
import ezpyzy as ez
from dataclasses import dataclass as settings; vars().update(settings=ez.settings)
import pathlib as pl
import shutil
import os
import math


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


loss_mask = -100


def load_merge_and_save_lora(lora_path: ez.filelike, merged_path: ez.filelike=None):
    lora_path = ez.File(lora_path).path
    print(lora_path)
    name = lora_path.name
    adapter_config = ez.File(lora_path / 'adapter_config.json').load()
    base_model_name = adapter_config['base_model_name_or_path']
    base_model = T5ForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model = peft.PeftModel.from_pretrained(base_model, lora_path)
    merged = model.merge_and_unload()
    if merged_path is None:
        merged_path = lora_path.parent / f"{name}.MERGED"
    merged.save_pretrained(merged_path, safe_serialization=False, save_peft_format=False)
    return merged_path


@settings
class T5Hyperparameters(ez.Settings):
    base: ez.ColStr = ez.Def('t5-{param_magnitude}')
    param_magnitude: ez.ColStr = ez.Def('small')
    format: ez.ColStr = ez.Def('''{input} {output}''')
    train_on_s2s_inputs: ez.ColBool = ez.Def(False)
    quantize: ez.ColStr = ez.Def('bf16')
    checkpoint: ez.ColStr = None
    checkpoint_after_every_x_epochs: ez.ColFloat = ez.Def(1.0)
    checkpoint_clean_up_after_train: ez.ColBool = ez.Def(True)
    epoch: ez.ColInt = ez.Def(0)
    step: ez.ColInt = ez.Def(0)
    epochs: ez.ColInt = ez.Def(1)
    max_sequence_length: ez.ColInt = ez.Def(512)
    train_batch_size: ez.ColInt = ez.Def(1)
    gradient_accumulation_steps: ez.ColInt = ez.Def(1)
    optimizer: ez.ColStr = ez.Def('adamw_bnb_8bit')
    learning_rate: ez.ColFloat = ez.Def(1e-4)
    weight_decay: ez.ColFloat = ez.Def(0.0)
    warmup_steps: ez.ColInt = ez.Def(0)
    lr_scheduler_type: ez.ColStr = ez.Def('constant')
    lora: ez.ColInt = ez.Def(8)
    lora_alpha: ez.ColInt = None
    lora_dropout: ez.ColFloat = ez.Def(0.1)
    lora_modules: ez.Column[list[str]] | list[str] | None = None
    lora_merge_on_load: ez.ColBool = ez.Def(True)
    gradient_checkpointing: ez.ColBool = ez.Def(True)
    max_output_length: ez.ColInt = ez.Def(512)
    repetition_penalty: ez.ColFloat = ez.Def(2.0)
    num_beams: ez.ColInt = ez.Def(1)
    temperature: ez.ColFloat = ez.Def(0.6)
    sampled_generation: ez.ColBool = ez.Def(False)
    top_p: ez.ColFloat = ez.Def(0.9)
    top_k: ez.ColInt = ez.Def(50)
    gen_batch_size: ez.ColInt = ez.Def(1)
    ppl_batch_size: ez.ColInt = ez.Def(1)
    dynamic_tokenization: ez.ColBool = ez.Def(True)

    def actual_train_batch_size(self):
        return self.train_batch_size // self.gradient_accumulation_steps

    def __post_init__(self):
        if '{param_magnitude}' in self.base:
            self.base = self.base.replace('{param_magnitude}', str(self.param_magnitude))



class T5(T5Hyperparameters):
    def __post_init__(self):
        T5Hyperparameters.__post_init__(self)
        if pl.Path(self.base).exists() and (pl.Path(self.base)/'hyperparameters.json').exists():
            loaded_hyperparams:dict = ez.File(pl.Path(self.base)/'hyperparameters.json').load()
            specified_hyperparameters = vars(self).pop('settings')
            hyperparameters = {**loaded_hyperparams, **specified_hyperparameters}
            vars(self).update(hyperparameters)
        self.hyperparameters: dict = dict(vars(self))
        tokenizer_reponame = "t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_reponame,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True
        )
        quant_kwargs = {}
        if self.quantize is not None:
            if self.quantize == 'nf4':
                quant_kwargs = dict(
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False
                    ),
                    device_map='auto'
                )
            elif self.quantize == 'int8':
                quant_kwargs = dict(
                    load_in_8bit=True,
                    device_map='auto'
                )
            elif self.quantize == 'bf16':
                quant_kwargs = dict(
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
            else:
                raise ValueError(
                    f"Invalid quantization level: {self.quantize}.\n"
                    f"Supported quantizations are: 'nf4', 'int8', 'bf16', None"
                )

        load_path = pl.Path(self.base)
        delete_merge_path = None
        if load_path.exists() and (load_path / 'adapter_config.json').exists() and self.lora_merge_on_load:
            merged_path = load_path.parent / f"{load_path.name}.MERGED"
            delete_after = not merged_path.exists()
            ez.subproc(load_merge_and_save_lora, load_path)
            if delete_after:
                delete_merge_path = merged_path
            load_path = merged_path
        else:
            load_path = self.base
        self.model = T5ForConditionalGeneration.from_pretrained(
            load_path, torch_dtype=torch.bfloat16, **quant_kwargs
        )






