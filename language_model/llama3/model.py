

import ezpyzy as ez
import dataclasses as dc

from language_model.tokens.tokenizer import HuggingfaceTokenizer, HuggingfaceTokenizerConfig
from language_model.llama3.templates import Llama3TemplateTokenizerConfig
from language_model.language_model_config import LanguageModelConfig


@dc.dataclass
class Llama3Config(LanguageModelConfig):
    tokenizer_templates: HuggingfaceTokenizer = HuggingfaceTokenizerConfig()


@dc.dataclass
class Llama3(ez.ImplementsConfig, Llama3Config):
    tokenizer_templates: Llama3TemplateTokenizerConfig = Llama3TemplateTokenizerConfig()
    def __post_init__(self):
        super().__post_init__()


if __name__ == '__main__':

    llama3 = Llama3()
    print(llama3.configured.json())