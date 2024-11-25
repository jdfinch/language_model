

import ezpyzy as ez
import dataclasses as dc

from language_model.llama3.templates import Llama3TemplateTokenizerConfig
from language_model.language_model_config import LanguageModelConfig


@dc.dataclass
class Llama3Config(LanguageModelConfig):
    tokenizer_templates: Llama3TemplateTokenizerConfig = Llama3TemplateTokenizerConfig()


@dc.dataclass
class Llama3(ez.ImplementsConfig, Llama3Config):
    def __post_init__(self):
        super().__post_init__()


if __name__ == '__main__':

    with ez.test('Config', crash=True):
        config = Llama3Config()
        print(config.configured.json())

    model = Llama3()
    print(model.tokenizer_templates.tokenizer.encode('Hello, world!'))

