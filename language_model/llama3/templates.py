
import dataclasses as dc

import ezpyzy as ez
import language_model.tokens as tok
from language_model.tokens import TemplateTokenizer

@dc.dataclass
class Text(tok.Template):
    """Raw token sequence without additional formatting."""
    template = "<content>"
    content: tok.Slot = tok.Output()

@dc.dataclass
class RoleHeader(tok.Template):
    """The prefix of an instruction chat like <|start_header_id|>user<|end_header_id|>\n\n"""
    template = "<|start_header_id|><role><|end_header_id|>\n\n"
    role: tok.Slot = tok.Input(trunc=False)

@dc.dataclass
class System(tok.Template):
    """General instructions that condition how Llama3 should respond to specific instructions."""
    template = "<|start_header_id|>system<|end_header_id|>\n\n<content><|eot_id|>"
    content: tok.Slot = tok.Input(trunc_side='R', trunc_rank=-1)

@dc.dataclass
class User(tok.Template):
    """A user turn in an instruction chat."""
    template = "<|start_header_id|>user<|end_header_id|>\n\n<content><|eot_id|>"
    content: tok.Slot = tok.Input(trunc_side='R', trunc_rank=1)

@dc.dataclass
class Assistant(tok.Template):
    """An assistant turn in an instruction chat."""
    template = "<|start_header_id|>assistant<|end_header_id|>\n\n<content>"
    content: tok.Slot = tok.Output(trunc_side='R', trunc_rank=1)


@dc.dataclass
class Llama3Templates(tok.Templates):
    text: tok.SegmentTemplate|Text = Text()
    role_header: tok.SegmentTemplate|RoleHeader = tok.SegmentTemplate(
        template=RoleHeader(), trunc_segment=False, trunc_content=False)
    system: tok.SegmentTemplate|System = tok.SegmentTemplate(
        template=System(), trunc_segment=False)
    user: tok.SegmentTemplate|User = tok.SegmentTemplate(
        template=User(), trunc_segment=True, trunc_segment_rank=0)
    assistant: tok.SegmentTemplate|Assistant = tok.SegmentTemplate(
        template=Assistant(), trunc_segment=True, trunc_segment_rank=0)

@dc.dataclass
class Llama3TemplateTokenizerConfig(tok.TemplateTokenizerConfig):
    templates: tok.Templates = Llama3Templates()
    tokenizer: tok.HuggingfaceTokenizerConfig = tok.HuggingfaceTokenizerConfig(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct')
    max_length: int = 256

@dc.dataclass
class Llama3TemplateTokenizer(TemplateTokenizer, ez.ImplementsConfig, Llama3TemplateTokenizerConfig):
    templates: tok.Templates = Llama3Templates()
    tokenizer: tok.HuggingfaceTokenizer = tok.HuggingfaceTokenizerConfig(
        repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct')
    max_length: int = 256

if __name__ == '__main__':
    config = Llama3TemplateTokenizerConfig()
    print(config.configured.json())

    tokenizer = Llama3TemplateTokenizer()
    print(tokenizer.tokenize([
        System(content="This is a system message."),
        User(content="This is a user message."),
    ]))
