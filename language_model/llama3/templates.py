
import dataclasses as dc

import language_model.tokens as tok



@dc.dataclass
class System(tok.Template):
    """General instructions that condition how Llama3 should respond to specific instructions."""
    template = "<|start_header_id|>system<|end_header_id|>\n\n<content><|eot_id|>"
    content: tok.Slot = tok.Input(trunc_side='R', trunc_rank=0)

@dc.dataclass
class User(tok.Template):
    """A user turn in an instruction chat."""
    template = "<|start_header_id|>user<|end_header_id|>\n\n<content><|eot_id|>"
    content: tok.Slot = tok.Input()

@dc.dataclass
class Assistant(tok.Template):
    """An assistant turn in an instruction chat."""
    template = "<|start_header_id|>assistant<|end_header_id|>\n\n<content><|eot_id|>"
    content: tok.Slot = tok.Input()

@dc.dataclass
class Response(tok.Template):
    """An assistant turn in an instruction chat."""
    template = "<|start_header_id|>assistant<|end_header_id|>\n\n<content><|eot_id|>"
    content: tok.Slot = tok.Output()


@dc.dataclass
class Llama3Templates(tok.Templates):
    system: tok.SegmentTemplate = tok.SegmentTemplate(template=System())
    user: tok.SegmentTemplate = tok.SegmentTemplate(template=User(), trunc_segment=True)
    assistant: tok.SegmentTemplate = tok.SegmentTemplate(template=Assistant(), trunc_segment=True)
    assistant_response: tok.SegmentTemplate = tok.SegmentTemplate(template=Response())


@dc.dataclass
class Llama3TemplateTokenizer(tok.TemplateTokenizer):
    templates: Llama3Templates = Llama3Templates()
    tokenizer: tok.HuggingfaceTokenizer = tok.HuggingfaceTokenizer(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct')
    max_length: int = 256


if __name__ == '__main__':
    tokenizer = Llama3TemplateTokenizer()
    print(tokenizer.configured.json())