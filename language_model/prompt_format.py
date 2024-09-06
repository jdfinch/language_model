
import inspect as ins
import typing as T


class PromptFormat:
    def __init__(self, fn):
        self.fn = fn
        signature = ins.signature(fn)
        args = []
        kwargs = {}
        for name, param in signature.parameters.items():
            if param.kind is param.VAR_POSITIONAL:
                args.extend((f'{{{name} 1}}', f'{{{name} 2}}', f'{{{name} ...}}'))
            else:
                kwargs[name] = f'{{{name}}}'
        self.format = fn(*args, **kwargs)
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


F = T.TypeVar('F')

def prompt_format(fn: F) -> F | PromptFormat:
    return PromptFormat(fn)