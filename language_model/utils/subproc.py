
"""
Run a python function or module as a subprocess programmatically.
"""


import inspect as ins
import subprocess as sp
import pickle as pkl
import functools as ft
import typing as T


def _subproc(fn_or_module, *args, **kwargs):
    arguments = pkl.dumps((args, kwargs))
    if ins.isfunction(fn_or_module):
        fn = fn_or_module
        module = ins.getmodule(fn)
    else:
        module = fn_or_module
        fn = None
    module_name = module.__name__
    fn_name = fn.__name__ if fn else None
    if fn_name:
        code = f"import {module_name}; " \
               f"import pickle, sys, io; " \
               f"sys.stdin = io.TextIOWrapper(sys.stdin.detach(), encoding='latin-1'); " \
               f"x = bytes(input(), 'latin-1'); " \
               f"args, kwargs = pickle.loads(x); " \
               f"{module_name}.{fn_name}(*args, **kwargs)"
    else:
        code = f"import {module_name}"
    p = sp.Popen(['python', '-c', code], stdin=sp.PIPE)
    p.communicate(arguments)
    p.wait()
    return p.returncode


F = T.TypeVar('F')

def subproc(fn_or_module: F) -> F:
    if ins.isfunction(fn_or_module):
        return ft.partial(_subproc, fn_or_module)
    else:
        _subproc(fn_or_module)
        return fn_or_module


# if __name__ == '__main__':
#     from language_model.model.llama import load_merge_and_save_lora
#     subproc(load_merge_and_save_lora, 'ex/test/Llama/lora_capital_langs')




