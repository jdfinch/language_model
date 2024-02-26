import test_language_model
import ezpyzy as ez
import inspect
import subprocess as sp
import pathlib as pl

with ez.CapturedVars() as models:
    from language_model.llama import Llama
    from language_model.t5 import T5

alpha = lambda s: ''.join(c for c in s if c.isalpha() or c == ' ')

root = pl.Path(test_language_model.__file__).parent

data_item_colors = ez.File(root / 'item_colors.json').load()
data_capital_langs = ez.File(root / 'capital_langs.json').load()

def eval(model, tests, assertion=True):
    total_correct = 0
    for prompt, answer in tests:
        generated = model.generate(prompt)
        print('\n', prompt, '==>', generated)
        correct = alpha(answer) in alpha(generated)
        if assertion and not correct:
            raise AssertionError(f'Incorrect: {prompt} ==> {generated}')
        total_correct += correct
    print()

tests = [
    root / 'lm/check_pretrained_lm.py',
    root / 'lm/check_train_lora.py',
    root / 'lm/check_load_lora.py',
]

if __name__ == '__main__':
    for model, _ in models:
        for test in tests:
            sp.run(['python', str(test), model])