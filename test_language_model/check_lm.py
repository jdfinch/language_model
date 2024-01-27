import ezpyzy as ez
import inspect
import subprocess as sp
import pathlib as pl

with ez.CapturedVars() as models:
    from language_model.llama import Llama

alpha = lambda s: ''.join(c for c in s if c.isalpha() or c == ' ')

data_item_colors = ez.File('test_language_model/item_colors.json').load()
data_capital_langs = ez.File('test_language_model/capital_langs.json').load()

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
    'test_language_model/lm/check_pretrained_lm.py',
    'test_language_model/lm/check_train_lora.py',
    'test_language_model/lm/check_load_lora.py',
]

if __name__ == '__main__':
    for model, _ in models:
        for test in tests:
            sp.run(['python', str(test), model])