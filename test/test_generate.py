
import ezpyzy as ez
import json
import pathlib as pl
import copy as cp

import language_model.llama3 as llama
import language_model.tokens as tok
from language_model.generate import Greedy


with ez.test("Construct Llama 3"):
    model = llama.Llama3(
        model_base='unsloth/Llama-3.2-3B-Instruct',
        quantization=None,
        template_tokenizer=llama.Llama3TemplateTokenizer(max_length=128, max_out=64),
        generation=Greedy(batch_size=3))

with ez.test("Create data"):
    captial_langs = json.loads(pl.Path('test/capital_langs.json').read_text())
    prompts = []
    for capital, languages in captial_langs.items():
        prompt = [
            llama.System("Give a long answer in as much detail as possible."),
            llama.User(f"What languages are spoken in {capital}?"),
            llama.Assistant(...),
            llama.User("Can you please tell me more? Provide details of history and culture. Use quotes from real people and sources."),
            llama.Assistant(...),
            llama.User("Summarize everything you've told me in one sentence."),
            llama.Assistant(...)
        ]
        prompts.append(prompt)

with ez.test("Llama 3 generation"):
    predictions: set[str] = set()
    for prompt in cp.deepcopy(prompts[:5]):
        response, = model.generate(prompt)
        print(prompt[1].content, '->')
        print(response, '\n')
        for word in response.split(' '):
            predictions.add(''.join(c.lower() for c in word if c.isalpha()))
    expected = set("pashto, dari, albanian, arabic, berber, catalan, portuguese".split(', '))
    print(f"expected: {', '.join(expected)}")
    assert len(predictions & expected) >= 2

with ez.test("Llama 3 generation (batched)"):
    predictions: set[str] = set()
    responses = model.generate(cp.deepcopy(prompts[:5]))
    for prompt, response in zip(prompts[:5], responses):
        print(prompt[1].content, '->')
        print(response, '\n')
        for word in response.split(' '):
            predictions.add(''.join(c.lower() for c in word if c.isalpha()))
    print(f"expected: {', '.join(expected)}")
    assert len(predictions & expected) >= 2

with ez.test("Llama3 chained generation"):

    chat = cp.deepcopy(prompts[0])
    response = not None
    while response is not None:
        prompt = model.template_tokenizer.tokenize(chat)
        print(f'Prompt len {len(prompt)}'.center(70, '.'))
        print('|'.join(prompt.tokens()))
        response ,= model.generate(chat)
        print(f'GENERATED: {response}', '\n\n')