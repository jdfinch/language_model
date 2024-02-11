
import ezpyzy as ez

from test_language_model.check_lm import models, eval, data_capital_langs
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else 'Llama'
Model = models[model_name]


with ez.check("Finetune capital languages"):
    model = Model()
    print('Perplexity:', model.perplexity(data_capital_langs.items()))
    for ppl in model.training(data_capital_langs.items()):
        print("Perplexity:", ppl)
    model.save(f'ex/test/{model_name}/capital_langs')
    eval(model, assertion=False, tests={
        "Buenos Aires": "Spanish",
        "Paris": "French",
        "Berlin": "German",
        "Athens": "Greek",
        "Kyiv": "Ukrainian",
        "London": "English",
        "Washington D.C.": "English",
        "Hanoi": "Vietnamese"
    }.items())








