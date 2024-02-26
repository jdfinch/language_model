
import ezpyzy as ez

from test_language_model.check_lm import models, eval
import sys

Model = models[sys.argv[1]] if len(sys.argv) > 1 else models['Llama']

model = Model()

with ez.check("translate english to french"):
    elapsed = eval(model, assertion=False, tests=[
        ('translate English to French: My name is John', "Je m'appelle John"),
        ('translate English to French: I like to read books', "J'aime lire des livres"),
        ('translate English to French: I went to the mountain and then to the store and then to the mountain and then to the store and then to the mountain and then to the store and then to the mountain and then to the store and then to the mountain and then to the store and then to the mountain', '---')
    ])

with ez.check("Pretrained country capitals"):
    elapsed = eval(model, assertion=False, tests=[
        ("The capital of the United States is", "Washington D.C."),
        ("What is the capital of Japan?", "Tokyo"),
        ("Germany's capital:", "Berlin")
    ])


