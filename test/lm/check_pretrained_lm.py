
import ezpyzy as ez

from test.check_lm import models, eval
import sys

Model = models[sys.argv[1]] if len(sys.argv) > 1 else models['Llama']

model = Model()
with ez.check("Pretrained country capitals"):
    elapsed = eval(model, [
        ("The capital of the United States is", "Washington D.C."),
        ("What is the capital of Japan?", "Tokyo"),
        ("Germany's capital:", "Berlin")
    ])


