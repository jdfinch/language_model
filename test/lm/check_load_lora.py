

import ezpyzy as ez

from test.check_lm import models, eval, data_capital_langs
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else 'Llama'
Model = models[model_name]

model = Model(
    f'ex/test/{model_name}/lora_capital_langs',
    lora_merge_on_load=True,
    optimized_inference_only=True
)

with ez.check("Load LoRA for capital languages"):
    eval(model, {
        "Kyiv": "Ukrainian",
        "London": "English",
        "Washington D.C.": "English",
        "Hanoi": "Vietnamese"
    }.items())