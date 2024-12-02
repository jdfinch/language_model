import ezpyzy as ez
import json
import pathlib as pl
import torch as pt

import language_model.llama3 as llama
from language_model.training import Training
from language_model.generate import Greedy
from language_model.scheduler import LinearWarmupSchedule



with ez.test('Load LoRA', crash=True):
    model = llama.Llama3Config('ex/test/lorav2', device='cuda:0')
    print(model.configured.json())


with ez.test('Create LoRA model'):
    model = llama.Llama3('ex/test/lorav2',
        template_tokenizer=llama.Llama3TemplateTokenizer(max_out=32),
        training=Training(epochs=3, scheduler=LinearWarmupSchedule(num_warmup_steps=0)),
        generation=Greedy(batch_size=10)
    )

    data = json.loads(pl.Path('test/capital_langs.json').read_text())
    training_data = []
    for capital, langauges in data.items():
        training_data.append([
            llama.System("State the language(s) spoken in the city."),
            llama.User(capital),
            llama.Assistant(langauges)
        ])

    def predict():
        eval_data = []
        for capital, languages in data.items():
            eval_data.append([
                llama.System("State the language(s) spoken in the city."),
                llama.User(capital),
                llama.Assistant(...)
            ])
        predictions = model.generate(eval_data)
        return predictions

    def evaluate(predictions):
        correct = 0
        for gold, prediction in zip(data.values(), predictions):
            if gold == prediction: correct += 1
        return correct / len(data)

    print(f"Model size: {pt.cuda.max_memory_allocated() / 1e9:.2f} GB")


with ez.test('Base model predictions', crash=True):
    base_predictions = predict()
    base_accuracy = evaluate(base_predictions)
    print(f"Base model got {100 * base_accuracy:.3f}")

    print(f"Model Prediction: {pt.cuda.max_memory_allocated() / 1e9:.2f} GB")
    pt.cuda.reset_peak_memory_stats()