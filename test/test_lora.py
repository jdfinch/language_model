
import ezpyzy as ez
import json
import pathlib as pl
import torch as pt

import language_model.llama3 as llama
from language_model.training import Training
from language_model.generate import Greedy
from language_model.scheduler import LinearWarmupSchedule


with ez.test('Create LoRA model'):
    model = llama.Llama3(
        template_tokenizer=llama.Llama3TemplateTokenizer(max_out=32),
        training=Training(epochs=10, scheduler=LinearWarmupSchedule(num_warmup_steps=0)),
        generation=Greedy(batch_size=10)
    )
    print(f"{model.unsloth = }")

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


with ez.test('LoRA training', crash=True):
    for epoch, ppl in enumerate(model.train_each_epoch(training_data)):
        print(f"Epoch {epoch} ppl {ppl:.3f}")

    print(f"LoRA training: {pt.cuda.max_memory_allocated() / 1e9:.2f} GB")


with ez.test('LoRA prediction'):
    lora_predictions = predict()
    lora_accuracy = evaluate(lora_predictions)
    print(f"LoRA training got {100 * lora_accuracy:.3f}")


with (ez.test('Disable LoRA and repredict')):
    model.deactivate_adapter()
    no_lora_predictions = predict()
    for i, (base_prediction, no_lora_prediction) in enumerate(zip(base_predictions, no_lora_predictions)):
        assert base_prediction == no_lora_prediction, \
            f"{base_prediction} != {no_lora_prediction} on prediction {i}"

with (ez.test('Re-Enable LoRA and repredict')):
    model.activate_adapter('adapter')
    lora_repredictions = predict()
    for i, (lora_reprediction, no_lora_prediction) in enumerate(zip(lora_repredictions, no_lora_predictions)):
        assert lora_reprediction == no_lora_prediction, \
            f"{lora_reprediction} != {no_lora_prediction} on prediction {i}"


