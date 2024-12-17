
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
        template_tokenizer=llama.Llama3TemplateTokenizerConfig(max_out=32),
        training=Training(
            epochs=5,
            scheduler=LinearWarmupSchedule(num_warmup_steps=0),
        ),
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


with ez.test('LoRA training', crash=True):
    for epoch, steps in enumerate(model.train_each_step_each_epoch(training_data)):
        ppls = list(steps)
        ppl = sum(ppls) / len(ppls)
        print(f"Epoch {epoch} ppl {ppl:.3f}")
        if epoch == 2:
            early_predictions = predict()
            early_accuracy = evaluate(early_predictions)
            print(f"Early predictions got {100 * early_accuracy:.3f}")
            model.save('ex/test/lorav2')
    print(f"LoRA training: {pt.cuda.max_memory_allocated() / 1e9:.6f} GB")


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


with ez.test('Delete model'):
    model.delete()
    del model
    gb_allocated = pt.cuda.memory_allocated() / 1e9
    print(f"After deleting model: {gb_allocated:.2f} GB allocated")
    assert gb_allocated < 0.1


with ez.test('Load LoRA'):
    model = llama.Llama3('ex/test/lorav2', training=Training(resume_previous_training=True))
    disk_lora_predictions = predict()
    disk_lora_accuracy = evaluate(disk_lora_predictions)
    print(f"Loaded LoRA got {100 * disk_lora_accuracy:.3f}")
    for original_prediction, loaded_prediction in zip(early_predictions, disk_lora_predictions):
        assert original_prediction == loaded_prediction, \
            f"{original_prediction} != {loaded_prediction}"


with ez.test('Resume training'):
    for i, steps in enumerate(model.train_each_step_each_epoch(training_data)):
        ppls = list(steps)
        ppl = sum(ppls) / len(ppls)
        print(f'Resumed epoch {i} ppl: {ppl:.3f}')
    resumed_lora_predictions = predict()
    resumed_lora_accuracy = evaluate(resumed_lora_predictions)
    print(f"Resumed LoRA got {100 * resumed_lora_accuracy:.3f}")


