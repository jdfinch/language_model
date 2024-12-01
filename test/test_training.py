
import ezpyzy as ez
import pathlib as pl
import copy as cp
import json
import torch as pt

import language_model.llama3 as llama
import language_model.training as tr
import language_model.optimizer as opt
import language_model.scheduler as sch
import language_model.tokens as tok


with ez.test("Construct Llama3"):
    model = llama.Llama3(
        quantization=None,
        training=tr.Training(
            epochs=10,
            batch_size=1,
            optimizer=opt.Adam(learning_rate=1e-4, quantization='8bit'),
            scheduler=sch.LinearWarmupSchedule(num_warmup_steps=0)
        ),
        template_tokenizer=llama.Llama3TemplateTokenizer(max_length=128, max_out=16)
    )
    raw_data = json.loads(pl.Path('test/capital_langs.json').read_text())
    data = []
    for capital, langauges in raw_data.items():
        data.append([
            llama.System("State the language(s) spoken in the city."),
            llama.User(capital),
            llama.Assistant(langauges)
        ])
    print(f"Model size is {pt.cuda.max_memory_allocated() / 1e9:.3f} GB")

with ez.test("Check Untrained Accuracy"):
    def evaluate():
        correct = 0
        for example in data:
            capital = example[1].content
            gold = example[-1].content
            prompt = cp.deepcopy(example)
            prompt[-1].content = ...
            prediction, = model.generate([prompt])
            # print(f"{'✓' if gold == prediction else 'X'} {capital}:  {gold = }  {prediction = }")
            if gold == prediction:
                correct += 1
        print(f'Accuracy: {100 * correct / len(data):.1f}%')
        return correct / len(data)
    untrained_accuracy = evaluate()

with ez.test("Full finetuning each epoch"):
    for epoch, ppl in enumerate(model.train_each_epoch(data)):
        print(f'Epoch {epoch}: {ppl:.3f} ppl', end=', ')
        accuracy = evaluate()
    assert accuracy > untrained_accuracy and accuracy >= 0.9

with ez.test("Full finetuning each step"):
    model.training.epochs = 2
    for epoch, steps in enumerate(model.train_each_step_each_epoch(data)):
        print(f'Epoch {epoch}:')
        for step, ppl in enumerate(steps):
            print(f"{ppl:.3f}", end=', ')
        print()
    print(f"Training took {pt.cuda.max_memory_allocated() / 1e9:.3f} GB")



    for epoch, ppl in enumerate(model.train_each_epoch(data)):
        ...
