
# Language Model

## Install

```shell
git clone https://github.com/jdfinch/language_model.git
cd language_model
conda env create --name PROJ python=3.10
conda activate PROJ
pip install -r requirements.txt
```
For Llama:

You will need access to a a >12GB GPU to run the model.

GPU usage will reach >30 GB for nontrivial model training on the smallest (7B) models.

For T5:

You can train most T5 models on small GPU.

## Usage

Two model architectures are provided as of now: Llama2 and T5. Once you instantiate your desired model with its hyperparameters, you can use them the same way in training and generation.

```python
from language_model.llama import Llama

llama = Llama(
    # Specify all hyperparameters here
    #   (or don't, and defaults will be used)
)
```

To use **Meta Llama2**, you will need to request access to the model [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), generate an access token [following these instructions](https://huggingface.co/docs/hub/security-tokens), and then store the token in a file `~/.cache/huggingface/token`.

Until you get access to the model, you can use the [NousResearch version](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) without a token by setting `base="NousResearch/Llama-2-7b-chat-hf"` in the `Llama` constructor.

```python
from language_model.t5 import T5

t5 = T5(
    # Specify all hyperparameters here
    #   (or don't, and defaults will be used)
)
```

You do not need any special permissions to create at T5 model. All examples below use the `llama` object, but you can replace it with `t5` if you are using a T5 model instead.

### Generate Text

```python
generated = llama.generate("Please list 5 unique mammals.")
print(generated)
```

### Batch Generation

```python
prompts = [
    "Please list 5 unique mammals.",
    "Please list 5 unique birds.",
    "Please list 5 unique reptiles."
]
generations = llama.generate(prompts)
for p, g in zip(prompts, generations):
    print(p, g)
```

Make sure to set `gen_batch_size` to a value > 1 to get a speedup from batch generation (at the cost of additional memory usage).


### Training

```python
data = [
    ("Please list 5 unique mammals.", "Elephant, Giraffe, Lion, Tiger, Bear"),
    ("Please list 5 unique birds.", "Eagle, Hawk, Sparrow, Pigeon, Parrot"),
    ("Please list 5 unique reptiles.", "Snake, Lizard, Crocodile, Turtle, Tortoise")
]
llama.train(data)
```

You can also customize the training loop using the `.training` method:

```python
for perplexity in llama.training(data):
    print('Perplexity:', perplexity)
    ... # code to run after each epoch
```

Set the parameter `epochs` to determine the number of training iterations.

You can also use `.training(data, yield_every_x_epochs=0.2)` or similar for the loop to return several times each epoch (such as after completing each 20% of each epoch in this example).


### Save and Load Trained Model

```python
llama.save('my_new_model')

loaded_llama = Llama(base='my_new_model')
```

Loading a saved model will keep all saved parameters and hyperparameters. You can override loaded hyperparameters by passing them into the constructor during loading.


### Calculate Perplexity

```python
ppl = llama.perplexity(data)
```

### Default Parameters and Performance Considerations

#### General

`train_batch_size=1` and `gen_batch_size=1`, with `gradient_accumulation_steps=1`. Increase batch size to speed up at the cost of higher memory utilization.

Consider reducing the default values of `max_sequence_length`, since they usually are larger than necessary for a task, so consider reducing to trigger left-side input truncation. To protect a minimum amount of input from being truncated, the `protected_input_length` parameter (default `512`) can be used to prevent right-side input tokens from being truncated (the right-hand side of output tokens will be truncated instead during training).

`max_output_length=512`, but generating this amount is somewhat slow.

Training is sensitive to `learning_rate`, so consider adjusting for your training task.

Generation is sensitive to `repetition_penalty` and `num_beams` (increasing `num_beams` increases memory cost but may improve generation quality).

#### Llama

Important default settings:

The 7 billion parameter chat variant of Llama2 is the default base model. This can be changed using the `base` and/or `param_magnitude` params to specify a different base model.

`nf4` quantization (`quantize` parameter can be set to `'nf4', 'int8', 'bf16', None`), which is fast and memory efficient but may sacrifice accuracy. 

Input and output is wrapped by the format `"[INST] <<SYS>> You are a helpful, respectful, and honest assistant. <</SYS>> {input} [/INST] {output} </s>"`. This format is expected by Llama2 and changing it can negatively affect the performance, but you can change it using the `format` parameter. 

Low Rank Adaptation (LoRA) specified as `lora=8`, but can be set to `None` to disable LoRA and make all parameters trainable.

By default LoRA applied to these modules: `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`





