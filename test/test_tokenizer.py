
import dataclasses as dc
from language_model_v2.utils.test import test
from language_model_v2.tokenizer import Tokenizer

import textwrap as tw


with test('Construct Tokenizers'):
    ll3_tokenizer = Tokenizer('meta-llama/Meta-Llama-3.1-8B-Instruct')
    ll2_tokenizer = Tokenizer('meta-llama/Llama-2-7b-chat-hf')

with test('Construct Token Templates'):
    ll3_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a]#
    
    ''').strip())

    print(ll3_template.display())

    ll2_template = ll2_tokenizer.templatize(tw.dedent('''
    
    [INST] <<SYS>> You are a helpful, respectful, and honest assistant. <</SYS>> #[input=q]# [/INST] #[output=a]#
    
    ''').strip())

    print(ll2_template.display())


with test('Fill Input Only'):
    ll3_input = ll3_template.fill(q='What is the capital of France?')
    print(ll3_input.display())

    ll2_input = ll2_template.fill(q='What is the capital of France?')
    print(ll2_input.display())


with test('Fill Input and Output'):
    ll3_input_output = ll3_template.fill(q='What is the capital of France?', a='Paris')
    print(ll3_input_output.display())

    ll2_input_output = ll2_template.fill(q='What is the capital of France?', a='Paris')
    print(ll2_input_output.display())



with test('Triple Slot Template'):
    ts_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a, eos=]# ... no end in sight ...
    
    ''').strip())
    print(ts_template.display())
    ts_1 = ts_template.fill(sys='Please answer without hallucinating.')
    print(ts_1.display())
    ts_2 = ts_template.fill(sys='Please answer without hallucinating.', q='What is the capital of France?')
    print(ts_2.display())
    ts_3 = ts_template.fill(sys="Please answer without hallucinating", q='What is the capital of France?', a='Paris')
    print(ts_3.display())
    ts_4 = ts_template.fill(q='What is the capital of France?', a='Paris')
    print(ts_4.display())


with test('Default Truncation'):
    short_template = ts_template.copy(max_length=30)
    print(short_template.display())
    short_seq = short_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='The capital of France is Paris.')
    print(short_seq.display())


with test('More Truncation'):
    shorter_template = ts_template.copy(max_length=25)
    print(shorter_template.display())
    shorter_seq = shorter_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='The capital of France is Paris.')
    print(shorter_seq.display())


with test('Protect Slot From Truncation'):
    ...







