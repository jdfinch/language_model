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


with test('Truncation Specified No Effect'):
    base_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a]#
    
    ''').strip())
    short_template = base_template.copy(max_length=50)
    print(short_template.display())
    short_seq = short_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='The capital of France is Paris.')
    print(short_seq.display())


with test('Default Truncation'):
    base_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a]#
    
    ''').strip())
    short_template = base_template.copy(max_length=30)
    print(short_template.display())
    short_seq = short_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='The capital of France is Paris.')
    print(short_seq.display())


with test('More Truncation'):
    shorter_template = base_template.copy(max_length=25)
    print(shorter_template.display())
    shorter_seq = shorter_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='The capital of France is Paris.')
    print(shorter_seq.display())

    assert shorter_seq.tokens(strip=True)[5:8] == ['without', 'halluc', 'inating']
    assert shorter_seq.tokens(strip=True)[13:20] == ['What', 'is', 'the', 'capital', 'of', 'France', '?']
    assert shorter_seq.tokens(strip=True)[-1] == '\n\n'


with test('Right Truncate'):
    rt_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys,trunc_side=R]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a]#
    
    ''').strip(), max_length=25)
        
    rt_seq = rt_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='The capital of France is Paris.')
    print(rt_seq.display())

    assert rt_seq.tokens(strip=True)[5:8] == ['Please', 'answer', 'without']
    assert rt_seq.tokens(strip=True)[13:20] == ['What', 'is', 'the', 'capital', 'of', 'France', '?']
    assert rt_seq.tokens()[-1] == '\n\n'


with test('Right Truncate Into User'):
    rt_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q,trunc_side=R]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a]#
    
    ''').strip(), max_length=20)
        
    rt_seq = rt_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='The capital of France is Paris.')
    print(rt_seq.display())

    assert rt_seq.tokens(strip=True)[10:15] == ['What', 'is', 'the', 'capital', 'of']
    assert rt_seq.tokens()[-1] == '\n\n'


with test('Exact Protect Slot From Truncation'):
    rt_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys,min=2,trunc_side=R]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q,min=6,trunc_side=R]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a,min=1,trunc_side=R]#
    
    ''').strip(), max_length=24)
        
    rt_seq = rt_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='Paris.')
    print(rt_seq.display())

    assert rt_seq.tokens(strip=True)[5:7] == ['Please', 'answer']
    assert rt_seq.tokens(strip=True)[12:18] == ['What', 'is', 'the', 'capital', 'of', 'France']
    assert rt_seq.tokens()[-1] == 'Paris'


with test('Conservative Protect Slot From Truncation'):
    rt_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys,min=2,trunc_side=R]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q,min=6,trunc_side=R]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a,min=1,trunc_side=R]#
    
    ''').strip(), max_length=26)
        
    rt_seq = rt_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='Paris.')
    print(rt_seq.display())

    assert rt_seq.tokens(strip=True)[5:8] == ['Please', 'answer', 'without']
    assert rt_seq.tokens(strip=True)[13:20] == ['What', 'is', 'the', 'capital', 'of', 'France', '?']
    assert rt_seq.tokens()[-1] == 'Paris'


with test('Incompatible Max Length and Slot Protection Length', raises=ValueError):
    rt_template = ll3_tokenizer.templatize(tw.dedent('''
    
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    #[input=sys,min=2,trunc_side=R]#<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    #[input=q,min=6,trunc_side=R]#<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    #[output=a,min=1,trunc_side=R]#
    
    ''').strip(), max_length=20)
        
    rt_seq = rt_template.fill(
        sys="Please answer without hallucinating",
        q='What is the capital of France?',
        a='Paris.')


# todo - changing slot truncation priority








