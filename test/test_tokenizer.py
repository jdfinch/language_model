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

    assert 'What is the capital of France?' in ll3_input.text()
    assert 'What is the capital of France?' in ll2_input.text()
    assert ll3_input.tokens()[-1] != '<|eot_id|>'


with test('Fill Input and Output'):
    ll3_input_output = ll3_template.fill(q='What is the capital of France?', a='Paris')
    print(ll3_input_output.display())

    ll2_input_output = ll2_template.fill(q='What is the capital of France?', a='Paris')
    print(ll2_input_output.display())

    assert 'What is the capital of France?' in ll3_input_output.text()
    assert 'What is the capital of France?' in ll2_input_output.text()
    assert 'Paris' in ll3_input_output.text()
    assert 'Paris' in ll2_input_output.text()
    assert ll3_input_output.tokens()[-1] == '<|eot_id|>'
    assert ll2_input_output.tokens()[-1] == '</s>'



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

    text = short_seq.text()
    assert 'Please answer without hallucinating' in text
    assert 'What is the capital of France?' in text
    assert 'The capital of France is Paris.' in text


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

    text = short_seq.text()
    assert 'Please answer without hallucinating' in text
    assert 'What is the capital of France?' in text
    assert 'The capital of France is Paris.' not in text
    assert 'The capital of' in text


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

    assert len(rt_seq) == 25
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

    assert len(rt_seq) == 20
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

    assert len(rt_seq) == 24
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

    assert len(rt_seq) == 26
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


with test('Construct Template Collection'):
    templates = ll3_tokenizer.templatize({
        'system_prompt (trunc_content=False, trunc_segment=False)':
            '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n#[input=text]#<|eot_id|>',
        'document (trunc_segment=False)':
            '<|start_header_id|>user<|end_header_id|>\n\nPlease base your answers on the doc:\n\n#[input=text, min=8, trunc_side=R]#<|eot_id|>',
        'user_instruction (trunc_content=False)':
            '<|start_header_id|>user<|end_header_id|>\n\n#[input=text]#<|eot_id|>',
        'assistant_response (trunc_content=False)':
            '<|start_header_id|>assistant<|end_header_id|>\n\n#[output=text]#',
        'assistant_prefix (trunc_content=False)':
            '<|start_header_id|>assistant<|end_header_id|>\n\n#[input=text]#',
        'assistant_continuation':
            '#[output=text, min=3]#',
    })

    for name, template in templates.templates.items():
        print(name, f'{template.trunc_segment=}', f'{template.trunc_content=}')
        print(template.display())


with test('Dialogue Sequence'):
    dialogue = [
        dict(temp='system_prompt', text='You are a helpful assistant.'),
        dict(temp='document', text='France is a country in Europe. It has a long history.'),
        dict(temp='user_instruction', text='Hi there, can you help me?'),
        dict(temp='assistant_response', text='Of course! How can I help you today?'),
        dict(temp='user_instruction', text='What is the capital of France?'),
        dict(temp='assistant_response', text='The capital of France is Paris.'),
    ]
    tokens = templates.fill(dialogue)
    print(tokens.display())

with test('Dialogue Sequence with Small Truncation'):
    templates_with_truncation = dc.replace(templates, max_length=87)
    tokens = templates_with_truncation.fill(dialogue)
    print(tokens.display())

with test('Dialogue Sequence with More Truncation'):
    templates_with_truncation = dc.replace(templates, max_length=80)
    tokens = templates_with_truncation.fill(dialogue)
    print(tokens.display())

with test('Dialogue Sequence with Large Truncation'):
    templates_with_truncation = dc.replace(templates, max_length=60)
    tokens = templates_with_truncation.fill(dialogue)
    print(tokens.display())

with test('Batch Dialogue Tokenization with Large Truncation'):
    dialogue2 = [
        dict(temp='system_prompt', text='You are helpful.'),
        dict(temp='document', text='France is a country.'),
        dict(temp='user_instruction', text='Can you help?'),
        dict(temp='assistant_response', text='Of course!'),
        dict(temp='user_instruction', text='What is the capital of France?'),
        dict(temp='assistant_response', text='It is Paris!'),
    ]
    dialogue3 = [
        dict(temp='system_prompt', text='You are a helpful assistant.'),
        dict(temp='document', text='France is a country in Europe. It has a long history and people speak French there.'),
        dict(temp='user_instruction', text='Hi there!'),
        dict(temp='assistant_response', text='How can I help you today? I am a helpful assistant.'),
        dict(temp='user_instruction', text='What is the capital of France?'),
        dict(temp='assistant_response', text='The capital of France is Paris. Would you like to know more?'),
    ]
    dialogues = [dialogue, dialogue2, dialogue3]
    temps = dc.replace(templates_with_truncation, max_length=90, pad_side='R')
    tokens = temps.fill(dialogues)
    print(tokens.display())










