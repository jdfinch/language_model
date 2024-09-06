
import transformers as hf
from language_model.utils.test import test


with test('Construct Tokenizers'):
    tokenizer = hf.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

with test('Construct Chat'):
    chat = [
        dict(role='system', content='You are a helpful assistant.'),
        dict(role='assistant', content='Hi! How can I help you today?'),
        dict(role='user', content='What is the capital of France?'),
        dict(role='assistant', content='The capital of France is Paris.'),
        dict(role='user', content='Thank you!'),
        dict(role='assistant', content='You are welcome!'),
    ]
    output = tokenizer.apply_chat_template(chat)
    print('||'.join(repr(tokenizer.decode(tok_id)) for tok_id in output))
    print(f'Sequence length: {len(output)}')


with test('Chat with Truncation'):
    tokenizer.truncation_side = 'left'
    output = tokenizer.apply_chat_template(chat, max_length=50, truncation=True)
    print('||'.join(repr(tokenizer.decode(tok_id)) for tok_id in output))
    print(f'Sequence length: {len(output)}')


with test('Chat with Documents'):
    documents = [
        dict(title='History of France', content='France is a country in Europe. It has a long history.'),
        dict(title='News from Paris', content='Paris is a beautiful city.'),
    ]
    output = tokenizer.apply_chat_template(chat, documents=documents)
    print('||'.join(repr(tokenizer.decode(tok_id)) for tok_id in output))
    print(f'Sequence length: {len(output)}')



