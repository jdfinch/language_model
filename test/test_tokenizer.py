
import ezpyzy as ez
from ezpyzy.test import test

import language_model.tokens as tok

import dataclasses as dc
import textwrap as tw



with test('Construct Tokenizers'):
    ll3_tokenizer = tok.HuggingfaceTokenizer(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct')
    ll2_tokenizer = tok.HuggingfaceTokenizer(repo_id='meta-llama/Llama-2-7b-chat-hf')


with test('Construct Templates'):

    @dc.dataclass
    class Turn(tok.Template):
        template = "<|start_header_id|><role><|end_header_id|>\n\n<text><|eot_id|>"
        role: tok.Slot = tok.Input()
        text: tok.Slot = tok.Input(min=5)

    @dc.dataclass
    class Document(tok.Template):
        template = "\n\nConsider the following document:\n\n# <title>\n\n<document_text>"
        title: tok.Slot = tok.Input()
        document_text: tok.Slot = tok.Input()

    @dc.dataclass
    class SystemRoleplayInstruction(tok.Template):
        template = Turn(text=f"You are a <profession>. Respond in a <style> manner.{Document.template}")
        role: tok.Slot = tok.Input()
        profession: tok.Slot = tok.Input()
        style: tok.Slot = tok.Input()
        title: tok.Slot = tok.Input()
        document_text: tok.Slot = tok.Input()


    assert SystemRoleplayInstruction.template == tw.dedent('''
        <|start_header_id|><role><|end_header_id|>
        
        You are a <profession>. Respond in a <style> manner.
        
        Consider the following document:
        
        # <title>
        
        <document_text><|eot_id|>''').strip()


with test('Construct Llama3 Templates Group'):
    @dc.dataclass
    class Llama3Templates(tok.Templates):
        tokenizer = ll3_tokenizer
        sequence_prefix = '<|begin_of_text|>'
        turn: tok.SegmentTemplate = Turn
        document: tok.SegmentTemplate = Document
        system_roleplay_instruction: tok.SegmentTemplate = SystemRoleplayInstruction
    templates = Llama3Templates()


with test('Configure Llama3 Templates Tokenization'):
    tokenizer = tok.TemplateTokenizer(
        templates=Llama3Templates(
            document=tok.SegmentTemplate(
                template=Document(
                    title=tok.Input(trunc=False),
                    document_text=tok.Input(min=8, max=16, trunc_side='R', trunc_rank=3)
                ),
                trunc_segment_rank=1,
            ),
            turn=tok.SegmentTemplate(
                template=Turn(text=tok.Input(max=20)),
                trunc_content=False,
                trunc_segment=True,
                trunc_segment_rank=2,
                trunc_segment_side='L',
            ),
        ),
        sequence_prefix='<|begin_of_text|>',
        max_segments=32,
        tokenizer=ll3_tokenizer
    )


with test('Construct Llama3 Data'):
    instruction = SystemRoleplayInstruction(
        role="system", profession="primary care physician", style="natural",
        title="Patient's Symptoms", document_text="A check-up requires taking the patient's temperature, blood pressure, pulse rate, and asking the patient if they have any symptoms.")
    dialogue = [
        instruction,
        Turn(role="user", text="I'm feeling a bit under the weather today."),
        Turn(role="assistant", text="What seems to be the problem?"),
        Turn(role="user", text="I've had a headache and a runny nose all day.")
    ]
    assert instruction.role == "system"
    assert instruction.profession == "primary care physician"
    assert instruction.style == "natural"
    assert instruction.title == "Patient's Symptoms"
    assert dialogue[-1].role == "user"
    assert dialogue[-1].text == "I've had a headache and a runny nose all day."
    assert str(instruction) == tw.dedent('''
        <|start_header_id|>system<|end_header_id|>
    
        You are a primary care physician. Respond in a natural manner.
        
        Consider the following document:
        
        # Patient's Symptoms
        
        A check-up requires taking the patient's temperature, blood pressure, pulse rate, and asking the patient if they have any symptoms.<|eot_id|>
    ''').strip()


with test('Tokenize Llama3 Data'):
    sequence = tokenizer.tokenize(dialogue)

    assert '|'.join(sequence.tokens()) == tw.dedent('''
        <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|
    
        |You| are| a| primary| care| physician|.| Respond| in| a| natural| manner|.
        
        |Consider| the| following| document|:
        
        |#| Patient|'s| Symptoms|
        
        |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|,| pulse| rate|,| and| asking| the| patient| if| they| have| any| symptoms|.|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
        
        |I|'m| feeling| a| bit| under| the| weather| today|.|<|eot_id|>|<|start_header_id|>|assistant|<|end_header_id|>|
        
        |What| seems| to| be| the| problem|?|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
        
        |I|'ve| had| a| headache| and| a| run|ny| nose| all| day|.|<|eot_id|>
    ''').strip()


with test('Tokenize with Truncation'):
    tokenizer = tok.TemplateTokenizer(
        templates=Llama3Templates(
            system_roleplay_instruction=tok.SegmentTemplate(
                template=SystemRoleplayInstruction(
                    title=tok.Input(trunc=False),
                    document_text=tok.Input(min=12, max=None, trunc_side='R', trunc_rank=3)
                ),
                trunc_segment=False,
            ),
            turn=tok.SegmentTemplate(
                template=Turn(text=tok.Input(max=20)),
                trunc_content=False,
                trunc_segment=True,
                trunc_segment_rank=2,
                trunc_segment_side='L',
            ),
        ),
        max_length=None,
        tokenizer=ll3_tokenizer
    )

    assert '|'.join(tokenizer.tokenize(dialogue).tokens()) == tw.dedent('''
    <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|

    |You| are| a| primary| care| physician|.| Respond| in| a| natural| manner|.
    
    |Consider| the| following| document|:
    
    |#| Patient|'s| Symptoms|
    
    |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|,| pulse| rate|,| and| asking| the| patient| if| they| have| any| symptoms|.|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
    
    |I|'m| feeling| a| bit| under| the| weather| today|.|<|eot_id|>|<|start_header_id|>|assistant|<|end_header_id|>|
    
    |What| seems| to| be| the| problem|?|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
    
    |I|'ve| had| a| headache| and| a| run|ny| nose| all| day|.|<|eot_id|>
    ''').strip()

    tokenizer.max_length = 90

    assert '|'.join(tokenizer.tokenize(dialogue).tokens()) == tw.dedent('''
    <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|

    |You| are| a| primary| care| physician|.| Respond| in| a| natural| manner|.
    
    |Consider| the| following| document|:
    
    |#| Patient|'s| Symptoms|
    
    |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|,| pulse| rate|...|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
    
    |I|'m| feeling| a| bit| under| the| weather| today|.|<|eot_id|>|<|start_header_id|>|assistant|<|end_header_id|>|
    
    |What| seems| to| be| the| problem|?|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
    
    |I|'ve| had| a| headache| and| a| run|ny| nose| all| day|.|<|eot_id|>
    ''').strip()

    tokenizer.max_length = 80

    assert '|'.join(tokenizer.tokenize(dialogue).tokens()) == tw.dedent('''
    <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|
    
    |You| are| a| primary| care| physician|.| Respond| in| a| natural| manner|.
    
    |Consider| the| following| document|:
    
    |#| Patient|'s| Symptoms|
    
    |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|,| pulse| rate|,| and| asking| the| patient|...|<|eot_id|>|<|start_header_id|>|assistant|<|end_header_id|>|
    
    |What| seems| to| be| the| problem|?|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
    
    |I|'ve| had| a| headache| and| a| run|ny| nose| all| day|.|<|eot_id|>
    ''').strip()

    tokenizer.max_length = 70

    assert '|'.join(tokenizer.tokenize(dialogue).tokens()) == tw.dedent('''
    <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|

    |You| are| a| primary| care| physician|.| Respond| in| a| natural| manner|.
    
    |Consider| the| following| document|:
    
    |#| Patient|'s| Symptoms|
    
    |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|,| pulse| rate|,| and| asking| the| patient| if| they|...|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
    
    |I|'ve| had| a| headache| and| a| run|ny| nose| all| day|.|<|eot_id|>
    ''').strip()

    tokenizer.max_length = 60

    assert '|'.join(tokenizer.tokenize(dialogue).tokens()) == tw.dedent('''
    <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|

    |You| are| a| primary| care| physician|.| Respond| in| a| natural| manner|.
    
    |Consider| the| following| document|:
    
    |#| Patient|'s| Symptoms|
    
    |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|...|<|eot_id|>|<|start_header_id|>|user|<|end_header_id|>|
    
    |I|'ve| had| a| headache| and| a| run|ny| nose| all| day|.|<|eot_id|>
    ''').strip()

    tokenizer.max_length = 50

    assert '|'.join(tokenizer.tokenize(dialogue).tokens()) == tw.dedent('''
    <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|

    |You| are| a| primary| care| physician|.| Respond| in| a| natural| manner|.
    
    |Consider| the| following| document|:
    
    |#| Patient|'s| Symptoms|
    
    |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|,| pulse| rate|,| and| asking| the| patient|...|<|eot_id|>
    ''').strip()

    tokenizer.max_length = 40

    assert '|'.join(tokenizer.tokenize(dialogue).tokens()) == tw.dedent('''
    <|begin_of_text|>|<|start_header_id|>|system|<|end_header_id|>|

    |You| are| a|...|.| Respond| in| a| natural| manner|.
    
    |Consider| the| following| document|:
    
    |#| Patient|'s| Symptoms|
    
    |A| check|-up| requires| taking| the| patient|'s| temperature|,| blood| pressure|...|<|eot_id|>
    ''').strip()



