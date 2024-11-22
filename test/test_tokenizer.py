import dataclasses as dc

from language_model.tokens import TokenSlot, Input
from language_model.utils.test import test
import language_model.tokens as lmt

import textwrap as tw



with test('Construct Tokenizers'):
    ll3_tokenizer = lmt.HuggingfaceTokenizer('meta-llama/Meta-Llama-3.1-8B-Instruct')
    ll2_tokenizer = lmt.HuggingfaceTokenizer('meta-llama/Llama-2-7b-chat-hf')


with test('Construct Templates', crash=True):

    @dc.dataclass
    class Turn(lmt.Template):
        template = "<|start_header_id|><role><|end_header_id|>\n\n<text><|eot_id|>"
        role: lmt.Slot = lmt.Input()
        text: lmt.Slot = lmt.Input(min=5)

    @dc.dataclass
    class Document(lmt.Template):
        template = "\n\nConsider the following document:\n\n# <title>\n\n<document_text>"
        title: lmt.Slot = lmt.Input()
        document_text: lmt.Slot = lmt.Input()

    @dc.dataclass
    class SystemRoleplayInstruction(lmt.Template):
        template = Turn(text=f"You are a <profession>. Respond in a <style> manner.{Document.template}")
        role: lmt.Slot = lmt.Input()
        profession: lmt.Slot = lmt.Input()
        style: lmt.Slot = lmt.Input()
        title: lmt.Slot = lmt.Input()
        document_text: lmt.Slot = lmt.Input()


    assert SystemRoleplayInstruction.template == tw.dedent('''
        <|start_header_id|><role><|end_header_id|>
        
        You are a <profession>. Respond in a <style> manner.
        
        Consider the following document:
        
        # <title>
        
        <document_text><|eot_id|>''').strip()
    assert {slot.name for slot in SystemRoleplayInstruction} == {
        'role', 'profession', 'style', 'title', 'document_text'}


with test('Construct Llama3 Templates Group'):

    @dc.dataclass
    class Llama3Templates(lmt.TemplateTokenizer):
        tokenizer = ll3_tokenizer
        sequence_prefix = '<|begin_of_text|>'
        turn: lmt.SegmentTemplate[Turn] = Turn
        document: lmt.SegmentTemplate[Document] = Document
        system_roleplay_instruction: lmt.SegmentTemplate[SystemRoleplayInstruction] = SystemRoleplayInstruction


with test('Configure Llama3 Templates Tokenization'):
    tokenizer = Llama3Templates(
        document=lmt.SegmentTemplate(
            template=Document(
                title=lmt.Input(truncatable=False),
                document_text=lmt.Input(min=8, max=16, trunc_side='R', trunc_rank=3)
            ),
            trunc_segment_rank=1,
        ),
        turn=lmt.SegmentTemplate(
            template=Turn(text=Input(max=20)),
            trunc_content=False,
            trunc_segment=True,
            trunc_segment_rank=2,
            trunc_segment_side='L',
        ),
        max_segments=35,
    )

    assert tokenizer.max_segments == 35
    assert tokenizer.document.template.title.truncatable is False
    assert tokenizer.tokenizer is ll3_tokenizer
    assert tokenizer.document.tokenizer is ll3_tokenizer


with test('Construct Llama3 Data'):
    instruction = Llama3Templates.system_roleplay_instruction(
        role="system", profession="primary care physician", style="natural",
        title="Patient's Symptoms", document_text="A check-up requires taking the patient's temperature, blood pressure, pulse rate, and asking the patient if they have any symptoms.")
    dialogue = [
        instruction,
        Llama3Templates.turn(role="user", text="I'm feeling a bit under the weather today."),
        Llama3Templates.turn(role="assistant", text="What seems to be the problem?"),
        Llama3Templates.turn(role="user", text="I've had a headache and a runny nose all day.")
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
    sequence = tokenizer.fill(dialogue)
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


with test('Tokenize with Truncation', crash=True):
    tokenizer = Llama3Templates(
        document=lmt.SegmentTemplate(
            template=SystemRoleplayInstruction(
                title=lmt.Input(truncatable=False),
                document_text=lmt.Input(min=8, max=16, trunc_side='R', trunc_rank=3)
            ),
            trunc_segment_rank=1,
        ),
        turn=lmt.SegmentTemplate(
            template=Turn(),
            trunc_content=False,
            trunc_segment=True,
            trunc_segment_rank=2,
            trunc_segment_side='L',
        ),
        max_length=32,
    )

    assert tokenizer.system_roleplay_instruction.template.title.truncatable is False

    sequence = tokenizer.fill(dialogue)
    print('|'.join(sequence.tokens()))







