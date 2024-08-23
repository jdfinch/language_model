
from language_model_v2.prompt_format import prompt_format


def SystemTurn(prompt):
    """System turn format for Llama3, to be passed into llama3format.Chat(*previous_turns)"""
    return f"<|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>"

def UserTurn(prompt):
    """User turn format for Llama3, to be passed into llama3format.Chat(*previous_turns)"""
    return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"

def AssistantTurn(response):
    """Assistant turn format for Llama3, to be passed into llama3format.Chat(*previous_turns)"""
    return f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"

@prompt_format
def Chat(*previous_turns):
    """Chat format for Llama3 including multiple turns. Use SystemTurn, UserTurn, and AssistantTurn to format individual previous_turns."""
    return f"<|begin_of_text|>{''.join(previous_turns)}<|start_header_id|>assistant<|end_header_id|>\n\n"

@prompt_format
def Instruction(user_instruction, system_instruction="You are a helpful assistant."):
    """Single instruction prompt format for Llama3. Pass in a user_instruction and an optional system_instruction as strings (without special formatting-- separator and format tokens will be added automatically)."""
    return f"<|begin_of_text|>{SystemTurn(system_instruction)}{UserTurn(user_instruction)}<|start_header_id|>assistant<|end_header_id|>\n\n"


if __name__ == '__main__':
    print(Chat.format)