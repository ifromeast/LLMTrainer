import json
import torch
import torch.nn.functional as F
from function_calling.tool_registry import get_tools

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def exist_and_not_none(d, key):
    return key in d and d[key] is not None


SYSTEM_PROMPT = '你是携程智能旅行助手问道，请尽可能回答用户的问题。你有以下工具可以使用:\n'

def get_system_prompt():
    tools = get_tools()
    tools = json.dumps(tools, indent=4, ensure_ascii=False)
    prompt = SYSTEM_PROMPT
    tools = json.loads(tools)
    prompt += json.dumps(tools, ensure_ascii=False)
    return prompt

# if __name__ == "__main__":
#     prompt = get_system_prompt()
#     print(prompt)
