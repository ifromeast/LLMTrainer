from dataclasses import dataclass
from enum import auto, Enum
import json

TOOL_PROMPT = '你是，请尽可能回答用户的问题。你有以下工具可以使用:\n'


class Role(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()
    INTERPRETER = auto()
    OBSERVATION = auto()

    def __str__(self):
        match self:
            case Role.SYSTEM:
                return "<reserved_105>"
            case Role.USER:
                return "<reserved_106>"
            case Role.ASSISTANT | Role.TOOL | Role.INTERPRETER:
                return "<reserved_107>"
            case Role.OBSERVATION:
                return "<reserved_108>"


@dataclass
class Conversation:
    role: Role
    content: str
    tool: str | None = None

    def __str__(self) -> str:
        print(self.role, self.content, self.tool)
        match self.role:
            case Role.SYSTEM | Role.USER | Role.ASSISTANT | Role.OBSERVATION:
                return f'{self.role}\n{self.content}'
            case Role.TOOL:
                return f'{self.role}{self.tool}\n{self.content}'
            case Role.INTERPRETER:
                return f'{self.role}interpreter\n{self.content}'

    # Human readable format
    def get_text(self) -> str:
        text = postprocess_text(self.content)
        match self.role.value:
            case Role.TOOL.value:
                text = f'Calling tool `{self.tool}`:\n{text}'
            case Role.INTERPRETER.value:
                text = f'{text}'
            case Role.OBSERVATION.value:
                text = f'Observation:\n```\n{text}\n```'
        return text


def preprocess_text(
        system: str | None,
        tools: list[dict] | None,
        history: list[Conversation],
) -> str:
    if tools:
        tools = json.dumps(tools, indent=4, ensure_ascii=False)

    prompt = f"{Role.SYSTEM}\n"
    prompt += system if not tools else TOOL_PROMPT
    if tools:
        tools = json.loads(tools)
        prompt += json.dumps(tools, ensure_ascii=False)
    for conversation in history:
        prompt += f'{conversation}'
    prompt += f'{Role.ASSISTANT}\n'
    return prompt


def postprocess_text(text: str) -> str:
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|observation|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    return text.strip()