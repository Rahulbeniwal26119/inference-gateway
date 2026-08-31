from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]
Stop = str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: Stop | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
