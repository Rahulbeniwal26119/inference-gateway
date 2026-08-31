from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gateway.domain.models import ChatRequest, Message


class ChatMessageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessageInput] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: str | list[str] | None) -> str | list[str] | None:
        if isinstance(value, str) and not value:
            raise ValueError("stop must not be empty")
        if isinstance(value, list):
            if not 1 <= len(value) <= 4:
                raise ValueError("stop must contain between one and four strings")
            if any(not item for item in value):
                raise ValueError("stop strings must not be empty")
        return value

    def to_domain(self) -> ChatRequest:
        stop = tuple(self.stop) if isinstance(self.stop, list) else self.stop
        return ChatRequest(
            model=self.model,
            messages=tuple(Message(role=item.role, content=item.content) for item in self.messages),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=stop,
        )

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be blank")
        return value
