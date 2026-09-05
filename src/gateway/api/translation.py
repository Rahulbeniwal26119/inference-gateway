import json
from typing import Any

from gateway.domain.errors import GatewayError, UpstreamProtocolError
from gateway.domain.events import (
    ProviderEvent,
    StreamFinished,
    StreamStarted,
    TextDelta,
    UsageReported,
)
from gateway.domain.models import Usage


def error_payload(error: GatewayError) -> dict[str, dict[str, str | None]]:
    return {
        "error": {
            "message": error.message,
            "type": error.error_type,
            "param": None,
            "code": error.code,
        }
    }


def error_sse(error: GatewayError) -> bytes:
    data = json.dumps(error_payload(error), separators=(",", ":"))
    return f"event: error\ndata: {data}\n\n".encode()


class CompletionAccumulator:
    def __init__(self, public_model: str) -> None:
        self.public_model = public_model
        self.completion_id: str | None = None
        self.created: int | None = None
        self.parts: list[str] = []
        self.finish_reason: str | None = None
        self.usage: Usage | None = None

    def add(self, event: ProviderEvent) -> None:
        if isinstance(event, StreamStarted):
            if self.completion_id is not None:
                raise UpstreamProtocolError
            self.completion_id = event.completion_id
            self.created = event.created
            return
        if self.completion_id is None:
            raise UpstreamProtocolError
        if isinstance(event, TextDelta):
            if self.finish_reason is not None:
                raise UpstreamProtocolError
            self.parts.append(event.text)
        elif isinstance(event, StreamFinished):
            if self.finish_reason is not None:
                raise UpstreamProtocolError
            self.finish_reason = event.finish_reason
        elif isinstance(event, UsageReported):
            # Gemini repeats cumulative usage on every chunk where OpenAI sends
            # it once at the end. The last report is the total either way.
            self.usage = event.usage

    def response(self) -> dict[str, Any]:
        if self.completion_id is None or self.created is None or self.finish_reason is None:
            raise UpstreamProtocolError
        response: dict[str, Any] = {
            "id": self.completion_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.public_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(self.parts)},
                    "finish_reason": self.finish_reason,
                }
            ],
        }
        if self.usage is not None:
            response["usage"] = usage_payload(self.usage)
        return response


class OpenAIStreamEncoder:
    def __init__(self, public_model: str) -> None:
        self.public_model = public_model
        self.completion_id: str | None = None
        self.created: int | None = None
        self.finished = False
        self.usage: Usage | None = None

    def encode(self, event: ProviderEvent) -> bytes:
        if isinstance(event, StreamStarted):
            if self.completion_id is not None:
                raise UpstreamProtocolError
            self.completion_id = event.completion_id
            self.created = event.created
            return self._chunk(
                [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            )

        if self.completion_id is None or self.created is None:
            raise UpstreamProtocolError
        if isinstance(event, TextDelta):
            if self.finished:
                raise UpstreamProtocolError
            return self._chunk(
                [{"index": 0, "delta": {"content": event.text}, "finish_reason": None}]
            )
        if isinstance(event, StreamFinished):
            if self.finished:
                raise UpstreamProtocolError
            self.finished = True
            return self._chunk([{"index": 0, "delta": {}, "finish_reason": event.finish_reason}])
        if isinstance(event, UsageReported):
            # Held rather than forwarded: an upstream that repeats cumulative
            # usage would otherwise emit a usage chunk per token batch. The
            # public contract is one usage chunk, last, carrying the total.
            self.usage = event.usage
            return b""
        raise UpstreamProtocolError

    def done(self) -> bytes:
        if not self.finished:
            raise UpstreamProtocolError
        tail = b"data: [DONE]\n\n"
        if self.usage is None:
            return tail
        return self._chunk([], usage=self.usage) + tail

    def _chunk(self, choices: list[dict[str, Any]], usage: Usage | None = None) -> bytes:
        payload: dict[str, Any] = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.public_model,
            "choices": choices,
        }
        if usage is not None:
            payload["usage"] = usage_payload(usage)
        data = json.dumps(payload, separators=(",", ":"))
        return f"data: {data}\n\n".encode()


def usage_payload(usage: Usage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
