"""Anthropic Messages API adapter.

The Messages API is not OpenAI-compatible: system prompts are a top-level
parameter, ``max_tokens`` is mandatory, stop sequences must be a list, streaming
is a named-event SSE protocol, and usage arrives in two halves. Those
differences are translated here so the application layer never sees them.
"""

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx

from gateway.adapters.providers.streaming import (
    ProviderTimeouts,
    iter_sse_frames,
    stream_with_timeouts,
    upstream_error_detail,
)
from gateway.domain.errors import (
    GatewayError,
    InvalidRequestError,
    UpstreamAuthenticationError,
    UpstreamConnectTimeoutError,
    UpstreamProtocolError,
    UpstreamRateLimitedError,
    UpstreamUnavailableError,
)
from gateway.domain.events import (
    ProviderEvent,
    StreamFinished,
    StreamStarted,
    TextDelta,
    UsageReported,
)
from gateway.domain.models import ChatRequest, Usage
from gateway.ports.provider import Capabilities

ANTHROPIC_VERSION = "2023-06-01"

# Anthropic stop reasons that have an honest OpenAI equivalent. Reasons tied to
# capabilities V1 does not expose (``tool_use``, ``pause_turn``) are deliberately
# absent: reporting them as "stop" would claim the answer is complete.
FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}

ERROR_TYPES = {
    "authentication_error": UpstreamAuthenticationError,
    "permission_error": UpstreamAuthenticationError,
    "rate_limit_error": UpstreamRateLimitedError,
    "invalid_request_error": UpstreamProtocolError,
}


class AnthropicProvider:
    """Translate the Anthropic Messages SSE protocol into domain events."""

    capabilities = Capabilities(
        requires_max_tokens=True,
        supports_leading_assistant_message=False,
    )

    def __init__(
        self,
        *,
        name: str,
        messages_url: str,
        upstream_model: str,
        api_key: str | None,
        connect_timeout: float,
        timeouts: ProviderTimeouts,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.messages_url = messages_url
        self.upstream_model = upstream_model
        self.api_key = api_key
        self.timeouts = timeouts
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout, read=None, write=connect_timeout, pool=5
            ),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            follow_redirects=False,
            headers={"user-agent": "inference-gateway/0.1.0"},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        return stream_with_timeouts(self._raw_events(request), self.timeouts)

    async def _raw_events(self, request: ChatRequest) -> AsyncGenerator[ProviderEvent, None]:
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with self._client.stream(
                "POST",
                self.messages_url,
                headers=headers,
                json=self._request_payload(request),
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise self._status_error(response.status_code, body)

                state = _StreamState()
                frames = iter_sse_frames(response.aiter_lines())

                try:
                    async for frame in frames:
                        for event in self._handle(frame.data, state):
                            yield event
                        if state.stopped:
                            break
                finally:
                    await frames.aclose()

                if not state.stopped or not state.started or not state.finished:
                    raise UpstreamProtocolError
        except GatewayError:
            raise
        except httpx.ConnectTimeout as exc:
            raise UpstreamConnectTimeoutError from exc
        except httpx.ConnectError as exc:
            raise UpstreamUnavailableError from exc
        except httpx.TimeoutException as exc:
            raise UpstreamUnavailableError("The upstream transport timed out.") from exc
        except httpx.HTTPError as exc:
            raise UpstreamUnavailableError from exc

    def _handle(self, data: str, state: "_StreamState") -> list[ProviderEvent]:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise UpstreamProtocolError from exc
        if not isinstance(payload, dict):
            raise UpstreamProtocolError

        kind = payload.get("type")
        if kind == "error":
            raise self._stream_error(payload)
        if kind == "ping":
            return []
        if kind == "message_start":
            return [self._message_start(payload, state)]
        if kind == "content_block_delta":
            return self._content_block_delta(payload, state)
        if kind == "message_delta":
            return self._message_delta(payload, state)
        if kind == "message_stop":
            if not state.finished:
                raise UpstreamProtocolError
            state.stopped = True
            return state.usage_events()
        if kind in {"content_block_start", "content_block_stop"}:
            # Thinking and tool blocks are structural; only their text deltas,
            # filtered below, can reach the client.
            return []
        raise UpstreamProtocolError

    def _message_start(self, payload: dict[str, Any], state: "_StreamState") -> ProviderEvent:
        if state.started:
            raise UpstreamProtocolError
        message = payload.get("message")
        if not isinstance(message, dict):
            raise UpstreamProtocolError

        completion_id = message.get("id")
        if not isinstance(completion_id, str) or not completion_id:
            raise UpstreamProtocolError

        usage = message.get("usage")
        if isinstance(usage, dict):
            state.prompt_tokens = _optional_token_count(usage, "input_tokens")

        state.started = True
        return StreamStarted(
            completion_id=completion_id,
            created=int(time.time()),
            upstream_model=str(message.get("model") or self.upstream_model),
        )

    def _content_block_delta(
        self, payload: dict[str, Any], state: "_StreamState"
    ) -> list[ProviderEvent]:
        if not state.started:
            raise UpstreamProtocolError
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            raise UpstreamProtocolError

        # Only visible assistant text crosses the boundary. Thinking deltas are
        # billed reasoning with no place in the chat completion schema, and tool
        # input deltas belong to a capability V1 does not expose.
        if delta.get("type") != "text_delta":
            return []

        text = delta.get("text")
        if not isinstance(text, str):
            raise UpstreamProtocolError
        if state.finished:
            raise UpstreamProtocolError
        return [TextDelta(text)] if text else []

    def _message_delta(self, payload: dict[str, Any], state: "_StreamState") -> list[ProviderEvent]:
        if not state.started or state.finished:
            raise UpstreamProtocolError

        usage = payload.get("usage")
        if isinstance(usage, dict):
            state.completion_tokens = _optional_token_count(usage, "output_tokens")

        delta = payload.get("delta")
        if not isinstance(delta, dict):
            raise UpstreamProtocolError
        stop_reason = delta.get("stop_reason")
        if stop_reason is None:
            return []
        if not isinstance(stop_reason, str) or stop_reason not in FINISH_REASONS:
            raise UpstreamProtocolError

        state.finished = True
        return [StreamFinished(FINISH_REASONS[stop_reason])]

    def _request_payload(self, request: ChatRequest) -> dict[str, Any]:
        if request.max_tokens is None:
            raise InvalidRequestError("The Anthropic Messages API requires max_tokens.")

        system = "\n\n".join(
            message.content for message in request.messages if message.role == "system"
        )
        payload: dict[str, Any] = {
            "model": self.upstream_model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
                if message.role != "system"
            ],
            "stream": True,
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            payload["stop_sequences"] = (
                list(request.stop) if isinstance(request.stop, tuple) else [request.stop]
            )
        return payload

    @staticmethod
    def _status_error(status_code: int, body: bytes = b"") -> GatewayError:
        if status_code == 400:
            # A 400 describes the caller's own request, so relaying the
            # upstream's wording turns an opaque rejection into a fixable one.
            detail = upstream_error_detail(body)
            message = "The upstream provider rejected the request."
            return InvalidRequestError(
                f"{message} {detail}" if detail else message, upstream_status=status_code
            )
        if status_code in {401, 403}:
            return UpstreamAuthenticationError(upstream_status=status_code)
        if status_code == 429:
            return UpstreamRateLimitedError(upstream_status=status_code)
        return UpstreamUnavailableError(upstream_status=status_code)

    @staticmethod
    def _stream_error(payload: dict[str, Any]) -> GatewayError:
        error = payload.get("error")
        error_type = error.get("type") if isinstance(error, dict) else None
        factory = ERROR_TYPES.get(str(error_type), UpstreamUnavailableError)
        return factory()


class _StreamState:
    __slots__ = ("completion_tokens", "finished", "prompt_tokens", "started", "stopped")

    def __init__(self) -> None:
        self.started = False
        self.finished = False
        self.stopped = False
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    def usage_events(self) -> list[ProviderEvent]:
        """Anthropic reports usage in two halves; report it only if both arrived."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return []
        return [
            UsageReported(
                Usage(
                    prompt_tokens=self.prompt_tokens,
                    completion_tokens=self.completion_tokens,
                    total_tokens=self.prompt_tokens + self.completion_tokens,
                )
            )
        ]


def _optional_token_count(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UpstreamProtocolError
    return value
