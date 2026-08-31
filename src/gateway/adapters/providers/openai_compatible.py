import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from gateway.domain.errors import (
    GatewayError,
    InvalidRequestError,
    UpstreamAuthenticationError,
    UpstreamConnectTimeoutError,
    UpstreamFirstTokenTimeoutError,
    UpstreamIdleTimeoutError,
    UpstreamProtocolError,
    UpstreamRateLimitedError,
    UpstreamTotalTimeoutError,
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


@dataclass(frozen=True, slots=True)
class ProviderTimeouts:
    first_token: float
    idle: float
    total: float


class OpenAICompatibleProvider:
    """Translate an OpenAI-compatible upstream SSE stream into domain events."""

    def __init__(
        self,
        *,
        name: str,
        chat_url: str,
        upstream_model: str,
        api_key: str | None,
        connect_timeout: float,
        timeouts: ProviderTimeouts,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.chat_url = chat_url
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
        return self._stream_with_timeouts(request)

    async def _stream_with_timeouts(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        started_at = time.monotonic()
        first_event = True
        raw_events = self._raw_events(request)

        try:
            while True:
                total_remaining = self.timeouts.total - (time.monotonic() - started_at)
                if total_remaining <= 0:
                    raise UpstreamTotalTimeoutError

                phase_timeout = self.timeouts.first_token if first_event else self.timeouts.idle
                total_is_limiting = total_remaining <= phase_timeout
                timeout = min(total_remaining, phase_timeout)

                try:
                    event = await asyncio.wait_for(anext(raw_events), timeout=timeout)
                except StopAsyncIteration:
                    return
                except TimeoutError as exc:
                    if total_is_limiting:
                        raise UpstreamTotalTimeoutError from exc
                    if first_event:
                        raise UpstreamFirstTokenTimeoutError from exc
                    raise UpstreamIdleTimeoutError from exc

                first_event = False
                yield event
        finally:
            await raw_events.aclose()

    async def _raw_events(self, request: ChatRequest) -> AsyncGenerator[ProviderEvent, None]:
        headers = {"accept": "text/event-stream", "content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"

        try:
            async with self._client.stream(
                "POST",
                self.chat_url,
                headers=headers,
                json=self._request_payload(request),
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise self._status_error(response.status_code)

                started = False
                finished = False
                saw_done = False
                data_lines: list[str] = []

                async for line in response.aiter_lines():
                    if line == "":
                        if not data_lines:
                            continue

                        data = "\n".join(data_lines)
                        data_lines.clear()
                        if data == "[DONE]":
                            saw_done = True
                            break

                        chunk, events = self._parse_chunk(data)
                        if not started:
                            started = True
                            yield StreamStarted(
                                completion_id=self._completion_id(chunk),
                                created=self._created(chunk),
                                upstream_model=str(chunk.get("model") or self.upstream_model),
                            )

                        for event in events:
                            if isinstance(event, StreamFinished):
                                if finished:
                                    raise UpstreamProtocolError
                                finished = True
                            elif isinstance(event, TextDelta) and finished:
                                raise UpstreamProtocolError
                            yield event
                        continue

                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))

                if data_lines or not saw_done or not started or not finished:
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

    def _request_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.upstream_model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = (
                list(request.stop) if isinstance(request.stop, tuple) else request.stop
            )
        return payload

    @staticmethod
    def _status_error(status_code: int) -> GatewayError:
        if status_code == 400:
            return InvalidRequestError(
                "The upstream provider rejected the request.", upstream_status=status_code
            )
        if status_code in {401, 403}:
            return UpstreamAuthenticationError(upstream_status=status_code)
        if status_code == 429:
            return UpstreamRateLimitedError(upstream_status=status_code)
        return UpstreamUnavailableError(upstream_status=status_code)

    @staticmethod
    def _parse_chunk(data: str) -> tuple[dict[str, Any], list[ProviderEvent]]:
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise UpstreamProtocolError from exc

        if not isinstance(chunk, dict) or "error" in chunk:
            raise UpstreamProtocolError

        choices = chunk.get("choices")
        usage_data = chunk.get("usage")
        if not isinstance(choices, list):
            raise UpstreamProtocolError
        if not choices and usage_data is None:
            raise UpstreamProtocolError

        events: list[ProviderEvent] = []
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict):
                raise UpstreamProtocolError
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise UpstreamProtocolError
            if "tool_calls" in delta or "function_call" in delta:
                raise UpstreamProtocolError

            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise UpstreamProtocolError
                if content:
                    events.append(TextDelta(content))

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if not isinstance(finish_reason, str):
                    raise UpstreamProtocolError
                events.append(StreamFinished(finish_reason))

        if usage_data is not None:
            events.append(UsageReported(OpenAICompatibleProvider._parse_usage(usage_data)))
        return chunk, events

    @staticmethod
    def _parse_usage(value: object) -> Usage:
        if not isinstance(value, dict):
            raise UpstreamProtocolError
        prompt_tokens = OpenAICompatibleProvider._usage_int(value, "prompt_tokens")
        completion_tokens = OpenAICompatibleProvider._usage_int(value, "completion_tokens")
        total_tokens = OpenAICompatibleProvider._usage_int(value, "total_tokens")
        return Usage(prompt_tokens, completion_tokens, total_tokens)

    @staticmethod
    def _usage_int(value: dict[object, object], key: str) -> int:
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise UpstreamProtocolError
        return item

    @staticmethod
    def _completion_id(chunk: dict[str, Any]) -> str:
        completion_id = chunk.get("id")
        if completion_id is None:
            return f"chatcmpl-{uuid.uuid4().hex}"
        if not isinstance(completion_id, str) or not completion_id:
            raise UpstreamProtocolError
        return completion_id

    @staticmethod
    def _created(chunk: dict[str, Any]) -> int:
        created = chunk.get("created")
        if created is None:
            return int(time.time())
        if not isinstance(created, int) or isinstance(created, bool) or created < 0:
            raise UpstreamProtocolError
        return created
