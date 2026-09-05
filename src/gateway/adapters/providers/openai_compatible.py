"""OpenAI-compatible provider adapter.

The upstream responds with Server-Sent Events (SSE).  Each event is one or
more ``data:`` lines followed by a blank line; comment lines beginning with
``:`` are ignored.  The payload of each ``data:`` line is JSON in the usual
OpenAI chat-completions streaming shape, for example::

    data: {"id":"chatcmpl-123", "choices":[{"delta":{"content":"Hello"}}]}

    data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}

    data: {"choices":[], "usage":{"prompt_tokens":8,"completion_tokens":1,"total_tokens":9}}

    data: [DONE]

This adapter converts those frames into the gateway's start, text, usage, and
finished events. Tool/function-call deltas are outside the current public
contract and are rejected rather than silently discarded.
"""

import json
import time
import uuid
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

__all__ = ["OpenAICompatibleProvider", "ProviderTimeouts"]

MAX_FRAME_DETAIL = 400


def _frame_detail(reason: str, data: str) -> str:
    """Name a protocol failure and quote a bounded prefix of the frame."""
    snippet = " ".join(data.split())
    if len(snippet) > MAX_FRAME_DETAIL:
        snippet = snippet[:MAX_FRAME_DETAIL].rstrip() + "…"
    return f"{reason}: {snippet}"


class OpenAICompatibleProvider:
    """Translate an OpenAI-compatible upstream SSE stream into domain events."""

    capabilities = Capabilities()

    def __init__(
        self,
        *,
        name: str,
        chat_url: str,
        upstream_model: str,
        api_key: str | None,
        connect_timeout: float,
        timeouts: ProviderTimeouts,
        max_tokens_field: str = "max_tokens",
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure one OpenAI-compatible upstream connection.

        ``chat_url`` is the upstream ``/chat/completions`` endpoint and
        ``upstream_model`` is the model name sent to it. ``name`` identifies
        this adapter locally. ``api_key`` becomes a Bearer token when present.
        ``connect_timeout`` limits connect and request-write work; ``timeouts``
        limits the streamed response's first event, idle periods, and lifetime.
        ``max_tokens_field`` names the request field carrying the output cap.

        If ``client`` is supplied, this provider uses but never closes it. When
        absent, it creates a client with the supplied connection-pool limits
        and closes that client from :meth:`aclose`.
        """
        self.name = name
        self.chat_url = chat_url
        self.upstream_model = upstream_model
        self.api_key = api_key
        self.timeouts = timeouts
        # OpenAI's reasoning models reject ``max_tokens`` outright and require
        # ``max_completion_tokens``; every other OpenAI-compatible upstream
        # still expects the original name. The registry decides which.
        self.max_tokens_field = max_tokens_field
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
        """Close the HTTP client created by this provider.

        A caller-provided ``httpx.AsyncClient`` belongs to its caller and is
        intentionally left open. This method returns after owned connections
        have been released.
        """
        if self._owns_client:
            await self._client.aclose()

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        """Return the timed gateway event stream for one validated chat request.

        ``request`` contains the public model alias, messages, and optional
        generation settings. The returned asynchronous iterator emits
        ``StreamStarted``, zero or more ``TextDelta`` events, optional
        ``UsageReported``, and one ``StreamFinished`` event. It also enforces
        the configured first-event, idle, and total-stream timeouts.
        """
        return stream_with_timeouts(self._raw_events(request), self.timeouts)

    async def _raw_events(self, request: ChatRequest) -> AsyncGenerator[ProviderEvent, None]:
        """Send ``request`` upstream and translate its SSE frames into events.

        The request is converted to the OpenAI-compatible JSON body documented
        by :meth:`_request_payload`. Successful ``data:`` frames must contain
        chunks accepted by :meth:`_parse_chunk`, followed by ``data: [DONE]``.
        The generator yields the domain events in stream order and translates
        upstream HTTP, connection, and protocol failures into ``GatewayError``
        subclasses. Closing this generator releases the upstream response.
        """
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
                    body = await response.aread()
                    raise self._status_error(response.status_code, body)

                started = False
                finished = False
                saw_done = False
                frames = iter_sse_frames(response.aiter_lines())

                try:
                    async for frame in frames:
                        if frame.data == "[DONE]":
                            saw_done = True
                            break

                        try:
                            chunk, events = self._parse_chunk(frame.data)
                        except UpstreamProtocolError as exc:
                            # An opaque protocol error is unactionable. Quote the
                            # frame that caused it so the log names the cause.
                            raise UpstreamProtocolError(
                                detail=_frame_detail("unsupported chunk", frame.data)
                            ) from exc
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
                                    raise UpstreamProtocolError(
                                        detail=_frame_detail("second finish_reason", frame.data)
                                    )
                                finished = True
                            elif isinstance(event, TextDelta) and finished:
                                raise UpstreamProtocolError(
                                    detail=_frame_detail("content after finish_reason", frame.data)
                                )
                            yield event
                finally:
                    await frames.aclose()

                if not saw_done or not started or not finished:
                    raise UpstreamProtocolError(
                        detail=(
                            f"incomplete stream: done={saw_done} "
                            f"started={started} finished={finished}"
                        )
                    )
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
        """Convert a gateway ``ChatRequest`` to an upstream JSON request body.

        For example, a user message becomes::

            {"model": "upstream-model", "messages": [{"role": "user",
             "content": "Hello"}], "stream": true,
             "stream_options": {"include_usage": true}}

        The public model alias in ``request.model`` is replaced by this
        provider's configured ``upstream_model``. Optional temperature, top-p,
        max-token, and stop settings are included only when supplied. The
        returned dictionary is passed to ``httpx`` as its JSON body.
        """
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
            payload[self.max_tokens_field] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = (
                list(request.stop) if isinstance(request.stop, tuple) else request.stop
            )
        return payload

    @staticmethod
    def _status_error(status_code: int, body: bytes = b"") -> GatewayError:
        """Map an upstream HTTP status to the gateway error returned to clients.

        ``400`` becomes ``InvalidRequestError``; ``401`` and ``403`` become
        ``UpstreamAuthenticationError``; ``429`` becomes
        ``UpstreamRateLimitedError``; every other error status becomes
        ``UpstreamUnavailableError``. Each returned error retains
        ``status_code`` as its upstream status.
        """
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
    def _parse_chunk(data: str) -> tuple[dict[str, Any], list[ProviderEvent]]:
        """Parse one upstream SSE ``data:`` value into its chunk and events.

        ``data`` must be JSON such as::

            {"choices": [{"delta": {"content": "Hello"},
                          "finish_reason": null}]}

        A text delta produces ``TextDelta("Hello")``. A non-null string
        ``finish_reason`` produces ``StreamFinished``. A usage-only final
        chunk has ``{"choices": [], "usage": {...}}`` and produces
        ``UsageReported``. The returned tuple preserves the decoded chunk for
        stream metadata and lists the events represented by it. Malformed JSON,
        unsupported tool/function calls, and invalid shapes raise
        ``UpstreamProtocolError``.
        """
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
        """Return ``Usage`` from an upstream ``usage`` object.

        ``value`` must have this JSON-object shape::

            {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}

        The returned ``Usage`` contains those three non-negative integer
        counts. Missing fields, booleans, negative numbers, and other types
        raise ``UpstreamProtocolError``.
        """
        if not isinstance(value, dict):
            raise UpstreamProtocolError
        prompt_tokens = OpenAICompatibleProvider._usage_int(value, "prompt_tokens")
        completion_tokens = OpenAICompatibleProvider._usage_int(value, "completion_tokens")
        total_tokens = OpenAICompatibleProvider._usage_int(value, "total_tokens")
        return Usage(prompt_tokens, completion_tokens, total_tokens)

    @staticmethod
    def _usage_int(value: dict[object, object], key: str) -> int:
        """Return one required non-negative integer field from a usage object.

        ``value`` is the decoded ``usage`` JSON object and ``key`` is one of
        ``prompt_tokens``, ``completion_tokens``, or ``total_tokens``. The
        method returns that count and rejects missing, boolean, negative, and
        non-integer values with ``UpstreamProtocolError``.
        """
        item = value.get(key)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise UpstreamProtocolError
        return item

    @staticmethod
    def _completion_id(chunk: dict[str, Any]) -> str:
        """Return the completion identifier from a decoded upstream chunk.

        The optional field appears as ``{"id": "chatcmpl-123", ...}``.
        Its non-empty string value is returned unchanged. If an otherwise valid
        provider chunk omits ``id``, a ``chatcmpl-`` identifier is generated so
        the gateway can still emit a valid start event. Other values raise
        ``UpstreamProtocolError``.
        """
        completion_id = chunk.get("id")
        if completion_id is None:
            return f"chatcmpl-{uuid.uuid4().hex}"
        if not isinstance(completion_id, str) or not completion_id:
            raise UpstreamProtocolError
        return completion_id

    @staticmethod
    def _created(chunk: dict[str, Any]) -> int:
        """Return an upstream completion chunk's creation time as Unix seconds.

        ``chunk`` is one JSON object decoded from an upstream SSE ``data:``
        payload. The first payload commonly includes metadata in this form::

            {"id": "chatcmpl-123", "created": 1720000000,
             "model": "model-a", "choices": [{"delta": {"content": "Hi"}}]}

        The method returns the non-negative integer in ``chunk["created"]``.
        When an otherwise valid provider response omits that optional field, it
        returns the gateway's current Unix time so the public response still
        has a creation timestamp. A non-integer or negative value is invalid
        upstream protocol data and raises ``UpstreamProtocolError``.
        """
        created = chunk.get("created")
        if created is None:
            return int(time.time())
        if not isinstance(created, int) or isinstance(created, bool) or created < 0:
            raise UpstreamProtocolError
        return created
