import json
from typing import Any

import httpx
import pytest

from gateway.adapters.providers.anthropic import AnthropicProvider
from gateway.adapters.providers.streaming import ProviderTimeouts
from gateway.domain.errors import (
    UpstreamAuthenticationError,
    UpstreamProtocolError,
    UpstreamRateLimitedError,
    UpstreamUnavailableError,
)
from gateway.domain.events import StreamFinished, StreamStarted, TextDelta, UsageReported
from gateway.domain.models import ChatRequest, Message, Usage


def sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


MESSAGE_START = sse(
    "message_start",
    {
        "type": "message_start",
        "message": {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [],
            "usage": {"input_tokens": 7, "output_tokens": 0},
        },
    },
)
BLOCK_START = sse(
    "content_block_start",
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
)
BLOCK_STOP = sse("content_block_stop", {"type": "content_block_stop", "index": 0})
MESSAGE_STOP = sse("message_stop", {"type": "message_stop"})


def text_delta(text: str) -> bytes:
    return sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    )


def message_delta(stop_reason: str, output_tokens: int | None = 11) -> bytes:
    usage = {} if output_tokens is None else {"output_tokens": output_tokens}
    return sse(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": usage},
    )


def stream_bytes(stop_reason: str = "end_turn", output_tokens: int | None = 11) -> bytes:
    return b"".join(
        [
            MESSAGE_START,
            BLOCK_START,
            text_delta("Hello"),
            text_delta(" world"),
            BLOCK_STOP,
            message_delta(stop_reason, output_tokens),
            MESSAGE_STOP,
        ]
    )


def request(**overrides: Any) -> ChatRequest:
    fields: dict[str, Any] = {
        "model": "public",
        "messages": (Message("user", "hello"),),
        "max_tokens": 64,
    }
    fields.update(overrides)
    return ChatRequest(**fields)


def provider(
    handler: httpx.MockTransport, *, first_token: float = 1, idle: float = 1
) -> AnthropicProvider:
    return AnthropicProvider(
        name="test",
        messages_url="http://upstream.test/v1/messages",
        upstream_model="claude-opus-5",
        api_key="provider-secret",
        connect_timeout=1,
        timeouts=ProviderTimeouts(first_token=first_token, idle=idle, total=5),
        client=httpx.AsyncClient(transport=handler),
    )


def responding(body: bytes, status_code: int = 200) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda _request: httpx.Response(
            status_code, content=body, headers={"content-type": "text/event-stream"}
        )
    )


@pytest.mark.asyncio
async def test_hoists_system_prompt_and_sends_anthropic_headers() -> None:
    captured: dict[str, Any] = {}
    seen_headers: dict[str, str] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        seen_headers.update(http_request.headers)
        return httpx.Response(
            200, content=stream_bytes(), headers={"content-type": "text/event-stream"}
        )

    adapter = provider(httpx.MockTransport(handler))
    chat = request(
        messages=(
            Message("system", "Be brief."),
            Message("user", "hello"),
            Message("system", "Be kind."),
        ),
        temperature=0.5,
        top_p=0.9,
        stop=("STOP",),
    )
    _ = [event async for event in adapter.stream(chat)]

    # System turns become the top-level parameter; the API key is not a bearer token.
    assert captured["system"] == "Be brief.\n\nBe kind."
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["model"] == "claude-opus-5"
    assert captured["max_tokens"] == 64
    assert captured["stream"] is True
    assert captured["stop_sequences"] == ["STOP"]
    assert captured["temperature"] == 0.5
    assert captured["top_p"] == 0.9
    assert seen_headers["x-api-key"] == "provider-secret"
    assert seen_headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in seen_headers


@pytest.mark.asyncio
async def test_scalar_stop_becomes_a_stop_sequence_list() -> None:
    captured: dict[str, Any] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        return httpx.Response(
            200, content=stream_bytes(), headers={"content-type": "text/event-stream"}
        )

    adapter = provider(httpx.MockTransport(handler))
    _ = [event async for event in adapter.stream(request(stop="END"))]

    assert captured["stop_sequences"] == ["END"]


@pytest.mark.asyncio
async def test_translates_a_successful_stream_into_domain_events() -> None:
    adapter = provider(responding(stream_bytes()))

    events = [event async for event in adapter.stream(request())]

    assert events == [
        StreamStarted(
            completion_id="msg_01", created=events[0].created, upstream_model="claude-opus-5"
        ),
        TextDelta("Hello"),
        TextDelta(" world"),
        StreamFinished("stop"),
        UsageReported(Usage(prompt_tokens=7, completion_tokens=11, total_tokens=18)),
    ]


@pytest.mark.asyncio
async def test_thinking_deltas_never_reach_the_client() -> None:
    body = b"".join(
        [
            MESSAGE_START,
            sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "internal reasoning"},
                },
            ),
            BLOCK_STOP,
            BLOCK_START,
            text_delta("visible"),
            BLOCK_STOP,
            message_delta("end_turn"),
            MESSAGE_STOP,
        ]
    )
    adapter = provider(responding(body))

    events = [event async for event in adapter.stream(request())]

    assert [event for event in events if isinstance(event, TextDelta)] == [TextDelta("visible")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
    ],
)
async def test_maps_stop_reasons(stop_reason: str, finish_reason: str) -> None:
    adapter = provider(responding(stream_bytes(stop_reason)))

    events = [event async for event in adapter.stream(request())]

    assert StreamFinished(finish_reason) in events


@pytest.mark.asyncio
async def test_rejects_stop_reasons_v1_cannot_represent() -> None:
    adapter = provider(responding(stream_bytes("tool_use")))

    with pytest.raises(UpstreamProtocolError):
        _ = [event async for event in adapter.stream(request())]


@pytest.mark.asyncio
async def test_omitted_usage_still_produces_a_valid_stream() -> None:
    adapter = provider(responding(stream_bytes(output_tokens=None)))

    events = [event async for event in adapter.stream(request())]

    assert not [event for event in events if isinstance(event, UsageReported)]
    assert StreamFinished("stop") in events


@pytest.mark.asyncio
async def test_truncated_stream_is_a_protocol_error() -> None:
    adapter = provider(responding(stream_bytes().removesuffix(MESSAGE_STOP)))

    with pytest.raises(UpstreamProtocolError):
        _ = [event async for event in adapter.stream(request())]


@pytest.mark.asyncio
async def test_malformed_frame_is_a_protocol_error() -> None:
    adapter = provider(responding(b"event: message_start\ndata: {nope}\n\n"))

    with pytest.raises(UpstreamProtocolError):
        await anext(adapter.stream(request()))


@pytest.mark.asyncio
async def test_mid_stream_error_event_maps_to_a_gateway_category() -> None:
    body = b"".join(
        [
            MESSAGE_START,
            BLOCK_START,
            text_delta("partial"),
            sse(
                "error",
                {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
            ),
        ]
    )
    adapter = provider(responding(body))
    stream = adapter.stream(request())

    assert isinstance(await anext(stream), StreamStarted)
    assert isinstance(await anext(stream), TextDelta)
    with pytest.raises(UpstreamRateLimitedError):
        await anext(stream)


@pytest.mark.asyncio
async def test_overloaded_error_event_maps_to_unavailable() -> None:
    body = MESSAGE_START + sse(
        "error", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    )
    adapter = provider(responding(body))
    stream = adapter.stream(request())

    assert isinstance(await anext(stream), StreamStarted)
    with pytest.raises(UpstreamUnavailableError):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, UpstreamAuthenticationError),
        (429, UpstreamRateLimitedError),
        (529, UpstreamUnavailableError),
    ],
)
async def test_maps_status_codes_before_the_first_event(
    status_code: int, expected: type[Exception]
) -> None:
    adapter = provider(responding(b'{"type":"error"}', status_code=status_code))

    with pytest.raises(expected):
        await anext(adapter.stream(request()))


@pytest.mark.asyncio
async def test_upstream_body_is_not_exposed_to_the_caller() -> None:
    adapter = provider(responding(b'{"error":{"message":"customer prompt leaked"}}', 429))

    with pytest.raises(UpstreamRateLimitedError) as caught:
        await anext(adapter.stream(request()))

    assert "customer prompt leaked" not in caught.value.message
    assert caught.value.upstream_status == 429
