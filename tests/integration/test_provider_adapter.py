import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from gateway.adapters.providers.openai_compatible import OpenAICompatibleProvider
from gateway.adapters.providers.streaming import ProviderTimeouts
from gateway.domain.errors import (
    UpstreamConnectTimeoutError,
    UpstreamFirstTokenTimeoutError,
    UpstreamIdleTimeoutError,
    UpstreamProtocolError,
    UpstreamRateLimitedError,
    UpstreamTotalTimeoutError,
)
from gateway.domain.events import StreamFinished, StreamStarted, TextDelta, UsageReported
from gateway.domain.models import ChatRequest, Message

SUCCESS_SSE = b"".join(
    [
        (
            b'data: {"id":"chatcmpl-1","created":123,"model":"private",'
            b'"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
        ),
        (
            b'data: {"id":"chatcmpl-1","created":123,"model":"private",'
            b'"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        ),
        (
            b'data: {"id":"chatcmpl-1","created":123,"model":"private","choices":[],'
            b'"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]
)


class SlowStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay: float) -> None:
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            if index or not chunk:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def request() -> ChatRequest:
    return ChatRequest("public", (Message("user", "hello"),), max_tokens=10)


def provider(
    handler: httpx.AsyncBaseTransport | httpx.MockTransport,
    *,
    first_token: float = 1,
    idle: float = 1,
) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=handler)
    return OpenAICompatibleProvider(
        name="test",
        chat_url="http://upstream.test/v1/chat/completions",
        upstream_model="private",
        api_key="provider-secret",
        connect_timeout=1,
        timeouts=ProviderTimeouts(first_token=first_token, idle=idle, total=5),
        client=client,
    )


@pytest.mark.asyncio
async def test_adapter_translates_request_and_successful_stream() -> None:
    captured: dict[str, object] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(http_request.content))
        assert http_request.headers["authorization"] == "Bearer provider-secret"
        return httpx.Response(
            200, content=SUCCESS_SSE, headers={"content-type": "text/event-stream"}
        )

    adapter = provider(httpx.MockTransport(handler))
    events = [event async for event in adapter.stream(request())]

    assert captured["model"] == "private"
    assert captured["stream"] is True
    assert [type(event) for event in events] == [
        StreamStarted,
        TextDelta,
        StreamFinished,
        UsageReported,
    ]


@pytest.mark.asyncio
async def test_adapter_maps_429_before_first_event() -> None:
    adapter = provider(httpx.MockTransport(lambda _request: httpx.Response(429)))

    with pytest.raises(UpstreamRateLimitedError):
        await anext(adapter.stream(request()))


@pytest.mark.asyncio
async def test_adapter_rejects_malformed_and_truncated_sse() -> None:
    malformed = provider(
        httpx.MockTransport(lambda _request: httpx.Response(200, content=b"data: nope\n\n"))
    )
    truncated = provider(
        httpx.MockTransport(
            lambda _request: httpx.Response(
                200, content=SUCCESS_SSE.removesuffix(b"data: [DONE]\n\n")
            )
        )
    )

    with pytest.raises(UpstreamProtocolError):
        await anext(malformed.stream(request()))
    with pytest.raises(UpstreamProtocolError):
        _ = [event async for event in truncated.stream(request())]


@pytest.mark.asyncio
async def test_adapter_enforces_first_token_timeout() -> None:
    slow = SlowStream([b"", SUCCESS_SSE], delay=0.05)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, stream=slow, headers={"content-type": "text/event-stream"}
        )
    )
    adapter = provider(transport, first_token=0.01)

    with pytest.raises(UpstreamFirstTokenTimeoutError):
        await anext(adapter.stream(request()))
    assert slow.closed


@pytest.mark.asyncio
async def test_adapter_maps_connection_timeout() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("deliberate", request=request)

    adapter = provider(httpx.MockTransport(timeout))

    with pytest.raises(UpstreamConnectTimeoutError):
        await anext(adapter.stream(request()))


@pytest.mark.asyncio
async def test_cancelling_consumer_closes_upstream_response() -> None:
    first_sse = SUCCESS_SSE.split(b"\n\n", 1)[0] + b"\n\n"
    hanging = SlowStream([first_sse, b""], delay=10)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, stream=hanging, headers={"content-type": "text/event-stream"}
        )
    )
    adapter = provider(transport)
    stream = adapter.stream(request())
    assert isinstance(await anext(stream), StreamStarted)
    assert isinstance(await anext(stream), TextDelta)

    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert hanging.closed


def frames_of(payload: bytes) -> list[bytes]:
    return [frame + b"\n\n" for frame in payload.split(b"\n\n") if frame]


@pytest.mark.asyncio
async def test_adapter_enforces_idle_timeout() -> None:
    """The gap between events is limited independently of the first-token wait."""
    first_frame, *rest = frames_of(SUCCESS_SSE)
    slow = SlowStream([first_frame, b"".join(rest)], delay=10)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, stream=slow, headers={"content-type": "text/event-stream"}
        )
    )
    adapter = provider(transport, first_token=5, idle=0.01)
    stream = adapter.stream(request())

    assert isinstance(await anext(stream), StreamStarted)
    assert isinstance(await anext(stream), TextDelta)
    with pytest.raises(UpstreamIdleTimeoutError):
        await anext(stream)
    assert slow.closed


@pytest.mark.asyncio
async def test_adapter_enforces_total_timeout() -> None:
    """A stream that never idles for long can still exceed its wall-clock budget."""
    slow = SlowStream(frames_of(SUCCESS_SSE), delay=0.15)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, stream=slow, headers={"content-type": "text/event-stream"}
        )
    )
    client = httpx.AsyncClient(transport=transport)
    adapter = OpenAICompatibleProvider(
        name="test",
        chat_url="http://upstream.test/v1/chat/completions",
        upstream_model="private",
        api_key=None,
        connect_timeout=1,
        timeouts=ProviderTimeouts(first_token=5, idle=5, total=0.2),
        client=client,
    )

    with pytest.raises(UpstreamTotalTimeoutError):
        _ = [event async for event in adapter.stream(request())]
    assert slow.closed
