import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from prometheus_client import CollectorRegistry, generate_latest
from pydantic import SecretStr

from gateway.config import Settings
from gateway.domain.errors import UpstreamRateLimitedError, UpstreamUnavailableError
from gateway.domain.events import ProviderEvent, StreamStarted, TextDelta
from gateway.domain.models import ChatRequest
from gateway.main import create_app
from gateway.ports.provider import Capabilities


class FailingProvider:
    name = "fake"
    capabilities = Capabilities()

    def __init__(self, *, mid_stream: bool) -> None:
        self.mid_stream = mid_stream

    def stream(self, _request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        async def events() -> AsyncIterator[ProviderEvent]:
            if not self.mid_stream:
                raise UpstreamRateLimitedError
            yield StreamStarted("chatcmpl-failure", 1_700_000_000, "fake")
            yield TextDelta("partial")
            raise UpstreamUnavailableError

        return events()


@asynccontextmanager
async def make_client(
    settings: Settings, provider: FailingProvider
) -> AsyncIterator[httpx.AsyncClient]:
    application = create_app(settings=settings, provider=provider, registry=CollectorRegistry())
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            yield client


async def test_non_streaming_completion(
    client: httpx.AsyncClient, chat_payload: dict[str, object]
) -> None:
    response = await client.post("/v1/chat/completions", json=chat_payload)

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.json() == {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gateway-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


async def test_streaming_completion(
    client: httpx.AsyncClient, chat_payload: dict[str, object]
) -> None:
    chat_payload["stream"] = True
    response = await client.post("/v1/chat/completions", json=chat_payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"delta":{"role":"assistant"}' in response.text
    assert '"delta":{"content":"hello"}' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


async def test_unsupported_field_returns_openai_error(
    client: httpx.AsyncClient, chat_payload: dict[str, object]
) -> None:
    chat_payload["tools"] = []
    response = await client.post("/v1/chat/completions", json=chat_payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["param"] == "tools"


async def test_unknown_model_returns_404(
    client: httpx.AsyncClient, chat_payload: dict[str, object]
) -> None:
    chat_payload["model"] = "unknown"
    response = await client.post("/v1/chat/completions", json=chat_payload)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_gateway_bearer_authentication(
    settings: Settings, chat_payload: dict[str, object]
) -> None:
    protected = settings.model_copy(update={"api_key": SecretStr("secret")})
    application = create_app(
        settings=protected,
        provider=FailingProvider(mid_stream=False),
        registry=CollectorRegistry(),
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway.test"
        ) as protected_client:
            rejected = await protected_client.post("/v1/chat/completions", json=chat_payload)
            accepted = await protected_client.post(
                "/v1/chat/completions",
                json=chat_payload,
                headers={"authorization": "Bearer secret"},
            )

    assert rejected.status_code == 401
    assert accepted.status_code == 429


async def test_pre_stream_failure_keeps_http_error(
    settings: Settings, chat_payload: dict[str, object]
) -> None:
    async with make_client(settings, FailingProvider(mid_stream=False)) as client:
        response = await client.post("/v1/chat/completions", json={**chat_payload, "stream": True})

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "upstream_rate_limited"


async def test_mid_stream_failure_is_explicit_sse_error(
    settings: Settings, chat_payload: dict[str, object]
) -> None:
    async with make_client(settings, FailingProvider(mid_stream=True)) as client:
        response = await client.post("/v1/chat/completions", json={**chat_payload, "stream": True})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"code":"upstream_unavailable"' in response.text
    assert "[DONE]" not in response.text


async def test_health_and_metrics(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health/live")).json() == {"status": "ok"}
    assert (await client.get("/health/ready")).json() == {"status": "ready"}
    metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    assert "gateway_requests_total" in metrics.text


class HangingProvider:
    """Streams two events, then keeps generating until the consumer gives up."""

    name = "fake"
    capabilities = Capabilities()

    def __init__(self) -> None:
        self.closed = asyncio.Event()

    def stream(self, _request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        async def events() -> AsyncIterator[ProviderEvent]:
            try:
                yield StreamStarted("chatcmpl-hang", 1_700_000_000, "fake")
                yield TextDelta("first")
                await asyncio.Event().wait()
            finally:
                self.closed.set()

        return events()


async def test_client_disconnect_stops_upstream_generation(
    settings: Settings, chat_payload: dict[str, object]
) -> None:
    """A disconnect must stop upstream work; abandoned generation is still billed."""
    provider = HangingProvider()
    application = create_app(settings=settings, provider=provider, registry=CollectorRegistry())

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:

            async def consume() -> None:
                async with client.stream(
                    "POST", "/v1/chat/completions", json={**chat_payload, "stream": True}
                ) as response:
                    assert response.status_code == 200
                    async for _chunk in response.aiter_bytes():
                        break

            # An ASGI server cancels the request task when the peer goes away.
            request_task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task

            await asyncio.wait_for(provider.closed.wait(), timeout=2)

    exported = generate_latest(application.state.gateway_metrics.registry).decode()
    assert 'gateway_client_cancellations_total{model="gateway-model"' in exported
    assert 'outcome="cancelled"' in exported
    assert 'gateway_active_requests{model="gateway-model",provider="fake",stream="true"} 0.0' in (
        exported
    )


async def test_reported_usage_becomes_token_metrics(
    client: httpx.AsyncClient, chat_payload: dict[str, object]
) -> None:
    assert (await client.post("/v1/chat/completions", json=chat_payload)).status_code == 200

    exported = (await client.get("/metrics")).text

    assert 'gateway_upstream_tokens_total{direction="prompt",model="gateway-model",' in exported
    assert 'gateway_upstream_tokens_total{direction="completion",model="gateway-model",' in exported
    assert 'gateway_upstream_responses_total{model="gateway-model",provider="fake",' in exported


async def test_upstream_status_class_is_recorded_on_failure(
    settings: Settings, chat_payload: dict[str, object]
) -> None:
    application = create_app(
        settings=settings,
        provider=FailingProvider(mid_stream=False),
        registry=CollectorRegistry(),
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            assert (await client.post("/v1/chat/completions", json=chat_payload)).status_code == 429

    exported = generate_latest(application.state.gateway_metrics.registry).decode()
    # The provider failed before any HTTP response was attributed to it.
    assert 'status_class="transport"' in exported
