from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from prometheus_client import CollectorRegistry
from pydantic import SecretStr

from gateway.config import Settings
from gateway.domain.errors import UpstreamRateLimitedError, UpstreamUnavailableError
from gateway.domain.events import ProviderEvent, StreamStarted, TextDelta
from gateway.domain.models import ChatRequest
from gateway.main import create_app


class FailingProvider:
    name = "fake"

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
