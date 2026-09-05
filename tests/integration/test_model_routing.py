"""Multi-model routing and the endpoints the request console depends on."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from prometheus_client import CollectorRegistry

from gateway.config import Settings
from gateway.domain.events import (
    ProviderEvent,
    StreamFinished,
    StreamStarted,
    TextDelta,
    UsageReported,
)
from gateway.domain.models import ChatRequest, Usage
from gateway.main import create_app
from gateway.ports.provider import Capabilities


class EchoProvider:
    """Reports the model it was handed so a test can prove where a request went."""

    capabilities = Capabilities()

    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[ChatRequest] = []

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        self.seen.append(request)

        async def events() -> AsyncIterator[ProviderEvent]:
            yield StreamStarted("chatcmpl-route", 1_700_000_000, self.name)
            yield TextDelta(request.model)
            yield StreamFinished("stop")
            yield UsageReported(Usage(1, 1, 2))

        return events()


def routed_settings(**overrides: object) -> Settings:
    return Settings(
        models=[
            {"name": "alpha", "kind": "anthropic", "upstream_model": "claude-opus-5"},
            {"name": "beta", "kind": "openai", "upstream_model": "gpt-x"},
        ],
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture
async def routed() -> AsyncIterator[tuple[httpx.AsyncClient, dict[str, EchoProvider]]]:
    providers = {"alpha": EchoProvider("anthropic"), "beta": EchoProvider("openai")}
    application = create_app(
        settings=routed_settings(dev_console=True),
        providers=providers,  # type: ignore[arg-type]
        registry=CollectorRegistry(),
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            yield client, providers


async def test_each_model_reaches_its_own_provider(
    routed: tuple[httpx.AsyncClient, dict[str, EchoProvider]],
) -> None:
    client, providers = routed

    for name in ("alpha", "beta"):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": name, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        assert response.json()["model"] == name

    assert [request.model for request in providers["alpha"].seen] == ["alpha"]
    assert [request.model for request in providers["beta"].seen] == ["beta"]


async def test_an_unrouted_model_is_rejected_without_reaching_a_provider(
    routed: tuple[httpx.AsyncClient, dict[str, EchoProvider]],
) -> None:
    client, providers = routed

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "gamma", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert not providers["alpha"].seen and not providers["beta"].seen


async def test_unrouted_models_share_one_metrics_series(
    routed: tuple[httpx.AsyncClient, dict[str, EchoProvider]],
) -> None:
    """A caller must not be able to create unbounded label cardinality."""
    client, _ = routed

    for name in ("nope-1", "nope-2"):
        await client.post(
            "/v1/chat/completions",
            json={"model": name, "messages": [{"role": "user", "content": "hi"}]},
        )

    metrics = (await client.get("/metrics")).text
    assert 'model="unrouted"' in metrics
    assert 'model="nope-1"' not in metrics and 'model="nope-2"' not in metrics


async def test_models_endpoint_lists_every_routable_model(
    routed: tuple[httpx.AsyncClient, dict[str, EchoProvider]],
) -> None:
    client, _ = routed

    body = (await client.get("/v1/models")).json()

    assert body["object"] == "list"
    assert [item["id"] for item in body["data"]] == ["alpha", "beta"]
    assert [item["owned_by"] for item in body["data"]] == ["anthropic", "openai"]


async def test_console_is_served_only_when_enabled() -> None:
    providers = {"alpha": EchoProvider("anthropic"), "beta": EchoProvider("openai")}
    application = create_app(
        settings=routed_settings(dev_console=False),
        providers=providers,  # type: ignore[arg-type]
        registry=CollectorRegistry(),
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            assert (await client.get("/__dev/console")).status_code == 404
            assert (await client.get("/__dev/config")).status_code == 404


async def test_console_config_exposes_the_timeouts_it_visualises(
    routed: tuple[httpx.AsyncClient, dict[str, EchoProvider]],
) -> None:
    client, _ = routed

    config = (await client.get("/__dev/config")).json()

    assert config["timeouts"]["idle_s"] == 20.0
    assert [item["upstream_model"] for item in config["models"]] == ["claude-opus-5", "gpt-x"]
    assert all(item["routable"] for item in config["models"])
    assert (await client.get("/__dev/console")).status_code == 200


class RepeatedUsageProvider:
    """An upstream that reports cumulative usage on every chunk, as Gemini does."""

    name = "gemini"
    capabilities = Capabilities()

    def stream(self, _request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        async def events() -> AsyncIterator[ProviderEvent]:
            yield StreamStarted("chatcmpl-cumulative", 1_700_000_000, "gemini-2.5-flash")
            yield TextDelta("one")
            yield UsageReported(Usage(4, 1, 5))
            yield TextDelta(" two")
            yield UsageReported(Usage(4, 2, 6))
            yield StreamFinished("stop")
            yield UsageReported(Usage(4, 3, 7))

        return events()


@pytest.fixture
async def cumulative() -> AsyncIterator[httpx.AsyncClient]:
    application = create_app(
        settings=Settings(
            models=[{"name": "flash", "kind": "gemini", "upstream_model": "gemini-2.5-flash"}]
        ),
        providers={"flash": RepeatedUsageProvider()},  # type: ignore[arg-type]
        registry=CollectorRegistry(),
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
            yield client


STREAMING_BODY = {
    "model": "flash",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
}


async def test_repeated_usage_does_not_break_the_stream(cumulative: httpx.AsyncClient) -> None:
    response = await cumulative.post("/v1/chat/completions", json=STREAMING_BODY)

    assert response.status_code == 200
    assert "upstream_protocol_error" not in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


async def test_the_client_sees_one_usage_chunk_carrying_the_total(
    cumulative: httpx.AsyncClient,
) -> None:
    text = (await cumulative.post("/v1/chat/completions", json=STREAMING_BODY)).text
    usages = [
        json.loads(line[6:])["usage"]
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]" and '"usage"' in line
    ]

    assert usages == [{"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}]


async def test_repeated_usage_is_counted_once(cumulative: httpx.AsyncClient) -> None:
    """Incrementing per report would multiply the counters by the chunk count."""
    await cumulative.post("/v1/chat/completions", json=STREAMING_BODY)

    metrics = (await cumulative.get("/metrics")).text

    assert (
        'gateway_upstream_tokens_total{direction="completion",model="flash",provider="gemini"} 3.0'
        in metrics
    )


async def test_repeated_usage_survives_the_non_streaming_path(
    cumulative: httpx.AsyncClient,
) -> None:
    payload = (
        await cumulative.post("/v1/chat/completions", json={**STREAMING_BODY, "stream": False})
    ).json()

    assert payload["choices"][0]["message"]["content"] == "one two"
    assert payload["usage"] == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
