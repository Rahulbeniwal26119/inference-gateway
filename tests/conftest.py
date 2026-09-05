import os
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


class SuccessfulProvider:
    name = "fake"
    capabilities = Capabilities()

    def stream(self, _request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        async def events() -> AsyncIterator[ProviderEvent]:
            yield StreamStarted("chatcmpl-test", 1_700_000_000, "fake-upstream")
            yield TextDelta("hello")
            yield TextDelta(" world")
            yield StreamFinished("stop")
            yield UsageReported(Usage(3, 2, 5))

        return events()


PROVIDER_KEY_VARIABLES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's .env and exported provider keys out of the tests.

    Settings reads a local .env plus the standard provider credential variables,
    so without this a machine configured for real upstreams quietly changes what
    the suite asserts.
    """
    for name in list(os.environ):
        if name.startswith("GATEWAY_") or name in PROVIDER_KEY_VARIABLES:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        provider_kind="openai",
        public_model="gateway-model",
        upstream_model="fake-success",
        upstream_base_url="http://fake.test/v1",
        api_key=None,
        upstream_api_key=None,
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    application = create_app(
        settings=settings,
        provider=SuccessfulProvider(),
        registry=CollectorRegistry(),
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway.test"
        ) as test_client:
            yield test_client


@pytest.fixture
def chat_payload() -> dict[str, object]:
    return {
        "model": "gateway-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
