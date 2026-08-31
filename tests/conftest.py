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


class SuccessfulProvider:
    name = "fake"

    def stream(self, _request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        async def events() -> AsyncIterator[ProviderEvent]:
            yield StreamStarted("chatcmpl-test", 1_700_000_000, "fake-upstream")
            yield TextDelta("hello")
            yield TextDelta(" world")
            yield StreamFinished("stop")
            yield UsageReported(Usage(3, 2, 5))

        return events()


@pytest.fixture
def settings() -> Settings:
    return Settings(
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
