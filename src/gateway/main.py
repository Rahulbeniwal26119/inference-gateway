from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from prometheus_client import CollectorRegistry

from gateway.adapters.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderTimeouts,
)
from gateway.api.middleware import RequestIdMiddleware
from gateway.api.routes import gateway_error_handler, router, validation_error_handler
from gateway.application.chat import ChatService
from gateway.config import Settings, get_settings
from gateway.domain.errors import GatewayError
from gateway.observability.logging import configure_logging
from gateway.observability.metrics import GatewayMetrics
from gateway.ports.provider import Provider


def create_app(
    *,
    settings: Settings | None = None,
    provider: Provider | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active_provider = provider or _build_provider(resolved_settings)
        application.state.chat_service = ChatService(
            active_provider, resolved_settings.public_model
        )
        try:
            yield
        finally:
            if provider is None and isinstance(active_provider, OpenAICompatibleProvider):
                await active_provider.aclose()

    application = FastAPI(
        title="Inference Gateway",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.gateway_metrics = GatewayMetrics(registry)
    if provider is not None:
        application.state.chat_service = ChatService(provider, resolved_settings.public_model)
    application.add_middleware(RequestIdMiddleware)
    application.include_router(router)
    application.add_exception_handler(GatewayError, gateway_error_handler)  # type: ignore[arg-type]
    application.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    return application


def _build_provider(settings: Settings) -> OpenAICompatibleProvider:
    api_key = settings.upstream_api_key
    return OpenAICompatibleProvider(
        name=settings.provider_name,
        chat_url=settings.upstream_chat_url,
        upstream_model=settings.upstream_model,
        api_key=api_key.get_secret_value() if api_key and api_key.get_secret_value() else None,
        connect_timeout=settings.connect_timeout_s,
        timeouts=ProviderTimeouts(
            first_token=settings.first_token_timeout_s,
            idle=settings.idle_timeout_s,
            total=settings.total_timeout_s,
        ),
        max_connections=settings.max_connections,
        max_keepalive_connections=settings.max_keepalive_connections,
    )


app = create_app()


def run() -> None:
    uvicorn.run("gateway.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
