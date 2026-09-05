from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from prometheus_client import CollectorRegistry

from gateway.adapters.providers.registry import build_providers
from gateway.api.middleware import RequestIdMiddleware
from gateway.api.routes import gateway_error_handler, router, validation_error_handler
from gateway.application.chat import ChatService
from gateway.config import Settings, get_settings
from gateway.domain.errors import GatewayError
from gateway.observability.logging import configure_logging
from gateway.observability.metrics import GatewayMetrics
from gateway.ports.provider import AsyncClosable, Provider


def create_app(
    *,
    settings: Settings | None = None,
    provider: Provider | None = None,
    providers: dict[str, Provider] | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    injected = providers
    if injected is None and provider is not None:
        injected = {resolved_settings.public_model: provider}

    def chat_service(active: dict[str, Provider]) -> ChatService:
        return ChatService(active, resolved_settings.default_max_tokens)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active = injected if injected is not None else build_providers(resolved_settings)
        application.state.chat_service = chat_service(active)
        try:
            yield
        finally:
            if injected is None:
                for built in active.values():
                    if isinstance(built, AsyncClosable):
                        await built.aclose()

    application = FastAPI(
        title="Inference Gateway",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.gateway_metrics = GatewayMetrics(registry)
    if injected is not None:
        application.state.chat_service = chat_service(injected)
    application.add_middleware(RequestIdMiddleware)
    application.include_router(router)
    application.add_exception_handler(GatewayError, gateway_error_handler)  # type: ignore[arg-type]
    application.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    return application


app = create_app()


def run() -> None:
    uvicorn.run("gateway.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
