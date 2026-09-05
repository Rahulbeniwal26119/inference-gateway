import asyncio
import secrets
import time
from collections.abc import AsyncIterator
from pathlib import Path

import structlog
from fastapi import APIRouter, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.background import BackgroundTask

from gateway.adapters.providers.registry import has_credential
from gateway.api.schemas import ChatCompletionInput
from gateway.api.translation import (
    CompletionAccumulator,
    OpenAIStreamEncoder,
    error_payload,
    error_sse,
)
from gateway.application.chat import ChatService
from gateway.config import Settings
from gateway.domain.errors import (
    AuthenticationError,
    GatewayError,
    InvalidRequestError,
    UpstreamProtocolError,
)
from gateway.domain.events import ProviderEvent
from gateway.observability.metrics import GatewayMetrics, RequestObserver
from gateway.ports.provider import AsyncClosable

router = APIRouter()

CONSOLE_FILE = Path(__file__).with_name("console.html")
STARTED_AT = int(time.time())


async def close_stream(stream: AsyncIterator[ProviderEvent]) -> None:
    if isinstance(stream, AsyncClosable):
        await stream.aclose()


def authenticate(settings: Settings, authorization: str | None) -> None:
    configured = settings.api_key
    if configured is None or not configured.get_secret_value():
        return
    scheme, separator, credential = (authorization or "").partition(" ")
    expected = configured.get_secret_value()
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or not secrets.compare_digest(credential, expected)
    ):
        raise AuthenticationError


@router.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionInput,
    request: Request,
    authorization: str | None = Header(default=None),
) -> Response:
    settings: Settings = request.app.state.settings
    authenticate(settings, authorization)
    service: ChatService = request.app.state.chat_service
    metrics: GatewayMetrics = request.app.state.gateway_metrics
    # Resolve before admission so an unroutable model is still labelled with a
    # bounded provider value rather than inventing a new metrics series.
    routed = service.provider_for(payload.model)
    observer = metrics.begin(
        provider=routed.name if routed is not None else "unrouted",
        model=payload.model if routed is not None else "unrouted",
        streaming=payload.stream,
    )

    stream: AsyncIterator[ProviderEvent] | None = None
    try:
        stream = service.stream(payload.to_domain())
        first_event = await anext(stream)
        observer.observe(first_event)
    except StopAsyncIteration as exc:
        error = UpstreamProtocolError()
        observer.fail(error, "pre_response")
        if stream is not None:
            await close_stream(stream)
        raise error from exc
    except asyncio.CancelledError:
        observer.cancel()
        if stream is not None:
            await close_stream(stream)
        raise
    except GatewayError as error:
        observer.fail(error, "pre_response")
        if stream is not None:
            await close_stream(stream)
        raise

    if not payload.stream:
        return await _non_streaming_response(stream, first_event, observer, payload.model)

    return _streaming_response(stream, first_event, observer, payload.model)


async def _non_streaming_response(
    stream: AsyncIterator[ProviderEvent],
    first_event: ProviderEvent,
    observer: RequestObserver,
    public_model: str,
) -> JSONResponse:
    accumulator = CompletionAccumulator(public_model)
    try:
        accumulator.add(first_event)
        async for event in stream:
            observer.observe(event)
            accumulator.add(event)
        response = accumulator.response()
        observer.succeed()
        return JSONResponse(response)
    except asyncio.CancelledError:
        observer.cancel()
        raise
    except GatewayError as error:
        observer.fail(error, "pre_response")
        raise
    except Exception as exc:
        structlog.get_logger().exception("unexpected_inference_failure")
        unexpected_error = GatewayError()
        observer.fail(unexpected_error, "pre_response")
        raise unexpected_error from exc
    finally:
        await close_stream(stream)


def _streaming_response(
    stream: AsyncIterator[ProviderEvent],
    first_event: ProviderEvent,
    observer: RequestObserver,
    public_model: str,
) -> StreamingResponse:
    encoder = OpenAIStreamEncoder(public_model)
    try:
        first_frame = encoder.encode(first_event)
    except GatewayError as error:
        observer.fail(error, "pre_response")
        raise

    async def body() -> AsyncIterator[bytes]:
        try:
            yield first_frame
            async for event in stream:
                observer.observe(event)
                # Held events encode to nothing; do not emit an empty frame.
                if frame := encoder.encode(event):
                    yield frame
            yield encoder.done()
            observer.succeed()
        except asyncio.CancelledError:
            observer.cancel()
            raise
        except GatewayError as error:
            observer.fail(error, "mid_stream")
            yield error_sse(error)
        except Exception:
            structlog.get_logger().exception("unexpected_streaming_failure")
            unexpected_error = GatewayError()
            observer.fail(unexpected_error, "mid_stream")
            yield error_sse(unexpected_error)
        finally:
            await close_stream(stream)

    async def cleanup() -> None:
        if not observer.finished:
            observer.cancel()
        await close_stream(stream)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        background=BackgroundTask(cleanup),
    )


@router.get("/v1/models")
async def list_models(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    settings: Settings = request.app.state.settings
    authenticate(settings, authorization)
    service: ChatService = request.app.state.chat_service
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": STARTED_AT,
                    "owned_by": service.providers[name].name,
                }
                for name in service.model_names
            ],
        }
    )


def _dev_console_enabled(request: Request) -> bool:
    settings: Settings = request.app.state.settings
    return bool(settings.dev_console)


@router.get("/__dev/console", include_in_schema=False)
async def dev_console(request: Request) -> Response:
    """Serve the request console. Read per request so edits need no restart."""
    if not _dev_console_enabled(request):
        return Response(status_code=404)
    if not CONSOLE_FILE.is_file():
        return Response(status_code=404)
    return HTMLResponse(CONSOLE_FILE.read_text(encoding="utf-8"))


@router.get("/__dev/config", include_in_schema=False)
async def dev_config(request: Request) -> Response:
    """Timeouts and routing the console needs to interpret what it observes."""
    if not _dev_console_enabled(request):
        return Response(status_code=404)
    settings: Settings = request.app.state.settings
    service: ChatService = request.app.state.chat_service
    return JSONResponse(
        {
            "requires_api_key": bool(settings.api_key and settings.api_key.get_secret_value()),
            "default_max_tokens": settings.default_max_tokens,
            "timeouts": {
                "connect_s": settings.connect_timeout_s,
                "first_token_s": settings.first_token_timeout_s,
                "idle_s": settings.idle_timeout_s,
                "total_s": settings.total_timeout_s,
            },
            "models": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "upstream_model": spec.upstream_model,
                    "routable": spec.name in service.providers,
                    # Surfaced so a missing provider credential is visible in the
                    # console before a request fails with an upstream 401.
                    "has_key": has_credential(settings, spec),
                }
                for spec in settings.routable_models()
            ],
        }
    )


@router.get("/health/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", include_in_schema=False)
async def readiness(request: Request) -> JSONResponse:
    ready = hasattr(request.app.state, "chat_service")
    return JSONResponse(
        {"status": "ready" if ready else "not_ready"}, status_code=200 if ready else 503
    )


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    gateway_metrics: GatewayMetrics = request.app.state.gateway_metrics
    return Response(generate_latest(gateway_metrics.registry), media_type=CONTENT_TYPE_LATEST)


async def gateway_error_handler(_request: Request, error: GatewayError) -> JSONResponse:
    return JSONResponse(error_payload(error), status_code=error.status_code)


async def validation_error_handler(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    first = error.errors()[0] if error.errors() else {}
    location = first.get("loc", ())
    param = ".".join(str(part) for part in location if part != "body") or None
    if first.get("type") == "extra_forbidden" and param:
        message = f"Unsupported request field: {param}."
    else:
        detail = str(first.get("msg", "The request body is invalid."))
        message = f"Invalid value for {param}: {detail}" if param else detail
    invalid = InvalidRequestError(message)
    response = error_payload(invalid)
    response["error"]["param"] = param
    return JSONResponse(response, status_code=400)
