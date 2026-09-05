"""Controllable upstream for deterministic development and tests.

The behaviour is selected by the requested model name so a single server can
produce failures real providers cannot be asked for on demand. Both supported
wire protocols are served here, because a scenario is only useful if every
adapter can be driven through it.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="Inference Gateway Fake Upstream")

WORDS = ("A", " reliable", " gateway", " streams", " safely.")

STATUS_SCENARIOS = {
    "fake-http-400": 400,
    "fake-http-401": 401,
    "fake-http-429": 429,
    "fake-http-500": 500,
}


def _sse(payload: dict[str, Any], event: str | None = None) -> bytes:
    data = json.dumps(payload, separators=(",", ":"))
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n".encode()


async def _preamble(model: str) -> None:
    """Reproduce the timing and cancellation scenarios shared by both protocols."""
    if model == "fake-hang":
        await asyncio.Event().wait()
    if model == "fake-delay-first":
        await asyncio.sleep(1.0)
    if model == "fake-ignore-cancel":
        try:
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            # Some providers do not stop immediately. Keep this delay bounded so
            # cancellation tests cannot leak permanent background work.
            await asyncio.sleep(0.2)
            raise


def _chunk(
    completion_id: str,
    model: str,
    *,
    content: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, str] = {}
    if content is not None:
        delta["content"] = content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _read_model(request: Request, default: str) -> tuple[dict[str, Any], str]:
    body: dict[str, Any] = await request.json()
    return body, str(body.get("model", default))


def _streamed(events: Callable[[], AsyncIterator[bytes]]) -> StreamingResponse:
    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body, model = await _read_model(request, "fake-success")
    if not body.get("stream"):
        return JSONResponse({"error": {"message": "fake requires stream=true"}}, status_code=400)

    if model in STATUS_SCENARIOS:
        return JSONResponse(
            {"error": {"message": f"deliberate fake status {STATUS_SCENARIOS[model]}"}},
            status_code=STATUS_SCENARIOS[model],
        )

    async def events() -> AsyncIterator[bytes]:
        completion_id = f"chatcmpl-fake-{uuid.uuid4().hex}"
        await _preamble(model)
        if model == "fake-malformed":
            yield b"data: {not-json}\n\n"
            return

        for index, word in enumerate(WORDS):
            if model == "fake-delay-between" and index:
                await asyncio.sleep(1.0)
            yield _sse(_chunk(completion_id, model, content=word))
            if model == "fake-close-early" and index == 1:
                return

        yield _sse(_chunk(completion_id, model, finish_reason="stop"))
        if model != "fake-omit-usage":
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 5,
                        "total_tokens": 9,
                    },
                }
            )
        yield b"data: [DONE]\n\n"

    return _streamed(events)


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    body, model = await _read_model(request, "fake-success")
    if not body.get("stream"):
        return JSONResponse(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "stream only"}},
            status_code=400,
        )
    if "max_tokens" not in body:
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "max_tokens is required"},
            },
            status_code=400,
        )

    if model in STATUS_SCENARIOS:
        status = STATUS_SCENARIOS[model]
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": "api_error", "message": f"deliberate fake status {status}"},
            },
            status_code=status,
        )

    async def events() -> AsyncIterator[bytes]:
        message_id = f"msg_fake_{uuid.uuid4().hex}"
        await _preamble(model)
        if model == "fake-malformed":
            yield b"event: message_start\ndata: {not-json}\n\n"
            return

        yield _sse(
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "usage": {"input_tokens": 4, "output_tokens": 0},
                },
            },
            "message_start",
        )
        yield _sse(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            "content_block_start",
        )

        for index, word in enumerate(WORDS):
            if model == "fake-delay-between" and index:
                await asyncio.sleep(1.0)
            yield _sse(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": word},
                },
                "content_block_delta",
            )
            if model == "fake-close-early" and index == 1:
                return

        yield _sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
        usage = {} if model == "fake-omit-usage" else {"output_tokens": 5}
        yield _sse(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage},
            "message_delta",
        )
        yield _sse({"type": "message_stop"}, "message_stop")

    return _streamed(events)


@app.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}
