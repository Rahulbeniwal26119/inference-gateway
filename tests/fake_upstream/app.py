import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="Inference Gateway Fake Upstream")


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    model = str(body.get("model", "fake-success"))
    if not body.get("stream"):
        return JSONResponse({"error": {"message": "fake requires stream=true"}}, status_code=400)

    status_scenarios = {
        "fake-http-400": 400,
        "fake-http-401": 401,
        "fake-http-429": 429,
        "fake-http-500": 500,
    }
    if model in status_scenarios:
        return JSONResponse(
            {"error": {"message": f"deliberate fake status {status_scenarios[model]}"}},
            status_code=status_scenarios[model],
        )

    async def events() -> AsyncIterator[bytes]:
        completion_id = f"chatcmpl-fake-{uuid.uuid4().hex}"
        if model == "fake-hang":
            await asyncio.Event().wait()
            return
        if model == "fake-delay-first":
            await asyncio.sleep(1.0)
        if model == "fake-ignore-cancel":
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                # Some providers do not stop immediately. Keep this delay bounded so
                # cancellation tests cannot leak permanent background work.
                await asyncio.sleep(0.2)
                return
        if model == "fake-malformed":
            yield b"data: {not-json}\n\n"
            return

        words = ("A", " reliable", " gateway", " streams", " safely.")
        for index, word in enumerate(words):
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

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "ok"}
