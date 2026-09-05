"""Shared upstream streaming mechanics: timeout classes and SSE framing.

Both provider adapters own an HTTP streaming context and translate a
provider-specific event stream into domain events. The timeout classes and the
Server-Sent Events framing are identical across upstreams, so they live here
rather than drifting between adapters.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass

from gateway.domain.errors import (
    UpstreamFirstTokenTimeoutError,
    UpstreamIdleTimeoutError,
    UpstreamProtocolError,
    UpstreamTotalTimeoutError,
)
from gateway.domain.events import ProviderEvent


@dataclass(frozen=True, slots=True)
class ProviderTimeouts:
    first_token: float
    idle: float
    total: float


@dataclass(frozen=True, slots=True)
class SSEFrame:
    event: str | None
    data: str


MAX_UPSTREAM_DETAIL = 300


def upstream_error_detail(body: bytes) -> str | None:
    """The upstream's own error message, when it sent one we can trust to relay.

    Both supported protocols wrap it as ``{"error": {"message": ...}}``. Only
    the message is taken, and only a bounded prefix of it: the surrounding body
    can carry request ids and infrastructure detail that is not the caller's.
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str) or not message.strip():
        return None
    message = " ".join(message.split())
    if len(message) > MAX_UPSTREAM_DETAIL:
        message = message[:MAX_UPSTREAM_DETAIL].rstrip() + "…"
    return message


async def iter_sse_frames(lines: AsyncIterator[str]) -> AsyncGenerator[SSEFrame, None]:
    """Group SSE lines into dispatched frames.

    A frame is dispatched on a blank line. An upstream that stops mid-frame is a
    truncated protocol, not an empty stream, so trailing data raises.
    """
    event: str | None = None
    data_lines: list[str] = []

    async for line in lines:
        if line == "":
            if data_lines:
                yield SSEFrame(event, "\n".join(data_lines))
                data_lines = []
            event = None
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))

    if data_lines:
        raise UpstreamProtocolError


async def stream_with_timeouts(
    raw_events: AsyncGenerator[ProviderEvent, None],
    timeouts: ProviderTimeouts,
) -> AsyncIterator[ProviderEvent]:
    """Apply the first-token, idle, and total limits to a provider event stream.

    The connect timeout belongs to the HTTP client. Closing or cancelling this
    generator closes ``raw_events``, which owns the upstream response.
    """
    started_at = time.monotonic()
    first_event = True

    try:
        while True:
            total_remaining = timeouts.total - (time.monotonic() - started_at)
            if total_remaining <= 0:
                raise UpstreamTotalTimeoutError

            phase_timeout = timeouts.first_token if first_event else timeouts.idle
            total_is_limiting = total_remaining <= phase_timeout
            timeout = min(total_remaining, phase_timeout)

            try:
                event = await asyncio.wait_for(anext(raw_events), timeout=timeout)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                if total_is_limiting:
                    raise UpstreamTotalTimeoutError from exc
                if first_event:
                    raise UpstreamFirstTokenTimeoutError from exc
                raise UpstreamIdleTimeoutError from exc

            first_event = False
            yield event
    finally:
        await raw_events.aclose()
