from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from gateway.domain.events import ProviderEvent
from gateway.domain.models import ChatRequest


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Upstream constraints the application must not assume away.

    A provider that cannot honour part of the public schema declares it here so
    the gateway can reject or adapt the request explicitly, rather than sending
    something the upstream will reinterpret.
    """

    requires_max_tokens: bool = False
    """The upstream rejects requests that omit ``max_tokens``."""

    supports_leading_assistant_message: bool = True
    """The upstream accepts a conversation whose first non-system turn is not the user's."""


class Provider(Protocol):
    name: str
    capabilities: Capabilities

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]: ...


@runtime_checkable
class AsyncClosable(Protocol):
    """Anything owning an upstream resource that must be released explicitly."""

    async def aclose(self) -> None: ...
