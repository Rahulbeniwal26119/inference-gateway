from collections.abc import AsyncIterator
from typing import Protocol

from gateway.domain.events import ProviderEvent
from gateway.domain.models import ChatRequest


class Provider(Protocol):
    name: str

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]: ...
