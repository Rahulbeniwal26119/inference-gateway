from collections.abc import AsyncIterator

from gateway.domain.errors import ModelNotFoundError
from gateway.domain.events import ProviderEvent
from gateway.domain.models import ChatRequest
from gateway.ports.provider import Provider


class ChatService:
    def __init__(self, provider: Provider, public_model: str) -> None:
        self.provider = provider
        self.public_model = public_model

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        if request.model != self.public_model:
            raise ModelNotFoundError(f"The model '{request.model}' does not exist.")
        return self.provider.stream(request)
