from collections.abc import AsyncIterator, Mapping
from dataclasses import replace

from gateway.domain.errors import InvalidRequestError, ModelNotFoundError
from gateway.domain.events import ProviderEvent
from gateway.domain.models import ChatRequest
from gateway.ports.provider import Provider


class ChatService:
    """Route a request to the provider serving its model and reconcile the two.

    Provider capabilities are applied here rather than inside an adapter so that
    a rejection is a gateway decision with an honest HTTP status, made before any
    upstream connection is opened. Capabilities are per provider, so they can
    only be applied once the model has chosen one.
    """

    def __init__(self, providers: Mapping[str, Provider], default_max_tokens: int) -> None:
        self.providers = dict(providers)
        self.default_max_tokens = default_max_tokens

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.providers))

    def provider_for(self, model: str) -> Provider | None:
        """The provider serving ``model``, or None. Never raises, so callers can
        label metrics for an unroutable model before rejecting it."""
        return self.providers.get(model)

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        provider = self.providers.get(request.model)
        if provider is None:
            raise ModelNotFoundError(f"The model '{request.model}' does not exist.")
        return provider.stream(self._reconcile(request, provider))

    def _reconcile(self, request: ChatRequest, provider: Provider) -> ChatRequest:
        capabilities = provider.capabilities

        if not capabilities.supports_leading_assistant_message:
            conversation = tuple(
                message for message in request.messages if message.role != "system"
            )
            if not conversation:
                raise InvalidRequestError(
                    "This provider requires at least one user or assistant message "
                    "in addition to any system messages."
                )
            if conversation[0].role != "user":
                raise InvalidRequestError(
                    "This provider requires the first non-system message to have the 'user' role."
                )

        if capabilities.requires_max_tokens and request.max_tokens is None:
            return replace(request, max_tokens=self.default_max_tokens)
        return request
