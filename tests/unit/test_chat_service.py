from collections.abc import AsyncIterator

import pytest

from gateway.application.chat import ChatService
from gateway.domain.errors import InvalidRequestError, ModelNotFoundError
from gateway.domain.events import ProviderEvent, StreamFinished
from gateway.domain.models import ChatRequest, Message
from gateway.ports.provider import Capabilities


class RecordingProvider:
    name = "recording"

    def __init__(self, capabilities: Capabilities) -> None:
        self.capabilities = capabilities
        self.seen: ChatRequest | None = None

    def stream(self, request: ChatRequest) -> AsyncIterator[ProviderEvent]:
        self.seen = request

        async def events() -> AsyncIterator[ProviderEvent]:
            yield StreamFinished("stop")

        return events()


def service(capabilities: Capabilities) -> tuple[ChatService, RecordingProvider]:
    provider = RecordingProvider(capabilities)
    return ChatService({"gateway-model": provider}, default_max_tokens=1234), provider


def chat(*messages: Message, max_tokens: int | None = None) -> ChatRequest:
    return ChatRequest("gateway-model", messages, max_tokens=max_tokens)


def test_unknown_public_model_is_rejected() -> None:
    chat_service, _ = service(Capabilities())

    with pytest.raises(ModelNotFoundError):
        chat_service.stream(ChatRequest("other", (Message("user", "hi"),)))


def test_default_max_tokens_is_applied_only_when_the_provider_requires_it() -> None:
    requiring, provider = service(Capabilities(requires_max_tokens=True))
    requiring.stream(chat(Message("user", "hi")))
    assert provider.seen is not None
    assert provider.seen.max_tokens == 1234

    relaxed, other = service(Capabilities())
    relaxed.stream(chat(Message("user", "hi")))
    assert other.seen is not None
    assert other.seen.max_tokens is None


def test_a_caller_supplied_max_tokens_is_never_overwritten() -> None:
    chat_service, provider = service(Capabilities(requires_max_tokens=True))

    chat_service.stream(chat(Message("user", "hi"), max_tokens=16))

    assert provider.seen is not None
    assert provider.seen.max_tokens == 16


def test_leading_assistant_message_is_rejected_for_providers_that_forbid_it() -> None:
    chat_service, _ = service(Capabilities(supports_leading_assistant_message=False))

    with pytest.raises(InvalidRequestError) as caught:
        chat_service.stream(chat(Message("assistant", "I spoke first"), Message("user", "hi")))

    assert "'user' role" in caught.value.message


def test_system_messages_do_not_count_as_the_leading_turn() -> None:
    chat_service, provider = service(Capabilities(supports_leading_assistant_message=False))

    chat_service.stream(chat(Message("system", "Be brief."), Message("user", "hi")))

    assert provider.seen is not None


def test_a_system_only_conversation_is_rejected() -> None:
    chat_service, _ = service(Capabilities(supports_leading_assistant_message=False))

    with pytest.raises(InvalidRequestError):
        chat_service.stream(chat(Message("system", "Be brief.")))


def test_providers_without_the_restriction_accept_a_leading_assistant_message() -> None:
    chat_service, provider = service(Capabilities())

    chat_service.stream(chat(Message("assistant", "I spoke first")))

    assert provider.seen is not None
