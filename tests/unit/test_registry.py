import pytest
from pydantic import SecretStr

from gateway.adapters.providers.anthropic import AnthropicProvider
from gateway.adapters.providers.openai_compatible import OpenAICompatibleProvider
from gateway.adapters.providers.registry import (
    PROVIDERS,
    UnknownProviderError,
    build_provider,
    provider_kinds,
)
from gateway.config import Settings


def settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {"upstream_model": "some-model"}
    fields.update(overrides)
    return Settings(**fields)


def test_registry_exposes_every_provider_including_a_local_one() -> None:
    assert provider_kinds() == ("anthropic", "gemini", "ollama", "openai")
    assert PROVIDERS["ollama"].default_base_url.startswith("http://127.0.0.1")


@pytest.mark.parametrize(
    ("kind", "expected_type", "url_attribute", "expected_url"),
    [
        (
            "openai",
            OpenAICompatibleProvider,
            "chat_url",
            "https://api.openai.com/v1/chat/completions",
        ),
        (
            "ollama",
            OpenAICompatibleProvider,
            "chat_url",
            "http://127.0.0.1:11434/v1/chat/completions",
        ),
        ("anthropic", AnthropicProvider, "messages_url", "https://api.anthropic.com/v1/messages"),
        (
            "gemini",
            OpenAICompatibleProvider,
            "chat_url",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
    ],
)
def test_each_kind_builds_its_adapter_against_the_default_base_url(
    kind: str, expected_type: type, url_attribute: str, expected_url: str
) -> None:
    provider = build_provider(settings(provider_kind=kind))

    assert isinstance(provider, expected_type)
    assert getattr(provider, url_attribute) == expected_url


def test_explicit_base_url_overrides_the_registry_default() -> None:
    provider = build_provider(
        settings(provider_kind="openai", upstream_base_url="http://127.0.0.1:9000/v1/")
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.chat_url == "http://127.0.0.1:9000/v1/chat/completions"


def test_unknown_kind_names_the_supported_kinds() -> None:
    with pytest.raises(UnknownProviderError) as caught:
        build_provider(settings(provider_kind="bedrock"))

    assert "anthropic, gemini, ollama, openai" in str(caught.value)


def test_blank_upstream_key_is_treated_as_absent() -> None:
    provider = build_provider(settings(provider_kind="openai", upstream_api_key=SecretStr("")))

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.api_key is None


def test_timeouts_reach_the_adapter() -> None:
    provider = build_provider(
        settings(provider_kind="anthropic", first_token_timeout_s=7, idle_timeout_s=3)
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.timeouts.first_token == 7
    assert provider.timeouts.idle == 3
