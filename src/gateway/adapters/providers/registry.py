"""Provider registry.

Adding a provider is one adapter module plus one entry in ``PROVIDERS``. The
registry owns the per-provider default base URL so configuration only has to
name a kind; nothing else in the gateway branches on which upstream is active.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from gateway.adapters.providers.anthropic import AnthropicProvider
from gateway.adapters.providers.openai_compatible import OpenAICompatibleProvider
from gateway.adapters.providers.streaming import ProviderTimeouts
from gateway.config import ModelSpec, Settings
from gateway.ports.provider import Provider


class UnknownProviderError(ValueError):
    """The configured provider kind has no registry entry."""


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    default_base_url: str
    build: Callable[[Settings, ModelSpec, str], Provider]


def _timeouts(settings: Settings) -> ProviderTimeouts:
    return ProviderTimeouts(
        first_token=settings.first_token_timeout_s,
        idle=settings.idle_timeout_s,
        total=settings.total_timeout_s,
    )


def _api_key(settings: Settings, spec: ModelSpec) -> str | None:
    """Per-model key first, then the credential for the provider kind."""
    if spec.api_key is not None and spec.api_key.get_secret_value():
        return spec.api_key.get_secret_value()
    return settings.api_key_for(spec.kind)


def _build_openai_compatible(
    settings: Settings,
    spec: ModelSpec,
    base_url: str,
    *,
    max_tokens_field: str = "max_tokens",
) -> Provider:
    return OpenAICompatibleProvider(
        name=spec.label,
        chat_url=f"{base_url}/chat/completions",
        upstream_model=spec.upstream_model,
        api_key=_api_key(settings, spec),
        connect_timeout=settings.connect_timeout_s,
        timeouts=_timeouts(settings),
        max_tokens_field=max_tokens_field,
        max_connections=settings.max_connections,
        max_keepalive_connections=settings.max_keepalive_connections,
    )


def _build_anthropic(settings: Settings, spec: ModelSpec, base_url: str) -> Provider:
    return AnthropicProvider(
        name=spec.label,
        messages_url=f"{base_url}/messages",
        upstream_model=spec.upstream_model,
        api_key=_api_key(settings, spec),
        connect_timeout=settings.connect_timeout_s,
        timeouts=_timeouts(settings),
        max_connections=settings.max_connections,
        max_keepalive_connections=settings.max_keepalive_connections,
    )


# Gemini is served through its OpenAI-compatible endpoint rather than a third
# adapter: it speaks the same SSE chunk shape and the same Bearer auth, so a
# dedicated adapter would be the OpenAI one with a different base URL.
PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec("https://api.anthropic.com/v1", _build_anthropic),
    "gemini": ProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta/openai", _build_openai_compatible
    ),
    "ollama": ProviderSpec("http://127.0.0.1:11434/v1", _build_openai_compatible),
    # OpenAI renamed the output cap: its reasoning models reject "max_tokens"
    # with a 400. "max_completion_tokens" is accepted across the current range,
    # so it is the right name for this kind and only this kind.
    "openai": ProviderSpec(
        "https://api.openai.com/v1",
        partial(_build_openai_compatible, max_tokens_field="max_completion_tokens"),
    ),
}


def provider_kinds() -> tuple[str, ...]:
    return tuple(sorted(PROVIDERS))


def build_one(settings: Settings, spec: ModelSpec) -> Provider:
    provider_spec = PROVIDERS.get(spec.kind)
    if provider_spec is None:
        raise UnknownProviderError(
            f"Unknown provider kind {spec.kind!r} for model {spec.name!r}. "
            f"Must be one of: {', '.join(provider_kinds())}."
        )
    return provider_spec.build(settings, spec, spec.base_url or provider_spec.default_base_url)


def build_provider(settings: Settings) -> Provider:
    """Build the provider named by the single-model environment variables.

    Deliberately independent of ``GATEWAY_MODELS``: reading the first routable
    model here would silently ignore ``provider_kind`` whenever a model list is
    configured, which is a difference no caller of this function expects.
    """
    return build_one(settings, settings.legacy_model())


def has_credential(settings: Settings, spec: ModelSpec) -> bool:
    """Whether a key is available for this model. Local upstreams need none."""
    return _api_key(settings, spec) is not None


def build_providers(settings: Settings) -> dict[str, Provider]:
    """Build one provider per routable model, keyed by the public model name.

    Each model gets its own adapter instance, and therefore its own connection
    pool: a saturated upstream cannot starve a model served from elsewhere.
    """
    return {spec.name: build_one(settings, spec) for spec in settings.routable_models()}
