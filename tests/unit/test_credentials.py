"""Per-provider credential resolution."""

import pytest
from pydantic import SecretStr

from gateway.adapters.providers.registry import build_one, has_credential
from gateway.config import ModelSpec, Settings


def spec(kind: str, **overrides: object) -> ModelSpec:
    fields: dict[str, object] = {"name": kind, "kind": kind, "upstream_model": "m"}
    fields.update(overrides)
    return ModelSpec(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("anthropic", "anthropic_api_key"),
        ("openai", "openai_api_key"),
        ("gemini", "gemini_api_key"),
    ],
)
def test_each_kind_reads_its_own_credential(kind: str, field: str) -> None:
    settings = Settings(**{field: SecretStr("kind-key")})  # type: ignore[arg-type]

    assert settings.api_key_for(kind) == "kind-key"


def test_one_kinds_credential_never_leaks_to_another() -> None:
    settings = Settings(anthropic_api_key=SecretStr("anthropic-only"))

    assert settings.api_key_for("anthropic") == "anthropic-only"
    assert settings.api_key_for("openai") is None
    assert settings.api_key_for("gemini") is None


def test_the_shared_key_still_serves_kinds_without_their_own() -> None:
    settings = Settings(upstream_api_key=SecretStr("shared"))

    assert settings.api_key_for("openai") == "shared"
    assert settings.api_key_for("ollama") == "shared"


def test_a_kind_specific_key_wins_over_the_shared_one() -> None:
    settings = Settings(upstream_api_key=SecretStr("shared"), openai_api_key=SecretStr("specific"))

    assert settings.api_key_for("openai") == "specific"
    assert settings.api_key_for("anthropic") == "shared"


def test_a_blank_credential_counts_as_absent() -> None:
    settings = Settings(openai_api_key=SecretStr(""), upstream_api_key=SecretStr(""))

    assert settings.api_key_for("openai") is None
    assert not has_credential(settings, spec("openai"))


def test_a_per_model_key_overrides_every_environment_credential() -> None:
    settings = Settings(anthropic_api_key=SecretStr("environment"))

    provider = build_one(settings, spec("anthropic", api_key=SecretStr("per-model")))

    assert provider.api_key == "per-model"  # type: ignore[attr-defined]


def test_gemini_is_served_through_the_openai_compatible_endpoint() -> None:
    settings = Settings(gemini_api_key=SecretStr("gem"))

    provider = build_one(settings, spec("gemini", upstream_model="gemini-2.5-pro"))

    assert provider.chat_url == (  # type: ignore[attr-defined]
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert provider.api_key == "gem"  # type: ignore[attr-defined]
    assert provider.upstream_model == "gemini-2.5-pro"  # type: ignore[attr-defined]


def test_openai_sends_the_output_cap_under_its_current_name() -> None:
    """GPT-5 rejects ``max_tokens`` outright; the registry must not send it."""
    provider = build_one(Settings(openai_api_key=SecretStr("k")), spec("openai"))

    assert provider.max_tokens_field == "max_completion_tokens"  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["gemini", "ollama"])
def test_other_openai_compatible_upstreams_keep_the_original_name(kind: str) -> None:
    provider = build_one(Settings(upstream_api_key=SecretStr("k")), spec(kind))

    assert provider.max_tokens_field == "max_tokens"  # type: ignore[attr-defined]


def test_a_protocol_error_detail_never_reaches_the_client() -> None:
    """Details can quote raw upstream bytes, so they are for logs only."""
    from gateway.api.translation import error_payload
    from gateway.domain.errors import UpstreamProtocolError

    error = UpstreamProtocolError(detail='unsupported chunk: {"secret":"upstream"}')
    payload = error_payload(error)

    assert error.detail is not None
    assert "secret" not in str(payload)
    assert payload["error"]["code"] == "upstream_protocol_error"
