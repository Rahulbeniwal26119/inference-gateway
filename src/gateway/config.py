from functools import lru_cache

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Each provider kind reads the credential name its own ecosystem already uses,
# so a machine already set up for these SDKs needs no gateway-specific variables.
PROVIDER_KEY_FIELDS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "gemini_api_key",
}


def _clean_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().rstrip("/")
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        raise ValueError("must start with http:// or https://")
    return value


class ModelSpec(BaseModel):
    """One publicly routable model and the upstream it resolves to.

    A gateway that serves a single model cannot be used to compare models, which
    is the whole point of a gateway. Each entry names a provider kind from the
    registry, so adding a model is configuration rather than code.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    """The model id clients request and see echoed back."""

    kind: str = Field(min_length=1)
    """Provider registry key: anthropic, openai, ollama."""

    upstream_model: str = Field(min_length=1)
    """The model id sent to the upstream provider."""

    base_url: str | None = None
    """Overrides the registry's default base URL for this kind."""

    api_key: SecretStr | None = None
    """Falls back to GATEWAY_UPSTREAM_API_KEY when unset."""

    provider_name: str | None = None
    """Metrics label for this upstream. Defaults to ``kind``."""

    @field_validator("name", "upstream_model", "provider_name")
    @classmethod
    def non_empty_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("kind")
    @classmethod
    def normalize_kind(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        return _clean_base_url(value)

    @property
    def label(self) -> str:
        return self.provider_name or self.kind


class Settings(BaseSettings):
    """Runtime configuration loaded from GATEWAY_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models: list[ModelSpec] = Field(default_factory=list)
    """Routable models as JSON. When empty, the single-model fields below apply."""

    provider_kind: str = "ollama"
    provider_name: str = "upstream"
    public_model: str = "gateway-model"
    upstream_model: str = "llama3.2"
    upstream_base_url: str | None = None
    upstream_api_key: SecretStr | None = None
    api_key: SecretStr | None = None

    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "GATEWAY_ANTHROPIC_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "GATEWAY_OPENAI_API_KEY"),
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GATEWAY_GEMINI_API_KEY"),
    )

    dev_console: bool = False
    """Serves the request console at /__dev/console. Never enable in production."""

    default_max_tokens: int = Field(default=4096, ge=1)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    first_token_timeout_s: float = Field(default=30.0, gt=0)
    idle_timeout_s: float = Field(default=20.0, gt=0)
    total_timeout_s: float = Field(default=300.0, gt=0)
    max_connections: int = Field(default=100, ge=1)
    max_keepalive_connections: int = Field(default=20, ge=0)
    log_level: str = "INFO"

    @field_validator("provider_name", "public_model", "upstream_model")
    @classmethod
    def non_empty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("provider_kind")
    @classmethod
    def normalize_kind(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("upstream_base_url")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        return _clean_base_url(value)

    @model_validator(mode="after")
    def unique_model_names(self) -> "Settings":
        seen: set[str] = set()
        for spec in self.models:
            if spec.name in seen:
                raise ValueError(f"duplicate model name {spec.name!r}")
            seen.add(spec.name)
        return self

    def api_key_for(self, kind: str) -> str | None:
        """The upstream credential for a provider kind.

        The kind's own variable wins, then the shared one, so a single-provider
        deployment can keep using GATEWAY_UPSTREAM_API_KEY unchanged. An empty
        value counts as absent: an exported-but-blank variable is not a key.
        """
        field = PROVIDER_KEY_FIELDS.get(kind)
        candidates = (getattr(self, field) if field else None, self.upstream_api_key)
        for candidate in candidates:
            if candidate is not None and candidate.get_secret_value():
                return candidate.get_secret_value()
        return None

    def legacy_model(self) -> ModelSpec:
        """The model described by the single-model environment variables."""
        return ModelSpec(
            name=self.public_model,
            kind=self.provider_kind,
            upstream_model=self.upstream_model,
            base_url=self.upstream_base_url,
            api_key=self.upstream_api_key,
            provider_name=self.provider_name,
        )

    def routable_models(self) -> tuple[ModelSpec, ...]:
        """The models this gateway serves, oldest single-model config included.

        The single-model environment variables stay authoritative when no
        ``GATEWAY_MODELS`` list is given, so existing deployments keep working
        without being rewritten as JSON.
        """
        return tuple(self.models) if self.models else (self.legacy_model(),)


@lru_cache
def get_settings() -> Settings:
    return Settings()
