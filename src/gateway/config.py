from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from GATEWAY_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider_name: str = "upstream"
    public_model: str = "gateway-model"
    upstream_model: str = "llama3.2"
    upstream_base_url: str = "http://127.0.0.1:11434/v1"
    upstream_api_key: SecretStr | None = None
    api_key: SecretStr | None = None

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

    @field_validator("upstream_base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value

    @property
    def upstream_chat_url(self) -> str:
        return f"{self.upstream_base_url}/chat/completions"


@lru_cache
def get_settings() -> Settings:
    return Settings()
