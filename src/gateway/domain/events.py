from dataclasses import dataclass

from gateway.domain.models import Usage


@dataclass(frozen=True, slots=True)
class StreamStarted:
    completion_id: str
    created: int
    upstream_model: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class UsageReported:
    usage: Usage


@dataclass(frozen=True, slots=True)
class StreamFinished:
    finish_reason: str


ProviderEvent = StreamStarted | TextDelta | UsageReported | StreamFinished
