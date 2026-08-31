import json

import pytest

from gateway.api.translation import CompletionAccumulator, OpenAIStreamEncoder
from gateway.domain.errors import UpstreamProtocolError
from gateway.domain.events import StreamFinished, StreamStarted, TextDelta, UsageReported
from gateway.domain.models import Usage


def test_accumulator_builds_openai_completion() -> None:
    accumulator = CompletionAccumulator("public-model")
    accumulator.add(StreamStarted("chatcmpl-1", 123, "private-model"))
    accumulator.add(TextDelta("hello"))
    accumulator.add(TextDelta(" world"))
    accumulator.add(StreamFinished("stop"))
    accumulator.add(UsageReported(Usage(3, 2, 5)))

    response = accumulator.response()

    assert response["model"] == "public-model"
    assert response["choices"][0]["message"]["content"] == "hello world"
    assert response["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_accumulator_rejects_text_after_finish() -> None:
    accumulator = CompletionAccumulator("public-model")
    accumulator.add(StreamStarted("chatcmpl-1", 123, "private-model"))
    accumulator.add(StreamFinished("stop"))

    with pytest.raises(UpstreamProtocolError):
        accumulator.add(TextDelta("too late"))


def test_stream_encoder_emits_openai_chunks_and_done() -> None:
    encoder = OpenAIStreamEncoder("public-model")
    frames = [
        encoder.encode(StreamStarted("chatcmpl-1", 123, "private-model")),
        encoder.encode(TextDelta("hello")),
        encoder.encode(StreamFinished("stop")),
        encoder.done(),
    ]

    first = json.loads(frames[0].decode().removeprefix("data: "))
    second = json.loads(frames[1].decode().removeprefix("data: "))
    assert first["choices"][0]["delta"] == {"role": "assistant"}
    assert second["choices"][0]["delta"] == {"content": "hello"}
    assert frames[-1] == b"data: [DONE]\n\n"
