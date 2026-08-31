import pytest
from pydantic import ValidationError

from gateway.api.schemas import ChatCompletionInput


def test_supported_fields_translate_to_domain() -> None:
    request = ChatCompletionInput.model_validate(
        {
            "model": "gateway-model",
            "messages": [{"role": "system", "content": "Be concise."}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 20,
            "stop": ["one", "two"],
        }
    )

    domain = request.to_domain()

    assert domain.stop == ("one", "two")
    assert domain.messages[0].role == "system"


def test_unsupported_field_is_rejected() -> None:
    with pytest.raises(ValidationError) as caught:
        ChatCompletionInput.model_validate(
            {
                "model": "gateway-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [],
            }
        )

    assert caught.value.errors()[0]["type"] == "extra_forbidden"
