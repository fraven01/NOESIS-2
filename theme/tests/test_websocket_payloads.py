import pytest
from pydantic import ValidationError

from theme.websocket_payloads import RagChatPayload


def test_payload_requires_message_and_tenant() -> None:
    with pytest.raises(ValidationError):
        RagChatPayload.model_validate({"message": "", "tenant_id": "tenant"})
    with pytest.raises(ValidationError):
        RagChatPayload.model_validate({"message": "hello", "tenant_id": " "})


def test_payload_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RagChatPayload.model_validate(
            {"message": "hello", "tenant_id": "tenant", "user_pk": "nope"}
        )


def test_payload_allows_hybrid_tuning_fields() -> None:
    payload = RagChatPayload.model_validate(
        {
            "message": "hello",
            "tenant_id": "tenant",
            "alpha": 0.6,
            "top_k": 3,
            "vec_limit": 5,
        }
    )

    assert payload.alpha == 0.6
    assert payload.top_k == 3
    assert payload.vec_limit == 5
