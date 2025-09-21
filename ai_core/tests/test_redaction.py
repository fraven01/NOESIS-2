import pytest

from common.redaction import MASK, Redactor, hash_email, hash_str, hash_user_id


@pytest.fixture(autouse=True)
def _hash_settings(settings):
    settings.LOG_HASH_SALT = "unit-test-salt"
    settings.LOG_LLM_TEXT = True


def test_key_based_redaction_masks_entire_value():
    redactor = Redactor()
    event = {"email": "user@example.com", "token": "abc"}

    result = redactor(None, "info", event)

    assert result["email"] == MASK
    assert result["token"] == MASK


def test_inline_patterns_are_redacted():
    redactor = Redactor()
    event = {
        "message": "Contact user@example.com or call +491234567890",
        "iban": "DE44500105175407324931",
    }

    result = redactor(None, "info", event)

    assert result["message"] == f"Contact {MASK} or call {MASK}"
    assert result["iban"] == MASK


def test_credit_card_and_bearer_tokens():
    redactor = Redactor()
    event = {
        "message": "Card 4111 1111 1111 1111 used",
        "headers": {"Authorization": "Bearer secret-token"},
    }

    result = redactor(None, "info", event)

    assert result["message"] == f"Card {MASK} used"
    assert result["headers"]["Authorization"] == MASK


def test_service_account_and_jwt_markers():
    redactor = Redactor()
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    event = {
        "jwt": jwt,
        "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
        "payload": '{"type": "service_account"}',
    }

    result = redactor(None, "info", event)

    assert result["jwt"] == MASK
    assert result["private_key"] == MASK
    assert result["payload"] == '{"type": "[REDACTED]"}'


def test_prompt_response_masked_when_disabled(settings):
    settings.LOG_LLM_TEXT = False
    redactor = Redactor()
    event = {"prompt": "Tell me a story", "response": "Once upon a time"}

    result = redactor(None, "info", event)

    for key in ("prompt", "response"):
        assert result[key].startswith(f"{MASK} len=")
        assert "hash=" in result[key]


def test_hash_helpers_use_salt(settings):
    settings.LOG_HASH_SALT = "alpha"
    first = hash_str("value")
    second = hash_str("value")
    assert first == second

    settings.LOG_HASH_SALT = "beta"
    third = hash_str("value")
    assert third != first


def test_hash_email_and_user_id_helpers():
    settings_value = hash_email("User@Example.com ")
    assert settings_value == hash_email("user@example.com")
    assert hash_user_id(123) == hash_user_id("123")
