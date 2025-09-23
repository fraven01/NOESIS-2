import json
from urllib.parse import parse_qsl, urlparse

import pytest

from ai_core.infra.mask_prompt import (
    mask_prompt as mask_prompt_with_config,
    mask_response,
)
from ai_core.infra.pii import mask_structured, mask_text
from ai_core.metrics.pii_metrics import PII_DETECTIONS


@pytest.mark.parametrize(
    "text",
    [
        "+49 170 1234567",
        "Call me at +1 (555) 123-4567",
    ],
)
def test_phone_with_plus_is_masked(text):
    masked = mask_text(text, "balanced", False, None)
    assert masked != text
    assert "+" not in masked


def test_email_is_masked():
    masked = mask_text("Reach me at user@example.com", "balanced", False, None)
    assert "example.com" not in masked
    assert "[REDACTED_EMAIL]" in masked


def test_jwt_structure_preserved():
    token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6Ikpv"
        "aG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    masked = mask_structured(token, "balanced", False, None)
    assert masked.count(".") == 2
    assert masked != token
    assert masked.startswith("[REDACTED_JWT_HEADER]")


def test_inline_jwt_is_masked():
    token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1MTYyMzkwMjJ9."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    text = f"Authorization: Bearer {token}"
    masked = mask_text(text, "balanced", False, None)
    assert masked != text
    assert "Bearer" in masked
    assert "[REDACTED_JWT_HEADER]" in masked or "<JWT_HEADER_" in masked


def test_service_account_json_values_masked():
    data = {
        "type": "service_account",
        "client_email": "service@example.com",
        "private_key_id": "abcdef123456",
        "private_key": "-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----\n",
        "client_id": "1234567890",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    masked_str = mask_structured(json.dumps(data), "balanced", False, None)
    masked = json.loads(masked_str)
    for key in {"type", "client_email", "private_key_id", "private_key", "client_id"}:
        assert masked[key].startswith("[REDACTED_" + key.upper() + "]")
    assert masked["token_uri"] == data["token_uri"]
    assert '""' not in masked_str


def test_url_query_masks_sensitive_parameters():
    url = "https://example.com/callback?code=abc123&state=xyz&email=user@example.com&next=/home"
    masked = mask_structured(url, "balanced", False, None)
    assert "state=xyz" in masked
    assert "next=/home" in masked
    assert "code=abc123" not in masked
    assert "email=user@example.com" not in masked


def test_inline_url_query_masks_sensitive_parameters():
    text = (
        "Visit https://example.com/callback?code=abc123&state=xyz&email=user@example.com "
        "for details"
    )
    masked = mask_text(text, "balanced", False, None)
    assert "state=xyz" in masked
    assert "code=abc123" not in masked
    assert "email=user@example.com" not in masked
    assert masked.startswith("Visit https://example.com/callback?")


def test_low_entropy_key_parameter_not_masked():
    url = "https://host/path?key=monkey&ok=1"
    masked = mask_structured(url, "balanced", False, None)
    assert "key=monkey" in masked
    assert "ok=1" in masked


def test_high_entropy_key_parameter_masked():
    secret = "sk_live_ABCDEFGHijklmnopQRST"
    url = f"https://host/path?key={secret}&next=/home"
    masked = mask_structured(url, "balanced", False, None)
    parsed = urlparse(masked)
    params = dict(parse_qsl(parsed.query))
    assert "key" in params
    assert params["key"] != secret
    assert params["key"].startswith("[REDACTED_KEY]") or params["key"].startswith(
        "<KEY_"
    )
    assert params.get("next") == "/home"


def test_iban_is_masked():
    iban = "DE89370400440532013000"
    masked = mask_text(iban, "balanced", False, None)
    assert masked != iban
    assert masked == "[REDACTED_IBAN]"


def test_numbers_not_masked_in_industrial_mode():
    text = "Report for 2024 shows 1500 units sold"
    masked = mask_text(text, "balanced", False, None)
    assert "2024" in masked
    assert "1500" in masked


def test_idempotent_masking():
    text = "Contact user@example.com or call +1-555-1234"
    once = mask_text(text, "balanced", False, None)
    twice = mask_text(once, "balanced", False, None)
    assert once == twice


def test_deterministic_masking_uses_hmac():
    text = "user@example.com"
    masked_one = mask_text(text, "balanced", True, b"secret")
    masked_two = mask_text(text, "balanced", True, b"secret")
    assert masked_one == masked_two
    assert masked_one.startswith("<EMAIL_")


@pytest.mark.gold
def test_gold_policy_lenient_does_not_mask_iban():
    iban = "DE89370400440532013000"
    masked = mask_text(iban, "balanced", False, None, mode="gold")
    assert masked == iban


@pytest.mark.gold
def test_gold_lenient_does_not_mask_inline_jwt():
    token = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1MTYyMzkwMjJ9."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    text = f"Bearer {token}"
    masked = mask_text(text, "lenient", False, None, mode="gold")
    assert masked == text


@pytest.mark.gold
def test_gold_policy_strict_masks_iban():
    iban = "DE89370400440532013000"
    masked = mask_text(iban, "strict", False, None, mode="gold")
    assert masked != iban
    assert masked.startswith("[REDACTED_IBAN]")


@pytest.mark.gold
def test_session_scope_stabilises_tokens():
    text = "user@example.com"
    scope = ("tenant", "case", "salt")
    first = mask_text(
        text, "balanced", True, b"secret", mode="gold", session_scope=scope
    )
    second = mask_text(
        text, "balanced", True, b"secret", mode="gold", session_scope=scope
    )
    assert first == second


@pytest.mark.gold
def test_session_scope_changes_tokens_on_salt_change():
    text = "user@example.com"
    base = mask_text(
        text,
        "balanced",
        True,
        b"secret",
        mode="gold",
        session_scope=("tenant", "case", "salt-1"),
    )
    changed = mask_text(
        text,
        "balanced",
        True,
        b"secret",
        mode="gold",
        session_scope=("tenant", "case", "salt-2"),
    )
    assert base != changed


@pytest.mark.gold
def test_gold_metrics_counter_tracks_detections():
    if not hasattr(PII_DETECTIONS, "value"):
        pytest.skip("Prometheus counter backend not inspectable in tests")
    before = PII_DETECTIONS.value(tag="EMAIL")
    mask_text("user@example.com", "balanced", False, None, mode="gold")
    after = PII_DETECTIONS.value(tag="EMAIL")
    assert after == pytest.approx(before + 1)


def _default_config(**overrides):
    base = {
        "mode": "industrial",
        "policy": "balanced",
        "deterministic": False,
        "post_response": False,
        "logging_redaction": True,
        "hmac_secret": None,
        "name_detection": False,
    }
    base.update(overrides)
    return base


def test_mask_prompt_appends_summary_block():
    config = _default_config()
    masked = mask_prompt_with_config("Email user@example.com", config=config)
    assert "[REDACTED_EMAIL]" in masked
    assert masked.endswith("[REDACTED: tags=<EMAIL> policy=<balanced>]")


def test_mask_prompt_placeholder_only():
    config = _default_config()
    assert (
        mask_prompt_with_config("secret", placeholder_only=True, config=config)
        == "XXXX"
    )


def test_mask_response_respects_flag():
    config = _default_config()
    original = "Call +1-555-123456"
    assert mask_response(original, config=config) == original

    config = _default_config(post_response=True)
    masked = mask_response(original, config=config)
    assert "[REDACTED_PHONE]" in masked
    assert "[REDACTED: tags=<" not in masked
