import os

import pytest

from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.policy import clear_session_scope, set_session_scope
from ai_core.infra.pii import mask_text


@pytest.mark.skipif(
    os.getenv("PII_MODE", "industrial") != "industrial",
    reason="default config is only asserted in industrial mode",
)
def test_get_pii_config_defaults(settings):
    """Smoke test that the default PII configuration is wired correctly."""

    assert settings.PII_MODE == "industrial"
    assert settings.PII_POLICY == "balanced"
    assert settings.PII_DETERMINISTIC is True
    assert settings.PII_POST_RESPONSE is False
    assert settings.PII_LOGGING_REDACTION is True
    assert settings.PII_HMAC_SECRET == ""
    assert settings.PII_NAME_DETECTION is False

    config = get_pii_config()

    assert config == {
        "mode": "industrial",
        "policy": "balanced",
        "deterministic": False,
        "post_response": False,
        "logging_redaction": True,
        "hmac_secret": None,
        "name_detection": False,
    }


@pytest.mark.gold
def test_get_pii_config_gold_mode(pii_config_env):
    overrides = pii_config_env(
        PII_MODE="gold",
        PII_POLICY="strict",
        PII_DETERMINISTIC=True,
        PII_POST_RESPONSE=True,
        PII_LOGGING_REDACTION=False,
        PII_HMAC_SECRET="secret",
        PII_NAME_DETECTION=True,
    )

    assert overrides == {
        "PII_MODE": "gold",
        "PII_POLICY": "strict",
        "PII_DETERMINISTIC": True,
        "PII_POST_RESPONSE": True,
        "PII_LOGGING_REDACTION": False,
        "PII_HMAC_SECRET": "secret",
        "PII_NAME_DETECTION": True,
    }

    config = get_pii_config()

    assert config == {
        "mode": "gold",
        "policy": "strict",
        "deterministic": True,
        "post_response": True,
        "logging_redaction": False,
        "hmac_secret": b"secret",
        "name_detection": True,
    }


def test_mask_text_deterministic_under_scope(pii_config_env):
    pii_config_env(
        PII_HMAC_SECRET="scope-secret",
        PII_DETERMINISTIC=True,
    )

    config = get_pii_config()
    assert config["deterministic"] is True
    assert config["hmac_secret"] == b"scope-secret"

    set_session_scope(
        tenant_id="tenant-a",
        case_id="case-42",
        session_salt="trace-99",
    )
    try:
        first = mask_text(
            "Contact user@example.com",
            config["policy"],
            config["deterministic"],
            config["hmac_secret"],
            mode=config["mode"],
        )
        second = mask_text(
            "Contact user@example.com",
            config["policy"],
            config["deterministic"],
            config["hmac_secret"],
            mode=config["mode"],
        )
    finally:
        clear_session_scope()

    assert first == second
    assert "<EMAIL_" in first


def test_pii_middleware_order(settings):
    middleware = list(settings.MIDDLEWARE)
    assert "ai_core.middleware.PIISessionScopeMiddleware" in middleware
    assert "ai_core.middleware.RequestContextMiddleware" in middleware

    scope_index = middleware.index("ai_core.middleware.PIISessionScopeMiddleware")
    request_context_index = middleware.index("ai_core.middleware.RequestContextMiddleware")

    assert scope_index < request_context_index, (
        "PII session scope middleware must run before request context logging middleware"
    )
