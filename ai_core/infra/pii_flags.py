"""PII configuration access helpers."""

from django.conf import settings


def get_pii_config() -> dict[str, object]:
    """Return the PII configuration derived from Django settings."""

    return {
        "mode": settings.PII_MODE,
        "policy": settings.PII_POLICY,
        "deterministic": settings.PII_DETERMINISTIC and bool(settings.PII_HMAC_SECRET),
        "post_response": settings.PII_POST_RESPONSE,
        "logging_redaction": settings.PII_LOGGING_REDACTION,
        "hmac_secret": (
            settings.PII_HMAC_SECRET.encode("utf-8") if settings.PII_HMAC_SECRET else None
        ),
        "name_detection": settings.PII_NAME_DETECTION and settings.PII_MODE == "gold",
    }
