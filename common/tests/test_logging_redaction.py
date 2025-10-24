import importlib
import io
import json
from contextlib import contextmanager
from typing import Iterable, Mapping
from urllib.parse import unquote

import structlog
from django.conf import settings
import pytest

# Respect environment: skip logging-redaction tests when masking is disabled.
_pii_mode_off = str(getattr(settings, "PII_MODE", "off")).lower() == "off"
_pii_policy_off = str(getattr(settings, "PII_POLICY", "off")).lower() == "off"
_pii_logging_disabled = not bool(getattr(settings, "PII_LOGGING_REDACTION", False))

pytestmark = pytest.mark.skipif(
    _pii_mode_off or _pii_policy_off or _pii_logging_disabled,
    reason="PII masking/logging redaction disabled by environment",
)
from django.test import override_settings

import common.logging as logging_utils
from ai_core.infra.pii_flags import clear_pii_config, set_pii_config


@contextmanager
def _scoped_pii_config(config: Mapping[str, object]):
    set_pii_config(config)
    try:
        yield
    finally:
        clear_pii_config()


def _pii_config(
    *,
    logging_redaction: bool,
    deterministic: bool,
    hmac_secret: str | bytes | None,
) -> dict[str, object]:
    secret_value: bytes | None
    if isinstance(hmac_secret, (bytes, bytearray)):
        secret_value = bytes(hmac_secret)
    elif hmac_secret in (None, ""):
        secret_value = None
    else:
        secret_value = str(hmac_secret).encode("utf-8")

    deterministic_enabled = bool(deterministic) and secret_value is not None

    # Tests must not depend on environment PII_MODE. Force industrial.
    return {
        "mode": "industrial",
        "policy": str(settings.PII_POLICY),
        "deterministic": deterministic_enabled,
        "post_response": bool(settings.PII_POST_RESPONSE),
        "logging_redaction": bool(logging_redaction),
        "hmac_secret": secret_value if deterministic_enabled else None,
        "name_detection": bool(settings.PII_NAME_DETECTION)
        and str(settings.PII_MODE) == "gold",
        "session_scope": None,
    }


def _capture_logs(
    entries: Iterable[tuple[str, Mapping[str, object] | None]],
    *,
    pii_config: Mapping[str, object],
    **overrides,
) -> list[dict[str, object]]:
    entries_list = list(entries)
    module = importlib.reload(logging_utils)
    module.LoggingInstrumentor = None
    stream = io.StringIO()
    structlog.reset_defaults()
    with override_settings(**overrides):
        module.configure_logging(stream)
        logger = module.get_logger("test")
        with _scoped_pii_config(pii_config):
            for event, payload in entries_list:
                extra = dict(payload or {})
                logger.info(event, **extra)
    contents = [line for line in stream.getvalue().splitlines() if line.strip()]
    if len(contents) > len(entries_list):
        contents = contents[-len(entries_list) :]
    return [json.loads(line) for line in contents]


def test_logging_redaction_masks_string_fields():
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("Contact user@example.com", {"detail": "+49 170 1234567", "optional": None})],
        pii_config=pii_config,
    )
    record = records[0]
    assert record["event"] == "Contact [REDACTED]"
    assert "user@example.com" not in record["event"]
    assert not record["event"].startswith('"')
    assert record["detail"] == "[REDACTED]"
    assert "1234567" not in record["detail"]
    assert not record["detail"].startswith('"')
    assert record["optional"] is None


def test_logging_redaction_deterministic_tokens():
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=True,
        hmac_secret="secret",
    )
    records = _capture_logs(
        [("user@example.com", None), ("user@example.com", None)],
        pii_config=pii_config,
    )
    events = [record["event"] for record in records]
    assert all(event == "[REDACTED]" for event in events)
    assert len(set(events)) == 1


def test_logging_redaction_can_be_disabled():
    pii_config = _pii_config(
        logging_redaction=False,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("user@example.com", None)],
        pii_config=pii_config,
    )
    event = records[0]["event"]
    assert event == "[REDACTED]"
    assert "user@example.com" not in event


def test_logging_redaction_preserves_json_spacing():
    payload = '{ "access_token": "secret", "note": "keep spacing" }'
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("event", {"payload": payload})],
        pii_config=pii_config,
    )
    masked_payload = records[0]["payload"]
    assert masked_payload != payload
    assert '"access_token": "[REDACTED]"' in masked_payload
    assert '", "note"' in masked_payload  # comma-space preserved


def test_logging_redaction_skips_structured_for_large_fields():
    large_chunk = "a" * (70 * 1024)
    large_text = f"prefix {large_chunk} user@example.com"
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [(large_text, None)],
        pii_config=pii_config,
    )
    event = records[0]["event"]
    assert event.endswith("[REDACTED]")


def test_logging_redaction_fast_path_preserves_large_json_spacing():
    large_value = "a" * (70 * 1024)
    payload = '{ "email": "user@example.com", "note": "' + large_value + '" }'
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("event", {"payload": payload})],
        pii_config=pii_config,
    )
    masked_payload = records[0]["payload"]
    assert masked_payload.startswith('{ "email": "')
    assert "user@example.com" not in masked_payload
    assert '"email": "[REDACTED]"' in masked_payload
    assert ' "note": "' in masked_payload
    assert masked_payload.endswith('" }')


def test_logging_redaction_leaves_pre_masked_tokens():
    pre_masked = "<EMAIL_ab12cd34>"
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=True,
        hmac_secret="secret",
    )
    records = _capture_logs(
        [(pre_masked, {"detail": "[REDACTED]", "note": "[REDACTED]"})],
        pii_config=pii_config,
    )
    record = records[0]
    assert record["event"] == pre_masked
    assert record["detail"] == "[REDACTED]"
    assert record["note"] == "[REDACTED]"


def test_logging_redaction_processor_order():
    module = importlib.reload(logging_utils)

    def dummy_redactor(logger, method_name, event_dict):
        return event_dict

    def dummy_pii(logger, method_name, event_dict):
        return event_dict

    processors = module._structlog_processors(dummy_redactor, dummy_pii)

    assert processors.index(dummy_pii) < processors.index(dummy_redactor)
    assert processors.index(dummy_redactor) < processors.index(
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter
    )


def test_logging_redaction_fast_path_skips_boring_messages():
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("processed 42 items", {"detail": "iteration complete"})],
        pii_config=pii_config,
    )
    record = records[0]
    assert record["event"] == "processed 42 items"
    assert record["detail"] == "iteration complete"


def test_logging_redaction_fast_path_skips_harmless_query_strings():
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("https://h/p?ok=1", None)],
        pii_config=pii_config,
    )
    assert records[0]["event"] == "https://h/p?ok=1"


def test_logging_redaction_fast_path_masks_email_queries():
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("https://h/p?email=a@b.de&ok=1", None)],
        pii_config=pii_config,
    )
    event = records[0]["event"]
    decoded = unquote(event)
    assert "a@b.de" not in decoded
    assert "[REDACTED]" in decoded


def test_logging_redaction_fast_path_masks_auth_headers():
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("login", {"auth": "Bearer eyJhbGciOi"})],
        pii_config=pii_config,
    )
    auth_value = records[0]["auth"]
    assert auth_value == "[REDACTED]"


def test_logging_redaction_fast_path_masks_phone_numbers():
    pii_config = _pii_config(
        logging_redaction=True,
        deterministic=False,
        hmac_secret=None,
    )
    records = _capture_logs(
        [("contact", {"text": "call +49 151 2345678"})],
        pii_config=pii_config,
    )
    masked = records[0]["text"]
    assert masked == "call [REDACTED]"
