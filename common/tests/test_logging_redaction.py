import importlib
import io
import json
from typing import Iterable, Mapping
from urllib.parse import unquote

import structlog
from django.test import override_settings

import common.logging as logging_utils


def _capture_logs(
    entries: Iterable[tuple[str, Mapping[str, object] | None]],
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
        for event, payload in entries_list:
            extra = dict(payload or {})
            logger.info(event, **extra)
    contents = [line for line in stream.getvalue().splitlines() if line.strip()]
    if len(contents) > len(entries_list):
        contents = contents[-len(entries_list) :]
    return [json.loads(line) for line in contents]


def test_logging_redaction_masks_string_fields():
    records = _capture_logs(
        [("Contact user@example.com", {"detail": "+49 170 1234567", "optional": None})],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    record = records[0]
    assert "[REDACTED_EMAIL]" in record["event"]
    assert "user@example.com" not in record["event"]
    assert not record["event"].startswith('"')
    assert record["detail"] == "[REDACTED_PHONE]"
    assert not record["detail"].startswith('"')
    assert record["optional"] is None


def test_logging_redaction_deterministic_tokens():
    records = _capture_logs(
        [("user@example.com", None), ("user@example.com", None)],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=True,
        PII_HMAC_SECRET="secret",
    )
    events = [record["event"] for record in records]
    assert all(event.startswith("<EMAIL_") for event in events)
    assert len(set(events)) == 1


def test_logging_redaction_can_be_disabled():
    records = _capture_logs(
        [("user@example.com", None)],
        PII_LOGGING_REDACTION=False,
    )
    event = records[0]["event"]
    assert event != "[REDACTED_EMAIL]"
    assert not event.startswith("<EMAIL_")


def test_logging_redaction_preserves_json_spacing():
    payload = '{ "access_token": "secret", "note": "keep spacing" }'
    records = _capture_logs(
        [("event", {"payload": payload})],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    masked_payload = records[0]["payload"]
    assert masked_payload != payload
    assert '"access_token": "[REDACTED_ACCESS_TOKEN]"' in masked_payload
    assert '", "note"' in masked_payload  # comma-space preserved


def test_logging_redaction_skips_structured_for_large_fields():
    large_chunk = "a" * (70 * 1024)
    large_text = f"prefix {large_chunk} user@example.com"
    records = _capture_logs(
        [(large_text, None)],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    event = records[0]["event"]
    assert "[REDACTED_EMAIL]" in event


def test_logging_redaction_fast_path_preserves_large_json_spacing():
    large_value = "a" * (70 * 1024)
    payload = '{ "email": "user@example.com", "note": "' + large_value + '" }'
    records = _capture_logs(
        [("event", {"payload": payload})],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    masked_payload = records[0]["payload"]
    assert masked_payload.startswith('{ "email": "')
    assert 'user@example.com' not in masked_payload
    assert '"email": "[REDACTED_EMAIL]"' in masked_payload
    assert ' "note": "' in masked_payload
    assert masked_payload.endswith('" }')


def test_logging_redaction_leaves_pre_masked_tokens():
    pre_masked = "<EMAIL_ab12cd34>"
    records = _capture_logs(
        [(pre_masked, {"detail": "[REDACTED_EMAIL]", "note": "[REDACTED]"})],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=True,
        PII_HMAC_SECRET="secret",
    )
    record = records[0]
    assert record["event"] == pre_masked
    assert record["detail"] == "[REDACTED_EMAIL]"
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
    records = _capture_logs(
        [("processed 42 items", {"detail": "iteration complete"})],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    record = records[0]
    assert record["event"] == "processed 42 items"
    assert record["detail"] == "iteration complete"


def test_logging_redaction_fast_path_skips_harmless_query_strings():
    records = _capture_logs(
        [("https://h/p?ok=1", None)],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    assert records[0]["event"] == "https://h/p?ok=1"


def test_logging_redaction_fast_path_masks_email_queries():
    records = _capture_logs(
        [("https://h/p?email=a@b.de&ok=1", None)],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    event = records[0]["event"]
    decoded = unquote(event)
    assert "a@b.de" not in decoded
    assert "[REDACTED_EMAIL]" in decoded


def test_logging_redaction_fast_path_masks_auth_headers():
    records = _capture_logs(
        [("login", {"auth": "Bearer eyJhbGciOi"})],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    auth_value = records[0]["auth"]
    assert auth_value != "Bearer eyJhbGciOi"
    assert auth_value.startswith("[REDACTED")


def test_logging_redaction_fast_path_masks_phone_numbers():
    records = _capture_logs(
        [("contact", {"text": "call +49 151 2345678"})],
        PII_LOGGING_REDACTION=True,
        PII_DETERMINISTIC=False,
        PII_HMAC_SECRET="",
    )
    masked = records[0]["text"]
    assert masked.startswith("call ")
    assert "[REDACTED_PHONE]" in masked
    assert "2345678" not in masked
