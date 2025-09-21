"""Smoke tests for the structlog configuration."""

from __future__ import annotations

import json

from opentelemetry import trace

from common.logging import configure_logging, get_logger


def _emit_log_line(capsys) -> dict[str, object]:
    configure_logging()
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("test-span"):
        logger = get_logger(__name__)
        logger.info("logging smoke test", foo="bar")

    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    payload: dict[str, object] = json.loads(line)

    return payload


def test_logging_emits_expected_json_fields(monkeypatch, capsys):
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    payload = _emit_log_line(capsys)

    for field in (
        "timestamp",
        "level",
        "event",
        "service.name",
        "service.version",
        "deployment.environment",
        "trace_id",
        "span_id",
    ):
        assert field in payload
        assert isinstance(payload[field], str)

    assert payload["event"] == "logging smoke test"
    assert payload["foo"] == "bar"
    assert "logging.googleapis.com/trace" not in payload


def test_logging_adds_gcp_trace_when_project_present(monkeypatch, capsys):
    monkeypatch.setenv("GCP_PROJECT", "demo-project")
    payload = _emit_log_line(capsys)

    assert payload["logging.googleapis.com/trace"].startswith(
        "projects/demo-project/traces/"
    )
