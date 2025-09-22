"""Smoke tests for the structlog configuration."""

from __future__ import annotations

import io
import json

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider

from common.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def _install_test_tracer_provider() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, ProxyTracerProvider):
        trace.set_tracer_provider(TracerProvider())


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


def test_logging_adds_spanId_when_trace_and_project(monkeypatch, capsys):
    configure_logging()

    class _Ctx:
        is_valid = True
        trace_id = 0xA
        span_id = 0xB

    class _Span:
        def get_span_context(self):
            return _Ctx()

    monkeypatch.setenv("GCP_PROJECT", "demo")
    monkeypatch.setattr("common.logging.trace.get_current_span", lambda: _Span())

    get_logger(__name__).info("span-present")

    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])

    assert payload["logging.googleapis.com/trace"].startswith(
        "projects/demo/traces/"
    )
    assert payload["logging.googleapis.com/spanId"] == f"{_Ctx.span_id:016x}"


def test_logging_uses_string_ids_when_span_invalid(monkeypatch, capsys):
    configure_logging()
    monkeypatch.delenv("GCP_PROJECT", raising=False)

    class _InvalidSpanContext:
        is_valid = False
        trace_id = 0
        span_id = 0

    class _InvalidSpan:
        def get_span_context(self):
            return _InvalidSpanContext()

    monkeypatch.setattr("common.logging.trace.get_current_span", lambda: _InvalidSpan())

    logger = get_logger(__name__)
    logger.info("invalid-span")

    payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])

    assert payload["trace_id"] == ""
    assert payload["span_id"] == ""


def test_configure_logging_switches_streams(capsys):
    configure_logging()
    logger = get_logger(__name__)
    logger.info("first-capture")

    first_capture = capsys.readouterr().err
    assert "first-capture" in first_capture

    alt_stream = io.StringIO()
    configure_logging(stream=alt_stream)
    logger.info("second-capture")

    assert "second-capture" in alt_stream.getvalue()
    assert "second-capture" not in capsys.readouterr().err


def test_configure_logging_recovers_from_closed_stream():
    first_stream = io.StringIO()
    configure_logging(stream=first_stream)

    logger = get_logger(__name__)
    logger.info("first")

    first_stream.close()

    second_stream = io.StringIO()
    configure_logging(stream=second_stream)
    logger.info("second")

    assert "second" in second_stream.getvalue()
