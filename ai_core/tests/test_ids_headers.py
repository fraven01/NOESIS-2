"""Tests for header normalization utilities."""

from __future__ import annotations

from typing import Any

import pytest

from ai_core.ids import (
    coerce_trace_id,
    normalize_case_header,
    normalize_idempotency_key,
    normalize_tenant_header,
)


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"HTTP_X_TENANT_ID": " foo "}, "foo"),
        ({"x-tenant-id": "bar"}, "bar"),
        ({"tenant_id": "baz"}, "baz"),
        ({"x-tenant-id": "  "}, None),
    ],
)
def test_normalize_tenant_header_variants(
    headers: dict[str, Any], expected: str | None
) -> None:
    assert normalize_tenant_header(headers) == expected


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"HTTP_X_CASE_ID": "  case-123  "}, "case-123"),
        ({"case-id": "abc"}, "abc"),
        ({"X-Case-ID": None}, None),
        ({"x-case-id": ""}, None),
    ],
)
def test_normalize_case_header_variants(
    headers: dict[str, Any], expected: str | None
) -> None:
    assert normalize_case_header(headers) == expected


@pytest.mark.parametrize(
    "headers,expected",
    [
        ({"x-idempotency-key": "idem"}, "idem"),
        ({"HTTP_IDEMPOTENCY_KEY": " key "}, "key"),
        ({"Idempotency-Key": "   "}, None),
    ],
)
def test_normalize_idempotency_key_variants(
    headers: dict[str, Any], expected: str | None
) -> None:
    assert normalize_idempotency_key(headers) == expected


def test_coerce_trace_id_prefers_traceparent() -> None:
    headers = {
        "Traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "X-Trace-ID": "ignored",
    }

    trace_id, span_id = coerce_trace_id(headers)

    assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert span_id == "00f067aa0ba902b7"


def test_coerce_trace_id_uses_explicit_trace_header_when_no_traceparent() -> None:
    headers = {"HTTP_X_TRACE_ID": "  explicit-trace  "}

    trace_id, span_id = coerce_trace_id(headers)

    assert trace_id == "explicit-trace"
    assert span_id is None


def test_coerce_trace_id_falls_back_to_request_id_and_warns() -> None:
    headers = {"request-id": "legacy"}
    messages: list[str] = []

    def _warn(message: str) -> None:
        messages.append(message)

    trace_id, span_id = coerce_trace_id(headers, warn=_warn)

    assert trace_id == "legacy"
    assert span_id is None
    assert any("request_id is deprecated" in msg for msg in messages)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"traceparent": "invalid-value"},
        {"X-Trace-ID": "   "},
    ],
)
def test_coerce_trace_id_raises_for_missing_values(headers: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        coerce_trace_id(headers)


def test_coerce_trace_id_handles_traceparent_without_span() -> None:
    headers = {"traceparent": "00-abc-"}

    trace_id, span_id = coerce_trace_id(headers)

    assert trace_id == "abc"
    assert span_id is None
