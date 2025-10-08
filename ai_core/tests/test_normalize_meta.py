"""Tests for metadata normalisation helpers."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from ai_core.graph import schemas
from ai_core.graph.schemas import ToolContext, normalize_meta
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_TRACE_ID_HEADER,
)


def _request(headers: dict | None = None, **attrs):
    base_headers = headers or {}
    return SimpleNamespace(headers=base_headers, META={}, **attrs)


def test_normalize_meta_returns_expected_mapping(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 7)
    request = _request(
        {
            X_TENANT_ID_HEADER: "tenant-a",
            X_CASE_ID_HEADER: "case-42",
            X_TRACE_ID_HEADER: "trace-123",
            X_TENANT_SCHEMA_HEADER: "tenant_schema",
            X_KEY_ALIAS_HEADER: "alias-1",
        },
        graph_name="info_intake",
        graph_version="v9",
    )

    meta = normalize_meta(request)

    assert meta["tenant_id"] == "tenant-a"
    assert meta["case_id"] == "case-42"
    assert meta["trace_id"] == "trace-123"
    assert meta["graph_name"] == "info_intake"
    assert meta["graph_version"] == "v9"
    assert meta["tenant_schema"] == "tenant_schema"
    assert meta["key_alias"] == "alias-1"
    assert meta["rate_limit"] == {"quota": 7}
    assert "requested_at" in meta
    # Ensure the timestamp is ISO 8601 parseable.
    datetime.fromisoformat(meta["requested_at"])

    tool_context = meta["tool_context"]
    assert isinstance(tool_context, dict)
    assert tool_context == {
        "tenant_id": "tenant-a",
        "case_id": "case-42",
        "trace_id": "trace-123",
        "idempotency_key": None,
    }
    assert ToolContext(**tool_context) == ToolContext(
        tenant_id="tenant-a",
        case_id="case-42",
        trace_id="trace-123",
        idempotency_key=None,
    )


def test_normalize_meta_raises_on_missing_required_keys(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 3)
    request = _request({X_TENANT_ID_HEADER: "tenant-a"}, graph_name="info_intake")

    with pytest.raises(ValueError) as excinfo:
        normalize_meta(request)

    message = str(excinfo.value)
    assert "case_id" in message
    assert "trace_id" in message


def test_normalize_meta_defaults_graph_version(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 5)
    request = _request(
        {
            X_TENANT_ID_HEADER: "tenant-a",
            X_CASE_ID_HEADER: "case-123",
            X_TRACE_ID_HEADER: "trace-999",
            IDEMPOTENCY_KEY_HEADER: "idem-1",
        },
        graph_name="needs",
    )

    meta = normalize_meta(request)

    assert meta["graph_version"] == "v0"
    tool_context = meta["tool_context"]
    assert tool_context["idempotency_key"] == "idem-1"
    assert meta["idempotency_key"] == "idem-1"


def test_normalize_meta_includes_tool_context(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 2)
    request = _request(
        {
            X_TENANT_ID_HEADER: "tenant-b",
            X_CASE_ID_HEADER: "case-b",
            X_TRACE_ID_HEADER: "trace-b",
            IDEMPOTENCY_KEY_HEADER: "idem-b",
        },
        graph_name="info_intake",
    )

    meta = normalize_meta(request)

    tool_context = meta["tool_context"]
    assert isinstance(tool_context, dict)
    assert tool_context == {
        "tenant_id": "tenant-b",
        "case_id": "case-b",
        "trace_id": "trace-b",
        "idempotency_key": "idem-b",
    }
    assert ToolContext(**tool_context) == ToolContext(
        tenant_id="tenant-b",
        case_id="case-b",
        trace_id="trace-b",
        idempotency_key="idem-b",
    )
    assert meta["idempotency_key"] == "idem-b"
