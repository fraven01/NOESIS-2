"""Tests for metadata normalisation helpers."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ai_core.graph import schemas
from ai_core.graph.schemas import normalize_meta
from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_TRACE_ID_HEADER,
)


def _request(headers: dict | None = None, **attrs):
    base_headers = headers or {}
    return SimpleNamespace(headers=base_headers, META={}, **attrs)


@pytest.mark.django_db
def test_normalize_meta_returns_expected_mapping(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 7)
    tenant_id = str(uuid4())
    run_id = "run-42"
    invocation_id = uuid4()
    request = _request(
        {
            X_TENANT_ID_HEADER: tenant_id,
            X_CASE_ID_HEADER: "case-42",
            X_TRACE_ID_HEADER: "trace-123",
            X_TENANT_SCHEMA_HEADER: "tenant_schema",
            X_KEY_ALIAS_HEADER: "alias-1",
        },
        graph_name="info_intake",
        graph_version="v9",
        run_id=run_id,
        invocation_id=invocation_id,
    )

    meta = normalize_meta(request)

    scope_context = meta["scope_context"]
    business_context = meta["business_context"]
    assert scope_context["tenant_id"] == tenant_id
    assert scope_context["trace_id"] == "trace-123"
    assert "case_id" not in scope_context
    assert business_context["case_id"] == "case-42"
    assert meta["graph_name"] == "info_intake"
    assert meta["graph_version"] == "v9"
    assert meta["tenant_schema"] == "tenant_schema"
    assert meta["key_alias"] == "alias-1"
    assert meta["rate_limit"] == {"quota": 7}
    assert "requested_at" in meta
    # Ensure the timestamp is ISO 8601 parseable.
    datetime.fromisoformat(meta["requested_at"])

    assert scope_context["run_id"] == run_id
    assert scope_context["invocation_id"] == str(invocation_id)

    context_metadata = meta["context_metadata"]
    assert context_metadata["graph_name"] == "info_intake"
    assert context_metadata["graph_version"] == "v9"
    assert context_metadata["requested_at"] == meta["requested_at"]

    tool_context = meta["tool_context"]
    assert isinstance(tool_context, dict)
    from ai_core.tool_contracts import ToolContext

    context = ToolContext.model_validate(tool_context)
    assert str(context.scope.tenant_id) == tenant_id
    assert context.business.case_id == "case-42"
    assert context.scope.trace_id == "trace-123"
    assert context.scope.idempotency_key is None
    assert context.scope.tenant_schema == "tenant_schema"
    assert context.metadata["graph_name"] == "info_intake"
    assert context.metadata["graph_version"] == "v9"
    assert context.metadata["requested_at"] == meta["requested_at"]


@pytest.mark.django_db
def test_normalize_meta_raises_on_missing_required_keys(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 3)
    request = _request(
        {X_CASE_ID_HEADER: "case-1", X_TRACE_ID_HEADER: "trace-1"},
        graph_name="info_intake",
    )

    with pytest.raises(ValueError) as excinfo:
        normalize_meta(request)

    message = str(excinfo.value)
    assert "tenant_id" in message


@pytest.mark.django_db
def test_normalize_meta_defaults_graph_version(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 5)
    tenant_id = str(uuid4())
    request = _request(
        {
            X_TENANT_ID_HEADER: tenant_id,
            X_CASE_ID_HEADER: "case-123",
            X_TRACE_ID_HEADER: "trace-999",
            IDEMPOTENCY_KEY_HEADER: "idem-1",
        },
        graph_name="retrieval_augmented_generation",
    )

    meta = normalize_meta(request)

    assert meta["graph_version"] == "v0"
    tool_context = meta["tool_context"]
    from ai_core.tool_contracts import ToolContext

    context = ToolContext.model_validate(tool_context)
    assert context.scope.idempotency_key == "idem-1"
    assert context.metadata["graph_version"] == "v0"
    assert meta["scope_context"]["idempotency_key"] == "idem-1"


@pytest.mark.django_db
def test_normalize_meta_includes_tool_context(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 2)
    tenant_id = str(uuid4())
    request = _request(
        {
            X_TENANT_ID_HEADER: tenant_id,
            X_CASE_ID_HEADER: "case-b",
            X_TRACE_ID_HEADER: "trace-b",
            IDEMPOTENCY_KEY_HEADER: "idem-b",
        },
        graph_name="info_intake",
    )

    meta = normalize_meta(request)

    tool_context = meta["tool_context"]
    assert isinstance(tool_context, dict)
    from ai_core.tool_contracts import ToolContext

    context = ToolContext.model_validate(tool_context)
    assert str(context.scope.tenant_id) == tenant_id
    assert context.business.case_id == "case-b"
    assert context.scope.trace_id == "trace-b"
    assert context.scope.idempotency_key == "idem-b"
    assert context.metadata["graph_name"] == "info_intake"
    assert context.metadata["graph_version"] == "v0"
    assert context.metadata["requested_at"] == meta["requested_at"]
    assert meta["scope_context"]["idempotency_key"] == "idem-b"


@pytest.mark.django_db
def test_normalize_meta_includes_collection_scope(monkeypatch):
    monkeypatch.setattr(schemas, "get_quota", lambda: 1)
    tenant_id = str(uuid4())
    request = _request(
        {
            X_TENANT_ID_HEADER: tenant_id,
            X_CASE_ID_HEADER: "case-c",
            X_TRACE_ID_HEADER: "trace-c",
            X_COLLECTION_ID_HEADER: "54d8d3b2-04de-4a38-a9c8-3c9a4b52c5b6",
        },
        graph_name="info_intake",
    )

    meta = normalize_meta(request)

    assert (
        meta["business_context"]["collection_id"]
        == "54d8d3b2-04de-4a38-a9c8-3c9a4b52c5b6"
    )
    from ai_core.tool_contracts import ToolContext

    context = ToolContext.model_validate(meta["tool_context"])
    assert context.business.collection_id == "54d8d3b2-04de-4a38-a9c8-3c9a4b52c5b6"
