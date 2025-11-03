"""Tests for ai_core.ids.contracts."""

from __future__ import annotations


import pytest

from ai_core.ids import (
    K_REQUIRED_SPAN_ATTRS,
    CorrelationIds,
    DocumentRef,
    normalize_trace_id,
    require_ids,
)


def test_require_ids_accepts_complete_payload() -> None:
    meta = {"tenant_id": "tenant", "trace_id": "trace", "workflow_id": "flow"}

    require_ids(meta)


def test_require_ids_raises_for_missing_fields() -> None:
    meta = {"tenant_id": "tenant"}

    with pytest.raises(ValueError) as exc:
        require_ids(meta)

    assert "trace_id" in str(exc.value)
    assert "workflow_id" in str(exc.value)


def test_normalize_trace_id_strips_and_returns_value() -> None:
    meta = {"tenant_id": "tenant", "trace_id": "  trace-value  "}

    result = normalize_trace_id(meta)

    assert result == "trace-value"
    assert meta["trace_id"] == "trace-value"


def test_normalize_trace_id_missing_values_raise_error() -> None:
    meta = {"tenant_id": "tenant"}

    with pytest.raises(ValueError):
        normalize_trace_id(meta)


def test_correlation_and_document_ref_dataclasses() -> None:
    correlation = CorrelationIds(
        tenant_id="tenant",
        trace_id="trace",
        workflow_id="workflow",
        case_id="case",
        span_id="span",
    )
    document_ref = DocumentRef(
        tenant_id="tenant",
        workflow_id="workflow",
        document_id="document",
        collection_id="collection",
        version="1",
    )

    assert correlation.case_id == "case"
    assert document_ref.collection_id == "collection"


def test_contract_constants() -> None:
    assert K_REQUIRED_SPAN_ATTRS == ("tenant_id", "case_id", "trace_id", "workflow_id")


def test_require_ids_custom_required_fields() -> None:
    meta = {"tenant_id": "tenant", "trace_id": "trace"}

    require_ids(meta, required=("tenant_id", "trace_id"))

    with pytest.raises(ValueError):
        require_ids(meta, required=("tenant_id", "trace_id", "workflow_id"))
