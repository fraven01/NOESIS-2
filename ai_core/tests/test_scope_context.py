"""Tests for ScopeContext business ID rejection."""

from __future__ import annotations

import pytest

from ai_core.contracts.scope import ScopeContext


@pytest.mark.parametrize(
    "field_name",
    [
        "case_id",
        "collection_id",
        "workflow_id",
        "document_id",
        "document_version_id",
    ],
)
def test_scope_context_rejects_business_id_fields(field_name: str) -> None:
    payload = {
        "tenant_id": "tenant-1",
        "trace_id": "trace-1",
        "invocation_id": "inv-1",
        "run_id": "run-1",
        field_name: "value",
    }

    with pytest.raises(ValueError) as excinfo:
        ScopeContext.model_validate(payload)

    message = str(excinfo.value)
    assert "business IDs" in message
    assert field_name in message


def test_scope_context_rejects_business_ids() -> None:
    payload = {
        "tenant_id": "tenant-1",
        "trace_id": "trace-1",
        "invocation_id": "inv-1",
        "run_id": "run-1",
        "case_id": "case-1",
    }

    with pytest.raises(ValueError) as excinfo:
        ScopeContext.model_validate(payload)

    message = str(excinfo.value)
    assert "business IDs" in message
    assert "case_id" in message
