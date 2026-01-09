"""Tests for ScopeContext business ID rejection."""

from __future__ import annotations

import uuid

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


def test_scope_context_accepts_uuid_user_id() -> None:
    user_id = str(uuid.uuid4())
    payload = {
        "tenant_id": "tenant-1",
        "trace_id": "trace-1",
        "invocation_id": "inv-1",
        "run_id": "run-1",
        "user_id": user_id,
    }

    scope = ScopeContext.model_validate(payload)

    assert scope.user_id == user_id


@pytest.mark.parametrize("value", ["not-a-uuid", 1234])
def test_scope_context_rejects_non_uuid_user_id(value: object) -> None:
    payload = {
        "tenant_id": "tenant-1",
        "trace_id": "trace-1",
        "invocation_id": "inv-1",
        "run_id": "run-1",
        "user_id": value,
    }

    with pytest.raises(ValueError, match="user_id must be a UUID string"):
        ScopeContext.model_validate(payload)
