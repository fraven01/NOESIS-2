"""Tests for ScopeContext business ID rejection."""

from __future__ import annotations

import pytest

from ai_core.contracts.scope import ScopeContext


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
