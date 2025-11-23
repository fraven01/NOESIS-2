from datetime import datetime, timezone
from uuid import uuid4

import pytest

from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import tool_context_from_scope


def test_tool_context_from_scope_preserves_runtime_ids_and_defaults():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    scope = ScopeContext(
        tenant_id=str(uuid4()),
        trace_id="trace-123",
        invocation_id=str(uuid4()),
        run_id="run-abc",
        case_id="case-xyz",
        workflow_id="workflow-1",
        idempotency_key="idem-1",
        timestamp=timestamp,
    )

    tool_context = tool_context_from_scope(scope, locale="de-DE")

    assert tool_context.run_id == "run-abc"
    assert tool_context.ingestion_run_id is None
    assert tool_context.case_id == "case-xyz"
    assert tool_context.workflow_id == "workflow-1"
    assert tool_context.idempotency_key == "idem-1"
    assert tool_context.locale == "de-DE"
    assert tool_context.now_iso == timestamp


def test_tool_context_from_scope_supports_ingestion_runs_and_overrides_now():
    scope = ScopeContext(
        tenant_id=str(uuid4()),
        trace_id="trace-456",
        invocation_id=str(uuid4()),
        ingestion_run_id="ingest-123",
    )

    override_now = datetime(2024, 2, 1, tzinfo=timezone.utc)
    tool_context = scope.to_tool_context(now=override_now, budget_tokens=512)

    assert tool_context.run_id is None
    assert tool_context.ingestion_run_id == "ingest-123"
    assert tool_context.trace_id == "trace-456"
    assert tool_context.budget_tokens == 512
    assert tool_context.now_iso == override_now


def test_tool_context_from_scope_respects_xor_validation():
    with pytest.raises(ValueError):
        ScopeContext(
            tenant_id=str(uuid4()),
            trace_id="trace-789",
            invocation_id=str(uuid4()),
            run_id="run-1",
            ingestion_run_id="ingest-1",
        )
