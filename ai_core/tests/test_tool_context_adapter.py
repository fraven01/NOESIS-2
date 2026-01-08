from datetime import datetime, timezone
from uuid import uuid4


from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import tool_context_from_scope


def test_tool_context_from_scope_preserves_runtime_ids_and_defaults():
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    scope = ScopeContext(
        tenant_id=str(uuid4()),
        trace_id="trace-123",
        invocation_id=str(uuid4()),
        run_id="run-abc",
        tenant_schema="tenant-schema",
        idempotency_key="idem-1",
        timestamp=timestamp,
    )

    business = BusinessContext(case_id="case-xyz", workflow_id="workflow-1")
    tool_context = tool_context_from_scope(scope, business=business, locale="de-DE")

    assert tool_context.scope.run_id == "run-abc"
    assert tool_context.scope.ingestion_run_id is None
    assert tool_context.business.case_id == "case-xyz"
    assert tool_context.business.workflow_id == "workflow-1"
    assert tool_context.scope.tenant_schema == "tenant-schema"
    assert tool_context.scope.idempotency_key == "idem-1"
    assert tool_context.locale == "de-DE"
    assert tool_context.scope.timestamp == timestamp


def test_tool_context_from_scope_supports_ingestion_runs_and_overrides_now():
    scope = ScopeContext(
        tenant_id=str(uuid4()),
        trace_id="trace-456",
        invocation_id=str(uuid4()),
        ingestion_run_id="ingest-123",
        tenant_schema="tenant-schema-2",
    )

    business = BusinessContext(case_id="case-456")
    override_now = datetime(2024, 2, 1, tzinfo=timezone.utc)
    tool_context = scope.to_tool_context(
        business=business, now=override_now, budget_tokens=512
    )

    assert tool_context.scope.run_id is None
    assert tool_context.scope.ingestion_run_id == "ingest-123"
    assert tool_context.scope.trace_id == "trace-456"
    assert tool_context.scope.tenant_schema == "tenant-schema-2"
    assert tool_context.budget_tokens == 512
    assert tool_context.scope.timestamp == scope.timestamp


def test_tool_context_from_scope_allows_both_run_ids():
    """Both run_id and ingestion_run_id can co-exist (Pre-MVP ID Contract)."""
    scope = ScopeContext(
        tenant_id=str(uuid4()),
        trace_id="trace-789",
        invocation_id=str(uuid4()),
        run_id="run-1",
        ingestion_run_id="ingest-1",
    )

    business = BusinessContext(case_id="case-789")
    tool_context = scope.to_tool_context(business=business)

    assert tool_context.scope.run_id == "run-1"
    assert tool_context.scope.ingestion_run_id == "ingest-1"
