from __future__ import annotations

import json

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


def _make_context(
    *,
    trace_id: str = "trace-1",
    invocation_id: str = "inv-1",
    tenant_id: str = "tenant-1",
    user_id: str = "00000000-0000-0000-0000-000000000001",
    run_id: str = "run-1",
    metadata: dict[str, object] | None = None,
    locale: str | None = None,
) -> ToolContext:
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        user_id=user_id,
        run_id=run_id,
    )
    business = BusinessContext(case_id="case-1", collection_id="col-1")
    return ToolContext(
        scope=scope,
        business=business,
        metadata=metadata or {},
        locale=locale,
    )


def test_tool_context_hash_same_for_different_headers():
    ctx_a = _make_context(
        trace_id="trace-a",
        metadata={"headers": {"X-Test": "one"}, "user_agent": "agent-1"},
        locale="en-US",
    )
    ctx_b = _make_context(
        trace_id="trace-b",
        metadata={"headers": {"X-Test": "two"}, "user_agent": "agent-2"},
        locale="de-DE",
    )

    assert ctx_a.tool_context_hash == ctx_b.tool_context_hash


def test_tool_context_hash_changes_when_identity_or_scope_changes():
    base = _make_context()
    changed_user = _make_context(user_id="00000000-0000-0000-0000-000000000002")
    changed_tenant = _make_context(tenant_id="tenant-2")

    assert base.tool_context_hash != changed_user.tool_context_hash
    assert base.tool_context_hash != changed_tenant.tool_context_hash


def test_tool_context_canonical_json_is_deterministic_order():
    ctx = _make_context()
    expected_payload = {
        "scope": {
            "tenant_id": "tenant-1",
            "user_id": "00000000-0000-0000-0000-000000000001",
            "service_id": None,
            "run_id": "run-1",
            "ingestion_run_id": None,
            "tenant_schema": None,
        },
        "business": {
            "case_id": "case-1",
            "collection_id": "col-1",
            "workflow_id": None,
            "thread_id": None,
            "document_id": None,
            "document_version_id": None,
        },
    }
    expected_json = json.dumps(
        expected_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    assert ctx.canonical_json() == expected_json
