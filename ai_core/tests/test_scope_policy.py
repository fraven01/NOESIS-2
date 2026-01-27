from __future__ import annotations

import pytest

from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.agent.scope_policy import PolicyViolation, guard_mutation
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


def _make_tool_context(
    *, case_id: str | None = None, workflow_id: str | None = None
) -> ToolContext:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        user_id="00000000-0000-0000-0000-000000000001",
        run_id="run-1",
    )
    business = BusinessContext(
        case_id=case_id,
        workflow_id=workflow_id,
    )
    return ToolContext(scope=scope, business=business)


def test_system_scope_blocks_document_upsert():
    ctx = _make_tool_context(case_id="case-1", workflow_id="wf-1")
    config = RuntimeConfig(execution_scope="SYSTEM")

    with pytest.raises(PolicyViolation):
        guard_mutation("document_upsert", ctx, config, details={})


def test_tenant_scope_blocks_case_event():
    ctx = _make_tool_context(case_id="case-1", workflow_id="wf-1")
    config = RuntimeConfig(execution_scope="TENANT")

    with pytest.raises(PolicyViolation):
        guard_mutation("case_event", ctx, config, details={})


def test_case_scope_requires_case_id_and_workflow_id():
    ctx = _make_tool_context(case_id=None, workflow_id=None)
    config = RuntimeConfig(execution_scope="CASE")

    with pytest.raises(PolicyViolation):
        guard_mutation("document_upsert", ctx, config, details={})
