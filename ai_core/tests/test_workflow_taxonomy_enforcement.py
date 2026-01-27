from __future__ import annotations

import pytest

from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.agent.scope_policy import PolicyViolation, guard_mutation
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


def _tool_context(workflow_id: str) -> ToolContext:
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="run",
    )
    business = BusinessContext(case_id="case-1", workflow_id=workflow_id)
    return ToolContext(scope=scope, business=business, metadata={})


def test_case_scope_rejects_unknown_workflow_id() -> None:
    with pytest.raises(PolicyViolation):
        guard_mutation(
            "document_upsert",
            _tool_context("UNKNOWN"),
            RuntimeConfig(execution_scope="CASE"),
        )


def test_workflow_taxonomy_suppression_allows_migration_value() -> None:
    guard_mutation(
        "document_upsert",
        _tool_context("MIGRATION"),
        RuntimeConfig(execution_scope="CASE"),
    )
