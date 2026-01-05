"""Chaos tests for ToolContext contract validation.

Tests the new compositional ToolContext structure with ScopeContext/BusinessContext
separation, ensuring proper parsing from meta dicts and validation of contract rules.

Contract under test:
- ai_core/tool_contracts/base.py: ToolContext, tool_context_from_meta()
- ai_core/contracts/scope.py: ScopeContext validation
- ai_core/contracts/business.py: BusinessContext separation
"""

from __future__ import annotations

import pytest

from ai_core.tool_contracts.base import tool_context_from_meta, ToolContext
from ai_core.contracts.scope import ScopeContext
from ai_core.contracts.business import BusinessContext
from tests.chaos.conftest import _build_chaos_meta

pytestmark = pytest.mark.chaos


def test_tool_context_from_new_meta_full():
    """Valid ToolContext parsing with full scope and business context."""
    meta = _build_chaos_meta(
        tenant_id="tenant-001",
        trace_id="trace-001",
        case_id="case-001",
        collection_id="collection-001",
    )

    context = tool_context_from_meta(meta)

    assert isinstance(context, ToolContext)
    assert context.scope.tenant_id == "tenant-001"
    assert context.scope.trace_id == "trace-001"
    assert context.scope.invocation_id == "chaos-invocation"
    assert context.scope.run_id == "chaos-run-default"
    assert context.scope.service_id == "chaos-test-runner"
    assert context.business.case_id == "case-001"
    assert context.business.collection_id == "collection-001"


def test_tool_context_from_new_meta_minimal():
    """Minimal valid ToolContext with scope only (no business context)."""
    meta = _build_chaos_meta(
        tenant_id="tenant-002",
        trace_id="trace-002",
        run_id="run-002",
    )

    context = tool_context_from_meta(meta)

    assert isinstance(context, ToolContext)
    assert context.scope.tenant_id == "tenant-002"
    assert context.scope.run_id == "run-002"
    assert context.business.case_id is None
    assert context.business.collection_id is None
    assert context.business.workflow_id is None


def test_tool_context_missing_runtime_id():
    """ScopeContext requires at least one runtime ID (run_id or ingestion_run_id)."""
    with pytest.raises(ValueError, match="At least one of run_id or ingestion_run_id"):
        ScopeContext(
            tenant_id="tenant-003",
            trace_id="trace-003",
            invocation_id="inv-003",
            service_id="test-service",
            # Missing both run_id and ingestion_run_id - INVALID
        )


def test_tool_context_business_id_rejection():
    """ScopeContext rejects business IDs (case_id, collection_id, etc.)."""
    with pytest.raises(ValueError, match="ScopeContext cannot include business IDs"):
        ScopeContext(
            tenant_id="tenant-004",
            trace_id="trace-004",
            invocation_id="inv-004",
            run_id="run-004",
            service_id="test-service",
            case_id="case-004",  # FORBIDDEN in ScopeContext
        )


def test_tool_context_identity_s2s_hop():
    """S2S Hop (Chaos tests) requires service_id, not user_id."""
    meta = _build_chaos_meta(
        tenant_id="tenant-005",
        trace_id="trace-005",
        service_id="chaos-s2s-service",
        run_id="run-005",
    )

    context = tool_context_from_meta(meta)

    assert context.scope.service_id == "chaos-s2s-service"
    assert context.scope.user_id is None  # User Request Hops only


def test_tool_context_identity_mutual_exclusion():
    """user_id and service_id are mutually exclusive."""
    with pytest.raises(
        ValueError, match="user_id and service_id are mutually exclusive"
    ):
        ScopeContext(
            tenant_id="tenant-006",
            trace_id="trace-006",
            invocation_id="inv-006",
            run_id="run-006",
            user_id="user-006",  # User Request Hop
            service_id="service-006",  # S2S Hop
            # Both set - INVALID
        )


def test_tool_context_backward_compat_properties():
    """Deprecated properties still work for backward compatibility."""
    meta = _build_chaos_meta(
        tenant_id="tenant-007",
        trace_id="trace-007",
        case_id="case-007",
        collection_id="collection-007",
        run_id="run-007",
    )

    context = tool_context_from_meta(meta)

    # NEW (preferred):
    assert context.scope.tenant_id == "tenant-007"
    assert context.business.case_id == "case-007"

    # OLD (deprecated but still works):
    assert context.tenant_id == "tenant-007"  # delegates to scope
    assert context.case_id == "case-007"  # delegates to business
    assert context.run_id == "run-007"  # delegates to scope


def test_tool_context_runtime_id_coexistence():
    """run_id and ingestion_run_id may co-exist (workflow triggers ingestion)."""
    meta = _build_chaos_meta(
        tenant_id="tenant-008",
        trace_id="trace-008",
        run_id="run-008",
        ingestion_run_id="ingestion-008",  # Both set - VALID
    )

    context = tool_context_from_meta(meta)

    assert context.scope.run_id == "run-008"
    assert context.scope.ingestion_run_id == "ingestion-008"


def test_tool_context_from_prebuilt_context():
    """tool_context_from_meta() accepts prebuilt ToolContext."""
    scope = ScopeContext(
        tenant_id="tenant-009",
        trace_id="trace-009",
        invocation_id="inv-009",
        run_id="run-009",
        service_id="service-009",
    )
    business = BusinessContext(case_id="case-009")
    prebuilt = ToolContext(scope=scope, business=business)

    meta = {"tool_context": prebuilt}
    context = tool_context_from_meta(meta)

    assert context is prebuilt  # Same instance returned


def test_tool_context_missing_scope_context():
    """tool_context_from_meta() requires scope_context or tool_context."""
    meta = {
        "business_context": {"case_id": "case-010"},
        # Missing scope_context - INVALID
    }

    with pytest.raises(ValueError, match="tool_context or scope_context is required"):
        tool_context_from_meta(meta)
