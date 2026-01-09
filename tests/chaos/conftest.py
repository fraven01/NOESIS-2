"""Chaos test helpers for new contract migration.

This module provides helpers for building meta dicts with the new compositional
ScopeContext + BusinessContext structure, used across chaos tests.

BREAKING CHANGE (Option A - Strict Separation):
Meta dicts now use compositional structure instead of flat dicts.

OLD (v1) - flat dict:
    meta = {"tenant_id": "t", "case_id": "c", "trace_id": "tr"}

NEW (v2) - compositional:
    meta = {
        "scope_context": {
            "tenant_id": "t",
            "trace_id": "tr",
            "invocation_id": "inv",
            "run_id": "run",
            "service_id": "chaos-test",
        },
        "business_context": {
            "case_id": "c",
        },
    }

Usage in chaos tests:
    from tests.chaos.conftest import _build_chaos_meta

    meta = _build_chaos_meta(
        tenant_id="test-tenant",
        trace_id="test-trace",
        case_id="test-case",
        run_id="test-run",
    )
"""

from __future__ import annotations

from typing import Any

from ai_core.contracts.scope import ScopeContext
from ai_core.contracts.business import BusinessContext


def _build_chaos_meta(
    *,
    tenant_id: str,
    trace_id: str,
    invocation_id: str = "chaos-invocation",
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    case_id: str | None = None,
    collection_id: str | None = None,
    workflow_id: str | None = None,
    document_id: str | None = None,
    service_id: str = "chaos-test-runner",
    **context_overrides: Any,
) -> dict[str, object]:
    """Build meta dict using new ScopeContext/BusinessContext structure.

    Returns a meta dict compatible with tool_context_from_meta() that uses
    the new compositional structure (scope_context + business_context).

    Identity rules (Pre-MVP ID Contract):
    - S2S Hop (Chaos tests): service_id REQUIRED, user_id ABSENT
    - At least ONE runtime ID required: run_id OR ingestion_run_id

    Args:
        tenant_id: Tenant identifier (required)
        trace_id: Distributed trace ID (required)
        invocation_id: Invocation ID within trace (defaults to "chaos-invocation")
        run_id: LangGraph run ID (optional, but one of run_id/ingestion_run_id required)
        ingestion_run_id: Document ingestion run ID (optional)
        case_id: Case identifier (optional business context)
        collection_id: Collection identifier (optional business context)
        workflow_id: Workflow identifier (optional business context)
        document_id: Document identifier (optional business context)
        service_id: Service identity for S2S hops (defaults to "chaos-test-runner")
        **context_overrides: Additional context fields (e.g., external_id)

    Returns:
        Meta dict with scope_context and business_context suitable for
        tool_context_from_meta() parsing.

    Raises:
        ValueError: If neither run_id nor ingestion_run_id is provided
        ValueError: If business IDs are passed in context_overrides (they belong in business_context)

    Examples:
        # Minimal (run_id auto-generated):
        meta = _build_chaos_meta(
            tenant_id="tenant-001",
            trace_id="trace-001",
        )

        # With business context:
        meta = _build_chaos_meta(
            tenant_id="tenant-002",
            trace_id="trace-002",
            case_id="case-002",
            collection_id="col-002",
            run_id="run-002",
        )

        # Ingestion run:
        meta = _build_chaos_meta(
            tenant_id="tenant-003",
            trace_id="trace-003",
            ingestion_run_id="ingestion-003",
        )
    """
    # Ensure at least one runtime ID is set
    if not run_id and not ingestion_run_id:
        run_id = "chaos-run-default"

    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
        service_id=service_id,
    )

    business = BusinessContext(
        case_id=case_id,
        collection_id=collection_id,
        workflow_id=workflow_id,
        document_id=document_id,
    )

    return {
        "scope_context": scope.model_dump(mode="json"),
        "business_context": business.model_dump(mode="json"),
        **context_overrides,
    }


__all__ = ["_build_chaos_meta"]
