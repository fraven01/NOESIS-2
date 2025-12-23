"""Test utilities for AI Core."""

from __future__ import annotations

import uuid
from typing import Any, Mapping

from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


class GraphTestMixin:
    """Mixin to simplify graph testing with automatic ID generation."""

    def make_graph_state(
        self,
        input_data: Mapping[str, Any],
        *,
        tenant_id: str = "test-tenant",
        case_id: str | None = "test-case",
        workflow_id: str | None = "test-workflow",
        trace_id: str | None = "test-trace",
        run_id: str | None = "test-run",
        ingestion_run_id: str | None = None,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a standardized graph state dictionary.

        Automatically handles ScopeContext structure and ID generation.
        Enforces mutual exclusion of run_id and ingestion_run_id (defaults to run_id).
        """
        # Build context dict with all IDs for CollectionSearchAdapter compatibility
        context = make_test_meta(
            tenant_id=tenant_id,
            case_id=case_id,
            workflow_id=workflow_id,
            trace_id=trace_id,
            run_id=run_id,
            ingestion_run_id=ingestion_run_id,
            extra=extra_meta,
        )

        # Return format compatible with CollectionSearchAdapter.run(state, meta)
        # where state contains input dict and meta is separate
        return {
            "input": dict(input_data),
            "context": context,  # CollectionSearchAdapter expects context at top level
        }

    def make_scope_context(self, **overrides: Any) -> ScopeContext:
        """Create a valid ScopeContext with defaults."""
        defaults = {
            "tenant_id": "test-tenant",
            "trace_id": "test-trace",
            "invocation_id": str(uuid.uuid4()),
            "run_id": "test-run",
        }
        defaults.update(overrides)
        return ScopeContext(**defaults)

    def make_tool_context(self, **overrides: Any) -> ToolContext:
        """Create a valid ToolContext with defaults."""
        defaults = {
            "tenant_id": "test-tenant",
            "trace_id": "test-trace",
            "invocation_id": str(uuid.uuid4()),
            "run_id": "test-run",
        }
        defaults.update(overrides)
        return ToolContext(**defaults)


def make_test_ids(
    *,
    tenant_id: str = "test-tenant",
    case_id: str | None = "test-case",
    workflow_id: str | None = "test-workflow",
    trace_id: str | None = "test-trace",
    run_id: str | None = "test-run",
    ingestion_run_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate a consistent set of test IDs.

    Both run_id and ingestion_run_id may co-exist (Pre-MVP ID Contract).
    At least one runtime ID is required.
    """
    if not run_id and not ingestion_run_id:
        run_id = str(uuid.uuid4())

    ids = {
        "tenant_id": tenant_id,
        "trace_id": trace_id or str(uuid.uuid4()),
        "case_id": case_id,
        "workflow_id": workflow_id,
    }

    if run_id:
        ids["run_id"] = run_id
    if ingestion_run_id:
        ids["ingestion_run_id"] = ingestion_run_id

    return {k: v for k, v in ids.items() if v is not None}


def make_test_meta(
    extra: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Create a standard metadata dictionary for graph execution.

    Accepts all arguments from `make_test_ids` via kwargs.
    """
    ids = make_test_ids(**kwargs)
    if extra:
        ids.update(extra)
    return ids


def make_test_scope_context(
    *,
    tenant_id: str = "test-tenant-id",
    trace_id: str | None = None,
    invocation_id: str | None = None,
    case_id: str | None = None,
    workflow_id: str | None = None,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    user_id: str | None = None,
    service_id: str | None = None,
    collection_id: str | None = None,
    idempotency_key: str | None = None,
) -> ScopeContext:
    """Create a valid ScopeContext for testing per AGENTS.md Pre-MVP ID Contract.

    Contract Rules (from AGENTS.md):
    - tenant_id: mandatory everywhere
    - trace_id: mandatory for correlation (auto-generated if not provided)
    - invocation_id: mandatory per ID contract (auto-generated if not provided)
    - case_id: optional at HTTP level, required for tool invocations
    - At least one runtime ID required: run_id and/or ingestion_run_id (may co-exist)
    - Identity IDs: user_id (User Request Hop) XOR service_id (S2S Hop)

    Args:
        tenant_id: Tenant identifier (default: test-tenant-id)
        trace_id: W3C trace ID (auto-generated if None)
        invocation_id: Per-hop request ID (auto-generated if None)
        case_id: Business case ID (None for HTTP-level tests, set for tool tests)
        workflow_id: Workflow identifier (optional)
        run_id: Workflow run ID (auto-generated if neither run_id nor ingestion_run_id provided)
        ingestion_run_id: Ingestion run ID (optional, may co-exist with run_id)
        user_id: User identity for User Request Hops (mutually exclusive with service_id)
        service_id: Service identity for S2S Hops (mutually exclusive with user_id)
        collection_id: Collection ID (optional)
        idempotency_key: Idempotency key (optional)

    Returns:
        Valid ScopeContext instance

    Examples:
        # HTTP-level scope (no case_id required)
        scope = make_test_scope_context(user_id="user-123")

        # Tool-level scope (case_id required)
        scope = make_test_scope_context(case_id="case-123", service_id="test-worker")

        # Derive ToolContext
        tool_ctx = scope.to_tool_context()
    """
    # Auto-generate required IDs if not provided
    if trace_id is None:
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"

    if invocation_id is None:
        invocation_id = f"inv-{uuid.uuid4().hex[:8]}"

    # Ensure at least one runtime ID (Pre-MVP ID Contract requirement)
    if run_id is None and ingestion_run_id is None:
        run_id = f"run-{uuid.uuid4().hex[:8]}"

    return ScopeContext(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        case_id=case_id,
        workflow_id=workflow_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
        user_id=user_id,
        service_id=service_id,
        collection_id=collection_id,
        idempotency_key=idempotency_key,
    )


def make_test_tool_context(
    *,
    tenant_id: str = "test-tenant-id",
    trace_id: str | None = None,
    invocation_id: str | None = None,
    case_id: str = "test-case-id",  # Mandatory for tools!
    workflow_id: str | None = None,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    user_id: str | None = None,
    service_id: str = "test-worker",  # Default for S2S
    collection_id: str | None = None,
) -> ToolContext:
    """Create a valid ToolContext for testing per AGENTS.md Pre-MVP ID Contract.

    Tool contexts have stricter requirements than HTTP-level scopes:
    - case_id: MANDATORY (tools require business context)
    - invocation_id: MANDATORY
    - At least one runtime ID: run_id and/or ingestion_run_id
    - Identity: user_id XOR service_id (defaults to service_id for S2S)

    Recommended usage:
        # Create ScopeContext first, then derive ToolContext
        scope = make_test_scope_context(case_id="case-123", service_id="worker")
        tool_ctx = scope.to_tool_context()

        # Or directly create ToolContext for tests
        tool_ctx = make_test_tool_context(case_id="case-123")

    Args:
        tenant_id: Tenant identifier
        trace_id: Trace ID (auto-generated if None)
        invocation_id: Invocation ID (auto-generated if None)
        case_id: Business case ID (MANDATORY for tools, default: test-case-id)
        workflow_id: Workflow identifier (optional)
        run_id: Workflow run ID (auto-generated if neither provided)
        ingestion_run_id: Ingestion run ID (optional)
        user_id: User identity (mutually exclusive with service_id)
        service_id: Service identity (default: test-worker for S2S)
        collection_id: Collection ID (optional)

    Returns:
        Valid ToolContext instance
    """
    # Create ScopeContext first, then derive ToolContext (recommended pattern)
    scope = make_test_scope_context(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        case_id=case_id,  # Required for tools
        workflow_id=workflow_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
        user_id=user_id,
        service_id=service_id,
        collection_id=collection_id,
    )

    # Use canonical conversion method
    return scope.to_tool_context()


__all__ = [
    "GraphTestMixin",
    "make_test_ids",
    "make_test_meta",
    "make_test_scope_context",
    "make_test_tool_context",
]
