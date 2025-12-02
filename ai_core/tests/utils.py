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
        meta = make_test_meta(
            tenant_id=tenant_id,
            case_id=case_id,
            workflow_id=workflow_id,
            trace_id=trace_id,
            run_id=run_id,
            ingestion_run_id=ingestion_run_id,
            extra=extra_meta,
        )

        return {
            "input": dict(input_data),
            "meta": {"context": meta},
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

    Enforces the XOR rule between run_id and ingestion_run_id.
    """
    if run_id and ingestion_run_id:
        raise ValueError("Cannot specify both run_id and ingestion_run_id")

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
