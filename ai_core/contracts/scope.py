"""Scope context contracts for tool and graph invocations.

Context identifiers are aligned across APIs, graphs and tools:
- ``tenant_id`` identifies the organizational tenant and drives schema/permission selection; it is mandatory everywhere.
- ``case_id`` ties executions to a business case within a tenant, bundling documents and decisions for its full lifetime.
- ``workflow_id`` labels a logical workflow inside a case (e.g., intake, assessment); repeated executions of the same workflow reuse
  the same ID provided by the caller or dispatcher.
- ``run_id`` marks a single LangGraph execution of a workflow; every execution gets a fresh, non-semantic value that belongs to
  exactly one ``workflow_id`` and ``case_id``.

Relationships: one tenant has many cases → each case can contain many workflows → each workflow can have many runs. Tools require
``tenant_id``, ``trace_id``, ``invocation_id`` and exactly one runtime identifier (``run_id`` or ``ingestion_run_id``). Graphs set
``case_id`` and ``workflow_id`` as soon as the business context is known, while ``run_id`` stays purely technical and is generated
per execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from ai_core.tool_contracts.base import ToolContext

TenantId = str
TraceId = str
InvocationId = str
CaseId = str
TenantSchema = str
WorkflowId = str
RunId = str
IngestionRunId = str
IdempotencyKey = str
CollectionId = UUID | None
Timestamp = datetime


class ScopeContext(BaseModel):
    """Canonical scope context containing mandatory correlation identifiers.

    The collection_id field represents the "Aktenschrank" (file cabinet) context,
    enabling multi-collection search and scoped operations.
    """

    tenant_id: TenantId
    trace_id: TraceId
    invocation_id: InvocationId
    run_id: RunId | None = None
    ingestion_run_id: IngestionRunId | None = None
    case_id: CaseId | None = None
    tenant_schema: TenantSchema | None = None
    workflow_id: WorkflowId | None = None
    idempotency_key: IdempotencyKey | None = None
    collection_id: CollectionId = Field(
        default=None,
        description="Collection UUID for scoped operations ('Aktenschrank')",
    )
    timestamp: Timestamp = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_run_scope(self) -> "ScopeContext":
        """Ensure exactly one runtime identifier is provided."""

        has_run_id = bool(self.run_id)
        has_ingestion_run_id = bool(self.ingestion_run_id)

        if has_run_id == has_ingestion_run_id:
            raise ValueError(
                "Exactly one of run_id or ingestion_run_id must be provided"
            )

        return self

    def to_tool_context(self, **overrides: object) -> "ToolContext":
        """Project this scope into a ``ToolContext`` with optional overrides."""

        from ai_core.tool_contracts.base import tool_context_from_scope

        return tool_context_from_scope(self, **overrides)


__all__ = [
    "CaseId",
    "CollectionId",
    "IdempotencyKey",
    "IngestionRunId",
    "InvocationId",
    "RunId",
    "ScopeContext",
    "TenantId",
    "TenantSchema",
    "Timestamp",
    "TraceId",
    "WorkflowId",
]
