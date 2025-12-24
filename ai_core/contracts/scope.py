"""Scope context contracts for tool and graph invocations.

Context identifiers are aligned across APIs, graphs and tools:
- ``tenant_id`` identifies the organizational tenant and drives schema/permission selection; it is mandatory everywhere.
- ``case_id`` ties executions to a business case within a tenant, bundling documents and decisions for its full lifetime.
- ``workflow_id`` labels a logical workflow inside a case (e.g., intake, assessment); repeated executions of the same workflow reuse
  the same ID provided by the caller or dispatcher.
- ``run_id`` marks a single LangGraph execution of a workflow; every execution gets a fresh, non-semantic value that belongs to
  exactly one ``workflow_id`` and ``case_id``.

Relationships: one tenant has many cases → each case can contain many workflows → each workflow can have many runs. Tools require
``tenant_id``, ``trace_id``, ``invocation_id`` and at least one runtime identifier (``run_id`` and/or ``ingestion_run_id``). Graphs set
``case_id`` and ``workflow_id`` as soon as the business context is known, while ``run_id`` stays purely technical and is generated
per execution.

Identity rules (strict):
- User Request Hop: ``user_id`` REQUIRED (when auth present), ``service_id`` ABSENT.
- S2S Hop (Celery/Graph): ``service_id`` REQUIRED, ``user_id`` ABSENT.
- ``initiated_by_user_id`` is for audit_meta only (causal tracking), never a principal in ScopeContext.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

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
CollectionId = str | None
Timestamp = datetime

# Identity IDs (Pre-MVP ID Contract)
UserId = str | None  # UUIDv7 string, required for User Request Hops
ServiceId = str | None  # Required for S2S Hops (e.g., "celery-ingestion-worker")


class ScopeContext(BaseModel):
    """Canonical scope context containing mandatory correlation identifiers.

    The collection_id field represents the "Aktenschrank" (file cabinet) context,
    enabling multi-collection search and scoped operations.

    Identity rules (Pre-MVP ID Contract):
    - User Request Hop: user_id REQUIRED (when auth present), service_id ABSENT.
    - S2S Hop (Celery/Graph): service_id REQUIRED, user_id ABSENT.
    - Both user_id and service_id being set is invalid.
    - Both being absent is only valid for public/unauthenticated endpoints.

    BREAKING CHANGE: collection_id is now str (UUID-string, previously UUID).
    BREAKING CHANGE: run_id and ingestion_run_id may co-exist (previously XOR).

    Note: case_id is optional at the ScopeContext level (for HTTP requests).
    It becomes mandatory when graphs/tools set business context. The ToolContext
    enforces case_id requirements for AI operations.
    """

    # Mandatory correlation IDs
    tenant_id: TenantId
    trace_id: TraceId
    invocation_id: InvocationId

    # Business context (optional at request level, required for tool invocations)
    case_id: CaseId | None = None

    # Identity IDs (mutually exclusive per hop type)
    user_id: UserId = Field(
        default=None,
        description="User identity for User Request Hops. Must be absent for S2S.",
    )
    service_id: ServiceId = Field(
        default=None,
        description="Service identity for S2S Hops (e.g., 'celery-ingestion-worker'). Must be absent for User Requests.",
    )

    # Runtime IDs (may co-exist when workflow triggers ingestion)
    run_id: RunId | None = None
    ingestion_run_id: IngestionRunId | None = None

    # Optional context
    tenant_schema: TenantSchema | None = None
    workflow_id: WorkflowId | None = None
    idempotency_key: IdempotencyKey | None = None
    collection_id: CollectionId = Field(
        default=None,
        description="Collection ID (UUID-string) for scoped operations ('Aktenschrank')",
    )
    timestamp: Timestamp = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_run_scope(self) -> "ScopeContext":
        """Ensure at least one runtime identifier is provided."""
        has_run_id = bool(self.run_id)
        has_ingestion_run_id = bool(self.ingestion_run_id)

        if not has_run_id and not has_ingestion_run_id:
            raise ValueError(
                "At least one of run_id or ingestion_run_id must be provided"
            )

        return self

    @model_validator(mode="after")
    def validate_identity(self) -> "ScopeContext":
        """Ensure user_id and service_id are mutually exclusive.

        Identity rules:
        - User Request Hop: user_id set, service_id absent
        - S2S Hop: service_id set, user_id absent
        - Both set: Invalid (ambiguous principal)
        - Both absent: Valid only for public endpoints (caller must enforce)
        """
        if self.user_id and self.service_id:
            raise ValueError(
                "user_id and service_id are mutually exclusive. "
                "User Request Hops have user_id, S2S Hops have service_id."
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
    "ServiceId",
    "TenantId",
    "TenantSchema",
    "Timestamp",
    "TraceId",
    "UserId",
    "WorkflowId",
]
