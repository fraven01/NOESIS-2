"""Scope context contracts for tool and graph invocations."""

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
WorkflowId = str
RunId = str
IngestionRunId = str
IdempotencyKey = str
Timestamp = datetime


class ScopeContext(BaseModel):
    """Canonical scope context containing mandatory correlation identifiers."""

    tenant_id: TenantId
    trace_id: TraceId
    invocation_id: InvocationId
    run_id: RunId | None = None
    ingestion_run_id: IngestionRunId | None = None
    case_id: CaseId | None = None
    workflow_id: WorkflowId | None = None
    idempotency_key: IdempotencyKey | None = None
    timestamp: Timestamp = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_run_scope(self) -> "ScopeContext":
        """Ensure exactly one runtime identifier is provided."""

        has_run_id = bool(self.run_id)
        has_ingestion_run_id = bool(self.ingestion_run_id)

        if has_run_id == has_ingestion_run_id:
            raise ValueError("Exactly one of run_id or ingestion_run_id must be provided")

        return self

    def to_tool_context(self, **overrides: object) -> "ToolContext":
        """Project this scope into a ``ToolContext`` with optional overrides."""

        from ai_core.tool_contracts.base import tool_context_from_scope

        return tool_context_from_scope(self, **overrides)


__all__ = [
    "CaseId",
    "IdempotencyKey",
    "IngestionRunId",
    "InvocationId",
    "RunId",
    "ScopeContext",
    "TenantId",
    "Timestamp",
    "TraceId",
    "WorkflowId",
]
