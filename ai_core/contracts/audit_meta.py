"""Audit metadata schema for entity persistence (Pre-MVP ID Contract).

This module defines the audit_meta JSON structure stored on core entities.
Use this schema for consistent audit trail across all persisted objects.

Key semantics:
- created_by_user_id: Who OWNS the entity (business creator). SET ONCE at creation, immutable.
- initiated_by_user_id: Who TRIGGERED the flow (root cause for S2S chains).
- last_hop_service_id: Which service LAST WROTE this entity (persistence scope).

IMPORTANT: ScopeContext.service_id != audit_meta.last_hop_service_id
- ScopeContext.service_id = who executes THIS hop (request scope)
- audit_meta.last_hop_service_id = who LAST WROTE this entity (persistence scope)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ai_core.contracts.scope import ScopeContext


class AuditMeta(BaseModel):
    """Audit metadata for entity persistence.

    Stored as JSON on core entities. Provides full traceability for
    debugging, compliance, and observability.
    """

    model_config = ConfigDict(frozen=True)

    # Mandatory observability IDs
    trace_id: str = Field(
        ...,
        description="W3C-compatible trace ID for end-to-end correlation across hops.",
    )
    invocation_id: str = Field(
        ...,
        description="Per-hop request/job identifier (UUIDv7 hex).",
    )

    # Runtime execution IDs (optional, at least one should be present)
    workflow_run_id: Optional[str] = Field(
        default=None,
        description="Workflow/graph execution identifier (UUIDv7).",
    )
    ingestion_run_id: Optional[str] = Field(
        default=None,
        description="Ingestion execution identifier (UUIDv7).",
    )

    # Idempotency
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Request-level determinism key.",
    )

    # User identity (immutable after creation)
    created_by_user_id: Optional[str] = Field(
        default=None,
        description="Who OWNS the entity (business creator). SET ONCE at creation, immutable.",
    )

    # Causal tracking for S2S flows
    initiated_by_user_id: Optional[str] = Field(
        default=None,
        description="Who TRIGGERED the flow (root cause). For S2S chains only.",
    )

    # Service identity for persistence
    last_hop_service_id: Optional[str] = Field(
        default=None,
        description="Which service LAST WROTE this entity. NOT the same as ScopeContext.service_id.",
    )

    # Timestamp of last modification
    last_modified_at: Optional[datetime] = Field(
        default=None,
        description="When this entity was last modified (UTC).",
    )


def build_audit_meta(
    *,
    trace_id: str,
    invocation_id: str,
    workflow_run_id: Optional[str] = None,
    ingestion_run_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    created_by_user_id: Optional[str] = None,
    initiated_by_user_id: Optional[str] = None,
    last_hop_service_id: Optional[str] = None,
    last_modified_at: Optional[datetime] = None,
) -> AuditMeta:
    """Build an AuditMeta instance with validated fields."""
    return AuditMeta(
        trace_id=trace_id,
        invocation_id=invocation_id,
        workflow_run_id=workflow_run_id,
        ingestion_run_id=ingestion_run_id,
        idempotency_key=idempotency_key,
        created_by_user_id=created_by_user_id,
        initiated_by_user_id=initiated_by_user_id,
        last_hop_service_id=last_hop_service_id,
        last_modified_at=last_modified_at,
    )


def audit_meta_from_scope(
    scope: "ScopeContext",
    *,
    created_by_user_id: Optional[str] = None,
    initiated_by_user_id: Optional[str] = None,
    last_modified_at: Optional[datetime] = None,
) -> AuditMeta:
    """Build AuditMeta from a ScopeContext.

    The service_id from ScopeContext becomes last_hop_service_id in audit_meta.
    For User Request Hops, user_id can be passed as created_by_user_id.

    Args:
        scope: The ScopeContext for the current hop.
        created_by_user_id: Override for entity owner (defaults to scope.user_id for creates).
        initiated_by_user_id: Who triggered the S2S flow (causal tracking).
        last_modified_at: When the entity was modified (defaults to scope.timestamp).
    """
    from ai_core.contracts.scope import ScopeContext

    if not isinstance(scope, ScopeContext):
        raise TypeError(f"Expected ScopeContext, got {type(scope).__name__}")

    return AuditMeta(
        trace_id=scope.trace_id,
        invocation_id=scope.invocation_id,
        workflow_run_id=scope.run_id,  # run_id maps to workflow_run_id in audit_meta
        ingestion_run_id=scope.ingestion_run_id,
        idempotency_key=scope.idempotency_key,
        created_by_user_id=created_by_user_id or scope.user_id,
        initiated_by_user_id=initiated_by_user_id,
        last_hop_service_id=scope.service_id,  # ScopeContext.service_id → audit_meta.last_hop_service_id
        last_modified_at=last_modified_at or scope.timestamp,
    )


__all__ = [
    "AuditMeta",
    "audit_meta_from_scope",
    "build_audit_meta",
]
