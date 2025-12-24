"""AI Core contracts package.

This package contains Pydantic models and TypedDicts for core contracts:
- ScopeContext: Canonical scope context for tool and graph invocations
- AuditMeta: Audit metadata schema for entity persistence
"""

from ai_core.contracts.audit_meta import (
    AuditMeta,
    audit_meta_from_scope,
    build_audit_meta,
)
from ai_core.contracts.scope import (
    CaseId,
    CollectionId,
    IdempotencyKey,
    IngestionRunId,
    InvocationId,
    RunId,
    ScopeContext,
    ServiceId,
    TenantId,
    TenantSchema,
    Timestamp,
    TraceId,
    UserId,
    WorkflowId,
)

__all__ = [
    # Scope
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
    # Audit
    "AuditMeta",
    "audit_meta_from_scope",
    "build_audit_meta",
]
