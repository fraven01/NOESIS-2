"""AI Core contracts package.

This package contains Pydantic models and TypedDicts for core contracts:
- ScopeContext: Infrastructure scope context (WHO/WHEN)
- BusinessContext: Business domain context (WHAT)
- AuditMeta: Audit metadata schema for entity persistence

BREAKING CHANGE (Option A - Strict Separation):
Business domain IDs (CaseId, CollectionId, WorkflowId, DocumentId, DocumentVersionId)
moved from scope to business module.
"""

from ai_core.contracts.audit_meta import (
    AuditMeta,
    audit_meta_from_scope,
    build_audit_meta,
)
from ai_core.contracts.business import (
    BusinessContext,
    CaseId,
    CollectionId,
    DocumentId,
    DocumentVersionId,
    WorkflowId,
)
from ai_core.contracts.scope import (
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
)

__all__ = [
    # Scope (Infrastructure - WHO/WHEN)
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
    # Business (Domain - WHAT)
    "BusinessContext",
    "CaseId",
    "CollectionId",
    "DocumentId",
    "DocumentVersionId",
    "WorkflowId",
    # Audit
    "AuditMeta",
    "audit_meta_from_scope",
    "build_audit_meta",
]
