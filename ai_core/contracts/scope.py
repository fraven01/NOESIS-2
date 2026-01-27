"""Scope context contracts for tool and graph invocations.

BREAKING CHANGE (Option A - Strict Separation):
Business domain identifiers (case_id, collection_id, workflow_id, document_id)
have been REMOVED from ScopeContext and moved to BusinessContext.

ScopeContext now contains ONLY request correlation and infrastructure identifiers:
- ``tenant_id`` identifies the organizational tenant and drives schema/permission selection
- ``trace_id`` enables distributed tracing across service boundaries
- ``invocation_id`` uniquely identifies a single invocation within a trace
- ``run_id`` marks a single LangGraph execution (one workflow run)
- ``ingestion_run_id`` marks a document ingestion run

Separation rationale:
- ScopeContext = Request Correlation (WHO, WHEN) - infrastructure concerns
- BusinessContext = Domain Context (WHAT) - business concerns
- ToolContext = Runtime Metadata (HOW) - execution concerns

Identity rules (strict):
- User Request Hop: ``user_id`` REQUIRED (when auth present), ``service_id`` ABSENT.
- S2S Hop (Celery/Graph): ``service_id`` REQUIRED, ``user_id`` ABSENT.
- ``initiated_by_user_id`` is for audit_meta only (causal tracking), never a principal in ScopeContext.

See: OPTION_A_IMPLEMENTATION_PLAN.md, OPTION_A_SOURCE_CODE_ANALYSIS.md
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from ai_core.contracts.business import BusinessContext
    from ai_core.tool_contracts.base import ToolContext

# Type aliases for ScopeContext fields (infrastructure/correlation IDs only)
TenantId = str
TraceId = str
InvocationId = str
TenantSchema = str
RunId = str
IngestionRunId = str
IdempotencyKey = str
Timestamp = datetime

# Identity IDs (Pre-MVP ID Contract)
UserId = str | None  # UUIDv7 string, required for User Request Hops
ServiceId = str | None  # Required for S2S Hops (e.g., "celery-ingestion-worker")


class ExecutionScope(str, Enum):
    CASE = "CASE"
    TENANT = "TENANT"
    SYSTEM = "SYSTEM"


class ScopeContext(BaseModel):
    """Request correlation scope containing infrastructure identifiers.

    BREAKING CHANGE (Option A):
    Business domain IDs (case_id, collection_id, workflow_id) REMOVED.
    These now live in BusinessContext (ai_core.contracts.business).

    ScopeContext represents the infrastructure-level request correlation:
    - WHO: tenant_id, user_id/service_id (identity)
    - WHEN: timestamp, trace_id, invocation_id
    - RUNTIME: run_id, ingestion_run_id

    Identity rules (Pre-MVP ID Contract):
    - User Request Hop: user_id REQUIRED (when auth present), service_id ABSENT.
    - S2S Hop (Celery/Graph): service_id REQUIRED, user_id ABSENT.
    - Both user_id and service_id being set is invalid.
    - Both being absent is only valid for public/unauthenticated endpoints.

    Runtime IDs:
    - run_id and ingestion_run_id may co-exist (when workflow triggers ingestion).
    - At least ONE runtime ID is required (enforced by validator).

    Migration from old ScopeContext:
    - OLD: ScopeContext(tenant_id="t", case_id="c", collection_id="col")
    - NEW: scope = ScopeContext(tenant_id="t")
           business = BusinessContext(case_id="c", collection_id="col")
           context = ToolContext(scope=scope, business=business)
    """

    # Mandatory correlation IDs
    tenant_id: TenantId
    trace_id: TraceId
    invocation_id: InvocationId

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

    # Optional technical context
    tenant_schema: TenantSchema | None = None
    idempotency_key: IdempotencyKey | None = None
    timestamp: Timestamp = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def forbid_business_ids(cls, data: object) -> object:
        """Reject business identifiers passed into ScopeContext."""
        if isinstance(data, ScopeContext):
            return data
        if isinstance(data, Mapping):
            forbidden = {
                "case_id",
                "collection_id",
                "workflow_id",
                "document_id",
                "document_version_id",
            }
            present = sorted(key for key in forbidden if key in data)
            if present:
                raise ValueError(
                    "ScopeContext cannot include business IDs. "
                    f"Move {', '.join(present)} to BusinessContext."
                )
        return data

    @model_validator(mode="before")
    @classmethod
    def normalize_user_id(cls, data: object) -> object:
        """Coerce and validate user_id as a UUID string when present."""
        if isinstance(data, ScopeContext):
            return data
        if isinstance(data, Mapping):
            if "user_id" not in data:
                return data
            raw_user_id = data.get("user_id")
            if raw_user_id in {None, ""}:
                if raw_user_id is None:
                    return data
                payload = dict(data)
                payload["user_id"] = None
                return payload
            try:
                parsed = UUID(str(raw_user_id))
            except (TypeError, ValueError) as exc:
                raise ValueError("user_id must be a UUID string") from exc
            payload = dict(data)
            payload["user_id"] = str(parsed)
            return payload
        return data

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

    def to_tool_context(
        self,
        business: "BusinessContext | None" = None,
        *,
        now: "datetime | None" = None,
        locale: str | None = None,
        timeouts_ms: int | None = None,
        budget_tokens: int | None = None,
        safety_mode: str | None = None,
        auth: dict[str, object] | None = None,
        visibility_override_allowed: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> "ToolContext":
        """Project this scope into a ToolContext with optional BusinessContext.

        BREAKING CHANGE (Phase 4):
        Removed **overrides in favor of explicit parameters for better type safety.

        Args:
            business: Optional BusinessContext (case_id, collection_id, etc.)
            now: Override timestamp (for testing). Ignored (scope.timestamp used).
            locale: Locale string (e.g., "de-DE")
            timeouts_ms: Timeout in milliseconds
            budget_tokens: Token budget for LLM calls
            safety_mode: Safety mode string
            auth: Authentication metadata
            visibility_override_allowed: Whether visibility overrides are allowed
            metadata: Additional runtime metadata

        Returns:
            ToolContext with compositional structure (scope + business + metadata)

        Example:
            from ai_core.contracts.business import BusinessContext

            scope = ScopeContext(tenant_id="t", trace_id="tr", ...)
            business = BusinessContext(case_id="c", collection_id="col")
            context = scope.to_tool_context(business=business, locale="de-DE")
        """
        from ai_core.tool_contracts.base import tool_context_from_scope

        return tool_context_from_scope(
            self,
            business,
            now=now,
            locale=locale,
            timeouts_ms=timeouts_ms,
            budget_tokens=budget_tokens,
            safety_mode=safety_mode,
            auth=auth,
            visibility_override_allowed=visibility_override_allowed,
            metadata=metadata,
        )


__all__ = [
    "ExecutionScope",
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
]
