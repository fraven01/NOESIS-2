"""Pydantic models defining the shared tool contract envelopes.

BREAKING CHANGE (Option A - Strict Separation):
ToolContext now uses COMPOSITION instead of flat structure.

OLD (v1):
    ToolContext(
        tenant_id="t",
        trace_id="tr",
        case_id="c",
        collection_id="col",
        locale="de-DE"
    )

NEW (v2):
    scope = ScopeContext(tenant_id="t", trace_id="tr", ...)
    business = BusinessContext(case_id="c", collection_id="col")
    ToolContext(scope=scope, business=business, locale="de-DE")

Backward compatibility via @property accessors (DEPRECATED, will be removed):
    context.tenant_id  # → context.scope.tenant_id (deprecated)
    context.case_id    # → context.business.case_id (deprecated)

New code should use explicit paths:
    context.scope.tenant_id  # ✅
    context.business.case_id  # ✅
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, ClassVar, Generic, Literal, Optional, TypeVar, Union

try:  # pragma: no cover - typing backport
    from typing import TypeAliasType
except ImportError:  # pragma: no cover - fallback for <3.12
    from typing_extensions import TypeAliasType

from pydantic import BaseModel, ConfigDict, Field

from ai_core.contracts.scope import ScopeContext
from ai_core.contracts.business import BusinessContext
from ai_core.tools.errors import ToolErrorType

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]

IT = TypeVar("IT", bound=BaseModel)
OT = TypeVar("OT", bound=BaseModel)


class ToolContext(BaseModel):
    """Complete tool invocation context with separated concerns.

    BREAKING CHANGE (Option A):
    Structure changed from flat to compositional:
    - scope: ScopeContext (WHO, WHEN) - request correlation
    - business: BusinessContext (WHAT) - domain context
    - Plus: Runtime metadata (HOW) - locale, budgets, permissions

    Golden Rule (Operationalized):
    - Tool-Inputs contain only functional parameters
    - Context contains Scope, Business, and Runtime Permissions
    - Tool-Run functions read identifiers exclusively from context, not params

    Backward Compatibility:
    Properties like context.tenant_id, context.case_id still work (DEPRECATED).
    They delegate to context.scope.X or context.business.X.
    New code should use explicit paths.

    Migration examples:
        OLD: context.tenant_id
        NEW: context.scope.tenant_id

        OLD: context.case_id
        NEW: context.business.case_id

        OLD: params.collection_id or context.collection_id  # RED FLAG!
        NEW: context.business.collection_id  # ✅ Only from context
    """

    model_config = ConfigDict(frozen=True)

    # === Compositional Structure (NEW in v2) ===
    scope: ScopeContext = Field(
        description="Request correlation scope (tenant, trace, runtime IDs)"
    )

    business: BusinessContext = Field(
        description="Business domain context (case, document, collection IDs)"
    )

    # === Runtime Metadata ===
    locale: Optional[str] = None
    timeouts_ms: Optional[PositiveInt] = None
    budget_tokens: Optional[int] = None
    safety_mode: Optional[str] = None
    auth: Optional[dict[str, Any]] = None
    visibility_override_allowed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    # === Backward Compatibility Properties (DEPRECATED) ===
    # These will be removed in a future version after migration is complete.
    # New code MUST use context.scope.X or context.business.X instead.

    @property
    def tenant_id(self) -> str:
        """DEPRECATED: Use context.scope.tenant_id instead."""
        return self.scope.tenant_id

    @property
    def trace_id(self) -> str:
        """DEPRECATED: Use context.scope.trace_id instead."""
        return self.scope.trace_id

    @property
    def invocation_id(self) -> str:
        """DEPRECATED: Use context.scope.invocation_id instead."""
        return self.scope.invocation_id

    @property
    def user_id(self) -> str | None:
        """DEPRECATED: Use context.scope.user_id instead."""
        return self.scope.user_id

    @property
    def service_id(self) -> str | None:
        """DEPRECATED: Use context.scope.service_id instead."""
        return self.scope.service_id

    @property
    def run_id(self) -> str | None:
        """DEPRECATED: Use context.scope.run_id instead."""
        return self.scope.run_id

    @property
    def ingestion_run_id(self) -> str | None:
        """DEPRECATED: Use context.scope.ingestion_run_id instead."""
        return self.scope.ingestion_run_id

    @property
    def tenant_schema(self) -> str | None:
        """DEPRECATED: Use context.scope.tenant_schema instead."""
        return self.scope.tenant_schema

    @property
    def idempotency_key(self) -> str | None:
        """DEPRECATED: Use context.scope.idempotency_key instead."""
        return self.scope.idempotency_key

    @property
    def now_iso(self) -> datetime:
        """DEPRECATED: Use context.scope.timestamp instead."""
        return self.scope.timestamp

    @property
    def case_id(self) -> str | None:
        """DEPRECATED: Use context.business.case_id instead."""
        return self.business.case_id

    @property
    def collection_id(self) -> str | None:
        """DEPRECATED: Use context.business.collection_id instead."""
        return self.business.collection_id

    @property
    def workflow_id(self) -> str | None:
        """DEPRECATED: Use context.business.workflow_id instead."""
        return self.business.workflow_id

    @property
    def document_id(self) -> str | None:
        """DEPRECATED: Use context.business.document_id instead."""
        return self.business.document_id

    @property
    def document_version_id(self) -> str | None:
        """DEPRECATED: Use context.business.document_version_id instead."""
        return self.business.document_version_id


def tool_context_from_scope(
    scope: ScopeContext,
    business: BusinessContext | None = None,
    *,
    now: datetime | None = None,
    **overrides: Any,
) -> ToolContext:
    """Build a ToolContext from ScopeContext and BusinessContext.

    BREAKING CHANGE (Option A):
    New signature requires BusinessContext as separate parameter.

    Args:
        scope: Request correlation scope (WHO, WHEN)
        business: Optional business domain context (WHAT). Defaults to empty.
        now: Override timestamp (for testing). Ignored (scope.timestamp used).
        **overrides: Additional ToolContext fields (locale, budget_tokens, etc.)

    Returns:
        ToolContext with compositional structure

    Examples:
        # Minimal (no business context):
        scope = ScopeContext(tenant_id="t", trace_id="tr", ...)
        context = tool_context_from_scope(scope)

        # With business context:
        business = BusinessContext(case_id="c", collection_id="col")
        context = tool_context_from_scope(scope, business, locale="de-DE")

        # Via ScopeContext helper:
        context = scope.to_tool_context(business=business, budget_tokens=512)
    """
    if business is None:
        business = BusinessContext()  # Empty business context

    payload: dict[str, Any] = {
        "scope": scope,
        "business": business,
    }

    # Add overrides (locale, timeouts, etc.)
    payload.update(overrides)

    return ToolContext(**payload)


class ToolErrorMeta(BaseModel):
    """Meta information for tool errors."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "took_ms": 250,
            }
        ]
    }

    took_ms: NonNegativeInt


class ToolErrorDetail(BaseModel):
    """Structured error payload returned by tools."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "type": "RATE_LIMIT",
                "message": "Quota exceeded",
                "code": "rate_limit",
                "details": {"limit": 100, "interval": "1m"},
                "retry_after_ms": 1500,
                "upstream_status": 429,
                "endpoint": "POST /tool",
            }
        ]
    }

    type: ToolErrorType
    message: str
    code: Optional[str] = None
    cause: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    retry_after_ms: Optional[int] = None
    upstream_status: Optional[int] = None
    endpoint: Optional[str] = None
    attempt: Optional[int] = None


class ToolError(BaseModel, Generic[IT]):
    """Envelope describing a failed tool invocation."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "status": "error",
                "input": {"query": "hello"},
                "error": {
                    "type": "VALIDATION",
                    "message": "Missing required field",
                    "code": "missing_field",
                },
                "meta": {"took_ms": 10},
            }
        ]
    }

    status: Literal["error"] = "error"
    input: IT
    error: ToolErrorDetail
    meta: ToolErrorMeta


class ToolResultMeta(BaseModel):
    """Meta data attached to successful tool responses."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "took_ms": 180,
                "source_counts": {"documents": 3},
                "routing": {"embedding_profile": "default"},
                "cache_hit": False,
                "token_usage": {"prompt": 1200, "completion": 800},
            }
        ]
    }

    took_ms: NonNegativeInt
    source_counts: Optional[dict[str, int]] = None
    routing: Optional[dict[str, Any]] = None
    cache_hit: Optional[bool] = None
    token_usage: Optional[dict[str, int]] = None


class ToolResult(BaseModel, Generic[IT, OT]):
    """Envelope describing a successful tool invocation."""

    model_config = ConfigDict(frozen=True)
    json_schema_extra: ClassVar[dict[str, Any]] = {
        "examples": [
            {
                "status": "ok",
                "input": {"query": "hello"},
                "data": {"answer": "Hi there!"},
                "meta": {
                    "took_ms": 42,
                    "token_usage": {"prompt": 12, "completion": 20},
                },
            }
        ]
    }

    status: Literal["ok"] = "ok"
    input: IT
    data: OT
    meta: ToolResultMeta


ToolOutput = TypeAliasType(
    "ToolOutput",
    Annotated[
        Union[ToolResult[IT, OT], ToolError[IT]],
        Field(discriminator="status"),
    ],
    type_params=(IT, OT),
)

__all__ = [
    "NonNegativeInt",
    "PositiveInt",
    "ToolContext",
    "tool_context_from_scope",
    "ToolErrorMeta",
    "ToolErrorDetail",
    "ToolError",
    "ToolResultMeta",
    "ToolResult",
    "ToolOutput",
]
