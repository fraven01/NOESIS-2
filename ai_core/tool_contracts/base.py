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

New code should use explicit paths:
    context.scope.tenant_id
    context.business.case_id
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, ClassVar, Generic, Literal, Optional, TypeVar, Union
import hashlib
import json

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

    Migration examples:
        OLD: context.tenant_id
        NEW: context.scope.tenant_id

        OLD: context.case_id
        NEW: context.business.case_id

        OLD: params.collection_id or context.collection_id  # RED FLAG!
        NEW: context.business.collection_id  # Only from context
    """

    model_config = ConfigDict(frozen=True)

    _CANONICAL_SCOPE_FIELDS: ClassVar[tuple[str, ...]] = (
        "tenant_id",
        "user_id",
        "service_id",
        "run_id",
        "ingestion_run_id",
        "tenant_schema",
    )
    _CANONICAL_BUSINESS_FIELDS: ClassVar[tuple[str, ...]] = (
        "case_id",
        "collection_id",
        "workflow_id",
        "thread_id",
        "document_id",
        "document_version_id",
    )

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

    def _canonical_payload(self) -> dict[str, dict[str, Any]]:
        scope_payload = {
            key: getattr(self.scope, key, None) for key in self._CANONICAL_SCOPE_FIELDS
        }
        business_payload = {
            key: getattr(self.business, key, None)
            for key in self._CANONICAL_BUSINESS_FIELDS
        }
        return {"scope": scope_payload, "business": business_payload}

    def canonical_json(self) -> str:
        """Return a deterministic JSON string for hashing."""
        payload = self._canonical_payload()
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def tool_context_hash(self) -> str:
        """Stable hash derived from the canonical JSON payload."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8"))
        return digest.hexdigest()


def tool_context_from_scope(
    scope: ScopeContext,
    business: BusinessContext | None = None,
    *,
    now: datetime | None = None,
    locale: str | None = None,
    timeouts_ms: int | None = None,
    budget_tokens: int | None = None,
    safety_mode: str | None = None,
    auth: dict[str, Any] | None = None,
    visibility_override_allowed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ToolContext:
    """Build a ToolContext from ScopeContext and BusinessContext.

    BREAKING CHANGE (Option A):
    New signature requires BusinessContext as separate parameter.

    BREAKING CHANGE (Phase 4):
    Removed **overrides in favor of explicit parameters. This prevents
    accidental override of scope/business and provides better type safety.

    Args:
        scope: Request correlation scope (WHO, WHEN)
        business: Optional business domain context (WHAT). Defaults to empty.
        now: Override timestamp (for testing). Ignored (scope.timestamp used).
        locale: Locale string (e.g., "de-DE")
        timeouts_ms: Timeout in milliseconds
        budget_tokens: Token budget for LLM calls
        safety_mode: Safety mode string
        auth: Authentication metadata
        visibility_override_allowed: Whether visibility overrides are allowed
        metadata: Additional runtime metadata

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

    return ToolContext(
        scope=scope,
        business=business,
        locale=locale,
        timeouts_ms=timeouts_ms,
        budget_tokens=budget_tokens,
        safety_mode=safety_mode,
        auth=auth,
        visibility_override_allowed=visibility_override_allowed,
        metadata=metadata or {},
    )


def tool_context_from_meta(meta: Mapping[str, Any]) -> ToolContext:
    """Parse a ToolContext from graph metadata.

    Prefers a prebuilt ``tool_context`` entry, with a fallback to scope/business
    metadata for legacy call sites.
    """

    tool_context_data = meta.get("tool_context")
    if isinstance(tool_context_data, ToolContext):
        return tool_context_data
    if isinstance(tool_context_data, Mapping):
        return ToolContext.model_validate(tool_context_data)

    scope_data = meta.get("scope_context")
    if scope_data is None:
        raise ValueError("tool_context or scope_context is required in meta")

    business_keys = {
        "case_id",
        "collection_id",
        "workflow_id",
        "thread_id",
        "document_id",
        "document_version_id",
    }

    if isinstance(scope_data, ScopeContext):
        scope = scope_data
        scope_payload: dict[str, Any] = {}
        scope_source: Mapping[str, Any] = {}
    elif isinstance(scope_data, Mapping):
        scope_source = scope_data
        scope_payload = {k: v for k, v in scope_data.items() if k not in business_keys}
        scope = ScopeContext.model_validate(scope_payload)
    else:
        raise TypeError("scope_context must be a mapping or ScopeContext")

    business_data = meta.get("business_context")
    if isinstance(business_data, BusinessContext):
        business = business_data
    else:
        business_payload: dict[str, Any] = {}
        if isinstance(business_data, Mapping):
            business_payload.update(business_data)
        if scope_source:
            for key in business_keys:
                if key in scope_source and key not in business_payload:
                    business_payload[key] = scope_source[key]
        business = BusinessContext.model_validate(business_payload)

    metadata = meta.get("context_metadata")
    metadata_payload = dict(metadata) if isinstance(metadata, Mapping) else {}
    initiated_by_user_id = meta.get("initiated_by_user_id")
    if initiated_by_user_id is not None:
        metadata_payload.setdefault("initiated_by_user_id", initiated_by_user_id)
    key_alias = meta.get("key_alias")
    if key_alias is not None:
        metadata_payload.setdefault("key_alias", key_alias)

    if not metadata_payload:
        return tool_context_from_scope(scope, business)
    return tool_context_from_scope(scope, business, metadata=metadata_payload)


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
    "tool_context_from_meta",
    "ToolErrorMeta",
    "ToolErrorDetail",
    "ToolError",
    "ToolResultMeta",
    "ToolResult",
    "ToolOutput",
]
