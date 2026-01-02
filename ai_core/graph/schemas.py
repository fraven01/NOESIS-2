"""Helpers for normalising request metadata and merging graph state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping
from uuid import uuid4

from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    META_CASE_ID_KEY,
    META_IDEMPOTENCY_KEY,
    META_KEY_ALIAS_KEY,
    META_COLLECTION_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    META_TRACE_ID_KEY,
    META_WORKFLOW_ID_KEY,
    META_DOCUMENT_ID_KEY,
    META_DOCUMENT_VERSION_ID_KEY,
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_TRACE_ID_HEADER,
    X_WORKFLOW_ID_HEADER,
    X_DOCUMENT_ID_HEADER,
    X_DOCUMENT_VERSION_ID_HEADER,
)

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.infra.rate_limit import get_quota


def _coalesce(request: Any, header: str, meta_key: str) -> str | None:
    headers: Mapping[str, str] = getattr(request, "headers", {}) or {}
    meta: MutableMapping[str, Any] = getattr(request, "META", {}) or {}
    value = headers.get(header)
    if value is None:
        value = meta.get(meta_key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _resolve_graph_name(request: Any) -> str:
    explicit = getattr(request, "graph_name", None)
    if isinstance(explicit, str) and explicit:
        return explicit

    resolver = getattr(request, "resolver_match", None)
    if resolver is not None:
        url_name = getattr(resolver, "url_name", None)
        if isinstance(url_name, str) and url_name:
            return url_name
        view_name = getattr(resolver, "view_name", None)
        if isinstance(view_name, str) and view_name:
            return view_name

    path = getattr(request, "path", None)
    if isinstance(path, str) and path:
        candidate = path.rstrip("/").split("/")[-1]
        if candidate:
            return candidate
    raise ValueError("graph name could not be determined from request")


def _normalize_header_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def _resolve_tenant_schema(request: Any) -> str | None:
    tenant_schema = _coalesce(request, X_TENANT_SCHEMA_HEADER, META_TENANT_SCHEMA_KEY)
    if tenant_schema:
        return tenant_schema

    from customers.tenant_context import TenantContext

    tenant = TenantContext.from_request(request, require=False)
    if tenant:
        return tenant.schema_name
    return None


def _build_scope_context(request: Any) -> ScopeContext:
    """Build a canonical ScopeContext from the request or attached scope."""
    existing_scope = getattr(request, "scope_context", None)
    if isinstance(existing_scope, ScopeContext):
        return existing_scope

    # If request is an HttpRequest, use the normalizer
    from django.http import HttpRequest

    if isinstance(request, HttpRequest):
        from ai_core.ids import normalize_request

        return normalize_request(request)

    # Fallback for non-HttpRequest objects (e.g. dicts or mocks)
    # This logic mimics the normalizer but for generic objects
    tenant_id_raw = _coalesce(request, X_TENANT_ID_HEADER, META_TENANT_ID_KEY)
    trace_id = _coalesce(request, X_TRACE_ID_HEADER, META_TRACE_ID_KEY) or uuid4().hex
    idempotency_key = _coalesce(request, IDEMPOTENCY_KEY_HEADER, META_IDEMPOTENCY_KEY)
    tenant_schema = _resolve_tenant_schema(request)

    headers: Mapping[str, str] = getattr(request, "headers", {}) or {}
    meta: MutableMapping[str, Any] = getattr(request, "META", {}) or {}
    invocation_id = (
        _normalize_header_value(
            getattr(request, "invocation_id", None)
            or headers.get("X-Invocation-ID")
            or meta.get("HTTP_X_INVOCATION_ID")
        )
        or uuid4().hex
    )

    run_id = _normalize_header_value(
        getattr(request, "run_id", None)
        or headers.get("X-Run-ID")
        or meta.get("HTTP_X_RUN_ID")
    )
    ingestion_run_id = _normalize_header_value(
        getattr(request, "ingestion_run_id", None)
        or headers.get("X-Ingestion-Run-ID")
        or meta.get("HTTP_X_INGESTION_RUN_ID")
    )

    if not tenant_id_raw:
        raise ValueError("missing required meta keys: tenant_id")

    tenant_id = str(tenant_id_raw).strip()
    if not tenant_id:
        raise ValueError("missing required meta keys: tenant_id")

    if not ingestion_run_id and not run_id:
        run_id = uuid4().hex

    # BREAKING CHANGE (Option A - Pre-MVP ID Contract):
    # Extract identity IDs (user_id for User Request Hops, service_id for S2S Hops)
    user_id: str | None = None
    service_id: str | None = None

    # Try to extract user_id from Django auth if available
    user = getattr(request, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        user_pk = getattr(user, "pk", None)
        if user_pk is not None:
            user_id = str(user_pk)

    # Try to extract service_id from headers/META (for S2S Hops)
    if not user_id:
        service_id = _normalize_header_value(
            getattr(request, "service_id", None)
            or headers.get("X-Service-ID")
            or meta.get("HTTP_X_SERVICE_ID")
        )

    # BREAKING CHANGE (Option A): Business IDs removed from ScopeContext
    # case_id, workflow_id, collection_id, document_id, document_version_id
    # are now extracted separately and placed in BusinessContext
    scope_kwargs = {
        "tenant_id": tenant_id,
        "trace_id": trace_id,
        "invocation_id": invocation_id,
        "user_id": user_id,
        "service_id": service_id,
        "run_id": run_id,
        "ingestion_run_id": ingestion_run_id,
        "idempotency_key": idempotency_key,
        "tenant_schema": tenant_schema,
        "timestamp": datetime.now(timezone.utc),
    }

    return ScopeContext.model_validate(scope_kwargs)


def normalize_meta(request: Any) -> dict:
    """Return a normalised metadata mapping for graph executions.

    BREAKING CHANGE (Option A - Graph-Specific Validation):
    No longer enforces case_id globally. Business context fields are optional.
    Individual graphs validate required fields based on their needs.

    Canonical runtime context injection:
    - `normalize_meta()` attaches `scope_context`, `business_context`, and `tool_context`.
    - Graph entrypoints should parse `tool_context` once (via `tool_context_from_meta`)
      and keep it in state for node access.
    """

    scope = _build_scope_context(request)

    # Extract business context IDs from request headers (all optional)
    case_id = _coalesce(request, X_CASE_ID_HEADER, META_CASE_ID_KEY)
    workflow_id = _coalesce(request, X_WORKFLOW_ID_HEADER, META_WORKFLOW_ID_KEY)
    collection_id = _coalesce(request, X_COLLECTION_ID_HEADER, META_COLLECTION_ID_KEY)
    document_id = _coalesce(request, X_DOCUMENT_ID_HEADER, META_DOCUMENT_ID_KEY)
    document_version_id = _coalesce(
        request, X_DOCUMENT_VERSION_ID_HEADER, META_DOCUMENT_VERSION_ID_KEY
    )

    # Build BusinessContext (all fields optional per Option A)
    business = BusinessContext(
        case_id=case_id,
        workflow_id=workflow_id,
        collection_id=collection_id,
        document_id=document_id,
        document_version_id=document_version_id,
    )

    graph_name = _resolve_graph_name(request)
    graph_version = getattr(request, "graph_version", "v0")
    key_alias = _coalesce(request, X_KEY_ALIAS_HEADER, META_KEY_ALIAS_KEY)

    requested_at = scope.timestamp.isoformat()

    meta = {
        "graph_name": graph_name,
        "graph_version": graph_version,
        "requested_at": requested_at,
        "rate_limit": {"quota": get_quota()},
        "scope_context": scope.model_dump(mode="json", exclude_none=True),
        "business_context": business.model_dump(mode="json", exclude_none=True),
    }

    if scope.tenant_schema:
        meta["tenant_schema"] = scope.tenant_schema
    if key_alias:
        meta["key_alias"] = key_alias

    context_metadata = {
        "graph_name": graph_name,
        "graph_version": graph_version,
        "requested_at": requested_at,
    }

    meta["context_metadata"] = context_metadata

    # Build ToolContext with BusinessContext
    tool_context = scope.to_tool_context(business=business, metadata=context_metadata)

    meta["tool_context"] = tool_context.model_dump(exclude_none=True)

    return meta


def merge_state(
    old: Mapping[str, Any] | None, incoming: Mapping[str, Any] | None
) -> dict:
    """Return a new state mapping with ``incoming`` values overwriting ``old``."""

    merged: dict[str, Any] = {}
    if old:
        merged.update(dict(old))
    if incoming:
        merged.update(dict(incoming))
    return merged
