"""HTTP request normalization for ScopeContext."""

from __future__ import annotations

import uuid
from typing import Any, Mapping, MutableMapping
from uuid import UUID

from django.http import HttpRequest

from ai_core.contracts.scope import ScopeContext
from ai_core.ids import (
    coerce_trace_id,
    normalize_case_header,
    normalize_idempotency_key,
)
from common.constants import (
    META_TENANT_SCHEMA_KEY,
    META_WORKFLOW_ID_KEY,
    X_TENANT_SCHEMA_HEADER,
    X_WORKFLOW_ID_HEADER,
)


def _coalesce(request: HttpRequest, header: str, meta_key: str) -> str | None:
    headers: Mapping[str, str] = getattr(request, "headers", {}) or {}
    meta: MutableMapping[str, Any] = getattr(request, "META", {}) or {}
    value = headers.get(header)
    if value is None:
        value = meta.get(meta_key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _normalize_header_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def _normalize_uuid_header(value: Any) -> UUID | None:
    """Normalize a header value to a UUID or None."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return UUID(stripped)
        except ValueError:
            return None
    return None


def normalize_request(request: HttpRequest) -> ScopeContext:
    """
    Normalize a Django HttpRequest into a ScopeContext.

    This function acts as the Single Source of Truth for extracting context
    from an HTTP request. It handles:
    - Header extraction and normalization
    - Trace ID coercion
    - Tenant context resolution
    - UUID generation for missing invocation_id/run_id
    - Collection ID extraction for scoped operations
    - Validation via ScopeContext model
    """
    headers: Mapping[str, str] = getattr(request, "headers", {}) or {}
    meta: MutableMapping[str, Any] = getattr(request, "META", {}) or {}

    # 1. Trace ID
    # Try coerce_trace_id first as it handles various formats/locations
    try:
        trace_id, _ = coerce_trace_id(meta)
    except ValueError:
        trace_id = uuid.uuid4().hex

    # 2. Tenant ID
    # Priority:
    # 1. TenantContext (resolved from domain/URL/middleware)
    # 2. Explicit Headers (X-Tenant-Id) - REMOVED to enforce TenantContext policy

    from customers.tenant_context import TenantContext

    tenant = TenantContext.from_request(request, require=False)

    tenant_id: str | None = None
    if tenant:
        tenant_id = tenant.schema_name

    # We no longer check headers manually here. TenantContext.from_request should handle it
    # if configured to allow headers.

    if not tenant_id:
        # If still no tenant_id, we cannot proceed as ScopeContext requires it.
        # We rely on TenantContext to have checked headers if allowed.
        from customers.tenant_context import TenantRequiredError

        raise TenantRequiredError("Tenant could not be resolved from request")

    # 3. Case ID
    case_id = normalize_case_header(meta)

    # 4. Workflow ID
    workflow_id = _coalesce(request, X_WORKFLOW_ID_HEADER, META_WORKFLOW_ID_KEY)

    # 5. Idempotency Key
    idempotency_key = normalize_idempotency_key(meta)

    # 6. Tenant Schema
    tenant_schema = _coalesce(request, X_TENANT_SCHEMA_HEADER, META_TENANT_SCHEMA_KEY)
    if not tenant_schema and tenant_id:
        # If we have a tenant_id but no explicit schema, they are often the same
        # But we'll leave it None unless explicitly provided or resolved from context
        from customers.tenant_context import TenantContext

        tenant = TenantContext.from_request(request, require=False)
        if tenant and tenant.schema_name == tenant_id:
            tenant_schema = tenant.schema_name

    # 7. Invocation ID
    invocation_id = (
        _normalize_header_value(
            headers.get("X-Invocation-ID") or meta.get("HTTP_X_INVOCATION_ID")
        )
        or uuid.uuid4().hex
    )

    # 8. Run ID / Ingestion Run ID
    run_id = _normalize_header_value(
        headers.get("X-Run-ID") or meta.get("HTTP_X_RUN_ID")
    )
    ingestion_run_id = _normalize_header_value(
        headers.get("X-Ingestion-Run-ID") or meta.get("HTTP_X_INGESTION_RUN_ID")
    )

    # Logic for run_id/ingestion_run_id:
    # If ingestion_run_id is present, we use it.
    # If NOT present, we MUST have a run_id. If run_id is also missing, generate one.
    # ScopeContext validation will ensure XOR.
    if not ingestion_run_id and not run_id:
        run_id = uuid.uuid4().hex

    # 9. Collection ID ("Aktenschrank" context for scoped operations)
    collection_id = _normalize_uuid_header(
        headers.get("X-Collection-ID") or meta.get("HTTP_X_COLLECTION_ID")
    )

    scope_kwargs = {
        "tenant_id": tenant_id,  # ScopeContext will validate this is not None
        "trace_id": trace_id,
        "invocation_id": invocation_id,
        "run_id": run_id,
        "ingestion_run_id": ingestion_run_id,
        "case_id": case_id,
        "workflow_id": workflow_id,
        "idempotency_key": idempotency_key,
        "tenant_schema": tenant_schema,
        "collection_id": collection_id,
    }

    return ScopeContext.model_validate(scope_kwargs)
