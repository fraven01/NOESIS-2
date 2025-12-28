"""HTTP request normalization for ScopeContext."""

from __future__ import annotations

import uuid
from typing import Any, Mapping, MutableMapping
from uuid import UUID

from django.http import HttpRequest

from ai_core.contracts.scope import ScopeContext
from ai_core.ids import (
    coerce_trace_id,
    normalize_idempotency_key,
)
from common.constants import (
    META_TENANT_SCHEMA_KEY,
    X_TENANT_SCHEMA_HEADER,
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


def normalize_request(
    request: HttpRequest, *, require_auth: bool = False
) -> ScopeContext:
    """
    Normalize a Django HttpRequest into a ScopeContext.

    This function acts as the Single Source of Truth for extracting context
    from an HTTP request. It handles:
    - Header extraction and normalization
    - Trace ID coercion
    - Tenant context resolution
    - User ID extraction from Django auth (for User Request Hops)
    - UUID generation for missing invocation_id/run_id
    - Collection ID extraction for scoped operations
    - Validation via ScopeContext model

    Identity rules (Pre-MVP ID Contract):
    - User Request Hop: user_id REQUIRED (when auth present), service_id ABSENT.
    - This function sets user_id from request.user if authenticated.
    - service_id is always None for HTTP requests (S2S Hops set it separately).

    Args:
        request: Django HttpRequest object.
        require_auth: If True, raise error if user is not authenticated.
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

    # BREAKING CHANGE (Option A): Business IDs removed from ScopeContext
    # case_id, workflow_id, collection_id are now extracted in normalize_meta()
    # and placed in BusinessContext, not here.

    # 3. Idempotency Key
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

    # Logic for run_id/ingestion_run_id (Pre-MVP ID Contract):
    # Both may co-exist (workflow triggers ingestion).
    # At least one runtime ID required. If neither present, generate run_id.
    if not ingestion_run_id and not run_id:
        run_id = uuid.uuid4().hex

    # BREAKING CHANGE (Option A): collection_id removed from ScopeContext
    # It's now a business domain ID, extracted in normalize_meta() → BusinessContext

    # 9. User ID (Pre-MVP ID Contract: User Request Hops)
    # Extract from Django auth if user is authenticated
    user_id: str | None = None
    user = getattr(request, "user", None)
    if user is not None and getattr(user, "is_authenticated", False):
        # User is authenticated - extract user_id
        user_pk = getattr(user, "pk", None)
        if user_pk is not None:
            user_id = str(user_pk)

    if require_auth and not user_id:
        from django.core.exceptions import PermissionDenied

        raise PermissionDenied("Authentication required but user is not authenticated")

    # 11. Service ID (Pre-MVP ID Contract: S2S Hops)
    # For HTTP requests, service_id is always None (User Request Hop pattern)
    # S2S Hops (Celery tasks, internal calls) set service_id separately
    service_id: str | None = None

    # BREAKING CHANGE (Option A - Strict Separation):
    # Business domain IDs (case_id, workflow_id, collection_id) are NO LONGER
    # part of ScopeContext. They are extracted separately in normalize_meta()
    # and placed in BusinessContext.
    scope_kwargs = {
        "tenant_id": tenant_id,  # ScopeContext will validate this is not None
        "trace_id": trace_id,
        "invocation_id": invocation_id,
        "user_id": user_id,
        "service_id": service_id,
        "run_id": run_id,
        "ingestion_run_id": ingestion_run_id,
        "idempotency_key": idempotency_key,
        "tenant_schema": tenant_schema,
        # REMOVED (Option A): case_id, workflow_id, collection_id → BusinessContext
    }

    return ScopeContext.model_validate(scope_kwargs)


def normalize_task_context(
    *,
    tenant_id: str,
    service_id: str,
    case_id: str | None = None,
    trace_id: str | None = None,
    invocation_id: str | None = None,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    workflow_id: str | None = None,
    idempotency_key: str | None = None,
    tenant_schema: str | None = None,
    collection_id: str | None = None,
    initiated_by_user_id: str | None = None,
) -> ScopeContext:
    """
    Build a ScopeContext for Celery task execution (S2S Hop).

    This function is the canonical entry point for creating scope in Celery tasks.
    It enforces the S2S identity pattern: service_id REQUIRED, user_id ABSENT.

    Identity rules (Pre-MVP ID Contract):
    - S2S Hop: service_id REQUIRED, user_id ABSENT.
    - initiated_by_user_id is NOT part of ScopeContext (it goes in audit_meta).

    BREAKING CHANGE (Option A - Strict Separation):
    - Business domain IDs (case_id, workflow_id, collection_id) are NO LONGER
      part of ScopeContext. They are kept as parameters for backward compatibility
      but are NOT included in the returned ScopeContext.
    - Callers should extract business IDs separately and build BusinessContext.

    Args:
        tenant_id: Mandatory tenant identifier.
        service_id: REQUIRED for S2S. E.g., "celery-ingestion-worker".
        trace_id: Inherited from parent hop, or generated if None.
        invocation_id: New for this hop, or generated if None.
        run_id: Runtime execution ID (may co-exist with ingestion_run_id).
        ingestion_run_id: Ingestion execution ID (may co-exist with run_id).
        idempotency_key: Optional request-level determinism key.
        tenant_schema: Optional tenant schema (derived from tenant_id if None).
        case_id: DEPRECATED - No longer part of ScopeContext (kept for compatibility).
        workflow_id: DEPRECATED - No longer part of ScopeContext (kept for compatibility).
        collection_id: DEPRECATED - No longer part of ScopeContext (kept for compatibility).
        initiated_by_user_id: NOT used in ScopeContext - for audit_meta only.
            Pass this to audit_meta_from_scope() when persisting entities.

    Returns:
        ScopeContext with service_id set (S2S Hop pattern).
    """
    if not service_id:
        raise ValueError("service_id is required for S2S Hops (Celery tasks)")

    # Generate IDs if not provided
    final_trace_id = trace_id or uuid.uuid4().hex
    final_invocation_id = invocation_id or uuid.uuid4().hex

    # At least one runtime ID required
    if not run_id and not ingestion_run_id:
        run_id = uuid.uuid4().hex

    # BREAKING CHANGE (Option A - Strict Separation):
    # Business domain IDs (case_id, workflow_id, collection_id) are NO LONGER
    # part of ScopeContext. Callers should build BusinessContext separately.
    scope_kwargs = {
        "tenant_id": tenant_id,
        "trace_id": final_trace_id,
        "invocation_id": final_invocation_id,
        "user_id": None,  # S2S Hops have no user_id
        "service_id": service_id,  # S2S Hops REQUIRE service_id
        "run_id": run_id,
        "ingestion_run_id": ingestion_run_id,
        "idempotency_key": idempotency_key,
        "tenant_schema": tenant_schema,
        # REMOVED (Option A): case_id, workflow_id, collection_id → BusinessContext
    }

    return ScopeContext.model_validate(scope_kwargs)
