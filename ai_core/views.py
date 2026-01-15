from __future__ import annotations

import hashlib
import os
import json
import re
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING
from pathlib import Path
from django.conf import settings
from uuid import uuid4


from opentelemetry import trace
from opentelemetry.trace import format_trace_id
from django.db import OperationalError, ProgrammingError
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest
from .schemas import CrawlerRunRequest, RagHardDeleteAdminRequest
from django.views.decorators.http import require_POST
from ai_core.graph.core import FileCheckpointer

from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    META_CASE_ID_KEY,
    META_COLLECTION_ID_KEY,
    META_IDEMPOTENCY_KEY,
    META_KEY_ALIAS_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    META_TRACE_ID_KEY,
    META_WORKFLOW_ID_KEY,
    X_TRACE_ID_HEADER,
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
    X_WORKFLOW_ID_HEADER,
)
from common.logging import bind_log_context, get_logger
from noesis2.api import (
    DeprecationHeadersMixin,
    RATE_LIMIT_ERROR_STATUSES,
    RATE_LIMIT_JSON_ERROR_STATUSES,
    curl_code_sample,
    default_extend_schema,
)
from noesis2.api.serializers import (
    IntakeRequestSerializer,
    IntakeResponseSerializer,
    PingResponseSerializer,
    RagQueryResponseSerializer,
)

# Crawler contracts and runtime structures used by the ingestion runner view.
from crawler.errors import CrawlerError, ErrorClass
from documents.contract_utils import (
    normalize_media_type as normalize_document_media_type,
)

# OpenAPI helpers and serializer types are referenced throughout the schema
# declarations below, so keep the imports explicit even if they appear unused
# near the view definitions.
from drf_spectacular.utils import OpenApiExample, inline_serializer
from rest_framework import serializers, status
from rest_framework.authentication import SessionAuthentication
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


from ai_core.authz.visibility import allow_extended_visibility
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import tool_context_from_meta
from ai_core.contracts.crawler_runner import CrawlerRunError
from ai_core.middleware import guardrails as guardrails_middleware
from ai_core.rag.guardrails import (
    GuardrailSignals,
)

# Re-export normalize_meta so tests can monkeypatch via ai_core.views
from ai_core.graph.schemas import normalize_meta as normalize_meta  # noqa: F401
from pydantic import ValidationError

from . import services
from ai_core.services.crawler_runner import run_crawler_runner
from .infra import rate_limit
from .infra.resp import apply_std_headers, build_tool_error_payload
from .ingestion import partition_document_ids as partition_document_ids  # test hook
from .ingestion import run_ingestion as run_ingestion  # re-export for tests
from .ingestion_status import (
    get_latest_ingestion_run,
)
from .rag.hard_delete import hard_delete
from .rag.ingestion_contracts import (
    resolve_ingestion_profile as resolve_ingestion_profile,  # test hook
)
from .views_response_utils import apply_response_headers

from cases.models import Case
from cases.services import CaseNotFoundError, ensure_case, resolve_case


GuardrailErrorCategory = guardrails_middleware.GuardrailErrorCategory

# Optional hooks for tests to provide lifecycle stores without
# importing heavy dependencies at module import time.
DOCUMENTS_LIFECYCLE_STORE: object | None = None
# Removed _DEFAULT_CRAWLER_GRAPH_MODULE (Finding #5): Now uses lazy import in _build_ingestion_graph

if TYPE_CHECKING:  # pragma: no cover - type checking only
    pass


logger = get_logger(__name__)

logger.info(
    "module_loaded",
    extra={"module": __name__, "path": str(Path(__file__).resolve())},
)


def _token_tenant_identifier(request: object) -> str | None:
    """Return the tenant identifier bound to an authenticated token if present."""

    auth = getattr(request, "auth", None)
    if isinstance(auth, Mapping):
        for key in ("tenant_id", "tenant", "tenant_schema"):
            value = auth.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr in ("tenant_id", "tenant", "tenant_schema"):
        value = getattr(auth, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def assert_case_active(
    request: HttpRequest | Request,
    case_id: str,
    *,
    tenant_identifier: str | None = None,
) -> Response | None:
    """Ensure the referenced case exists and can accept new work."""

    from customers.tenant_context import TenantContext, TenantRequiredError

    tenant_header = None
    if hasattr(request, "headers"):
        tenant_header = request.headers.get(X_TENANT_ID_HEADER)
    if not tenant_header:
        meta = getattr(request, "META", {}) or {}
        tenant_header = meta.get(META_TENANT_ID_KEY)

    allow_header_resolution = bool(
        getattr(settings, "TESTING", False) or os.environ.get("PYTEST_CURRENT_TEST")
    )
    header_schema = None
    if hasattr(request, "headers"):
        header_schema = request.headers.get(X_TENANT_SCHEMA_HEADER)
    if header_schema is None:
        meta = getattr(request, "META", {}) or {}
        header_schema = meta.get(META_TENANT_SCHEMA_KEY)
    header_tenant = None
    if allow_header_resolution:
        if header_schema:
            header_tenant = TenantContext.resolve_identifier(header_schema)
        if header_tenant is None and tenant_header:
            header_tenant = TenantContext.resolve_identifier(
                tenant_header, allow_pk=True
            )
        if tenant_header is None and header_tenant is not None:
            tenant_header = getattr(header_tenant, "schema_name", None)
        if tenant_header is None and header_tenant is not None:
            tenant_header = getattr(header_tenant, "schema_name", None)

    try:
        resolved_tenant = TenantContext.from_request(
            request,
            allow_headers=True,
            require=True,
            use_connection_schema=False,
        )
    except TenantRequiredError as exc:
        if allow_header_resolution and header_tenant is not None:
            resolved_tenant = header_tenant
        else:
            return _error_response(
                str(exc),
                "tenant_not_found",
                status.HTTP_403_FORBIDDEN,
            )
    if allow_header_resolution and header_tenant is not None:
        public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
        if (
            resolved_tenant is None
            or getattr(resolved_tenant, "schema_name", "") == public_schema
        ):
            resolved_tenant = header_tenant

    explicit_tenant = None
    if tenant_identifier is not None:
        explicit_tenant = TenantContext.resolve_identifier(tenant_identifier)
        if explicit_tenant is None:
            return _error_response(
                "Tenant context for this request could not be resolved.",
                "tenant_not_found",
                status.HTTP_404_NOT_FOUND,
            )

    tenant_obj = explicit_tenant or resolved_tenant

    if explicit_tenant is not None and explicit_tenant != resolved_tenant:
        return _error_response(
            "Tenant identifier does not match the authenticated request tenant.",
            "tenant_mismatch",
            status.HTTP_403_FORBIDDEN,
        )

    if (
        tenant_header
        and tenant_obj is not None
        and tenant_header.strip() != getattr(tenant_obj, "schema_name", None)
    ):
        return _error_response(
            "Tenant identifier does not match the authenticated request tenant.",
            "tenant_mismatch",
            status.HTTP_403_FORBIDDEN,
        )

    token_tenant = _token_tenant_identifier(request)
    if token_tenant and token_tenant != getattr(tenant_obj, "schema_name", None):
        return _error_response(
            "Token tenant does not match the requested tenant context.",
            "tenant_mismatch",
            status.HTTP_403_FORBIDDEN,
        )

    normalized_case_id = (case_id or "").strip()
    if not normalized_case_id:
        return _error_response(
            "Case header is required and must use the documented format.",
            "invalid_case_header",
            status.HTTP_400_BAD_REQUEST,
        )

    try:
        case = Case.objects.filter(
            tenant=tenant_obj, external_id=normalized_case_id
        ).first()
    except (ProgrammingError, OperationalError):
        case = None

    if case is None:
        try:
            case = resolve_case(tenant_obj, normalized_case_id)
        except (ProgrammingError, OperationalError, CaseNotFoundError):
            case = None

    if case is None:
        # Allow dev cases for dev tenant (include test schema in test runs)
        test_schema = getattr(settings, "TEST_TENANT_SCHEMA", None)
        dev_schemas = {"dev", "autotest"}
        if test_schema:
            dev_schemas.add(test_schema)
        is_dev_tenant = getattr(tenant_obj, "schema_name", "") in dev_schemas
        is_dev_case = (
            normalized_case_id.startswith("dev-case-") or normalized_case_id == "upload"
        )
        if is_dev_tenant and is_dev_case:
            return None

        return _error_response(
            "Case not found for this tenant. Create or activate the case before invoking this endpoint.",
            "case_not_found",
            status.HTTP_404_NOT_FOUND,
        )

    if case.status == Case.Status.CLOSED:
        return _error_response(
            "Case is closed and cannot accept additional requests.",
            "case_closed",
            status.HTTP_409_CONFLICT,
        )

    return None


def make_fallback_external_id(filename: str, size: int, data: bytes) -> str:
    """Derive a deterministic fallback external_id for uploads.

    Combines filename, declared size and raw bytes using SHA-256, matching
    historical behaviour used by tests and legacy ingestion flows.
    """
    try:
        name_bytes = filename.encode("utf-8")
    except Exception:  # pragma: no cover - defensive
        name_bytes = str(filename).encode("utf-8", errors="ignore")
    size_bytes = str(size).encode("utf-8")
    buffer = name_bytes + size_bytes + (data or b"")

    return hashlib.sha256(buffer).hexdigest()


def _build_guardrail_error(
    category: GuardrailErrorCategory,
    reason: str,
    signals: GuardrailSignals,
    attributes: Mapping[str, object],
) -> CrawlerError:
    error_class = {
        GuardrailErrorCategory.POLICY_DENY: ErrorClass.POLICY_DENY,
        GuardrailErrorCategory.TIMEOUT: ErrorClass.TIMEOUT,
    }.get(category, ErrorClass.POLICY_DENY)
    return CrawlerError(
        error_class=error_class,
        reason=reason,
        source=signals.canonical_source,
        provider=signals.provider,
        attributes=dict(attributes or {}),
    )


def _resolve_tenant_id(request: HttpRequest) -> str | None:
    """Derive the active tenant schema for the current request."""

    from customers.tenant_context import TenantContext, TenantRequiredError

    try:
        tenant = TenantContext.from_request(
            request,
            allow_headers=False,
            require=False,
            use_connection_schema=False,
        )
    except TenantRequiredError:
        return None

    return tenant.schema_name if tenant else None


DEFAULT_CASE_ID = "general"
DEV_DEFAULT_CASE_ID = "dev-case-local"

KEY_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _error_response(detail: str, code: str, status_code: int) -> Response:
    """Return a standardised error payload."""
    payload = build_tool_error_payload(
        message=detail,
        status_code=status_code,
        code=code,
    )
    return Response(payload, status=status_code)


def _extract_internal_key(request: object) -> str | None:
    """Return the internal service key from *request* when present."""

    if hasattr(request, "headers"):
        key = request.headers.get("X-Internal-Key")
        if isinstance(key, str) and key.strip():
            return key.strip()
    meta = getattr(request, "META", None)
    if isinstance(meta, dict):
        key_meta = meta.get("HTTP_X_INTERNAL_KEY")
        if isinstance(key_meta, str) and key_meta.strip():
            return key_meta.strip()
    return None


def _normalize_media_type_value(value: str | None) -> str | None:
    """Return a normalized ``type/subtype`` media type or ``None`` when invalid."""

    if not value:
        return None
    candidate = value.split(";", 1)[0].strip()
    if not candidate:
        return None
    try:
        return normalize_document_media_type(candidate)
    except ValueError:
        return None


def _sanitize_primary_text(value: str | None) -> str:
    """Strip disallowed control characters (e.g. NUL) from primary text payloads."""

    if not value:
        return ""
    sanitized = value.replace("\x00", " ")
    return sanitized


def _resolve_hard_delete_actor(
    request: object, operator_label: str | None
) -> dict[str, object]:
    """Derive the Celery actor payload for hard delete requests."""

    allowed_keys = getattr(settings, "RAG_INTERNAL_KEYS", ()) or ()
    candidates: list[object] = []
    if request is not None:
        candidates.append(request)
        django_request = getattr(request, "_request", None)
        if django_request is not None and django_request is not request:
            candidates.append(django_request)

    internal_key = None
    for candidate in candidates:
        internal_key = _extract_internal_key(candidate)
        if internal_key:
            break
    if internal_key:
        if internal_key not in allowed_keys:
            raise PermissionDenied("Service key is not authorised for hard delete")
        actor: dict[str, object] = {"internal_key": internal_key}
        if operator_label:
            actor["label"] = operator_label
        return actor

    user = None
    for candidate in candidates:
        candidate_user = getattr(candidate, "user", None)
        if candidate_user is not None and getattr(
            candidate_user, "is_authenticated", False
        ):
            user = candidate_user
            break

    if user is None:
        for candidate in candidates:
            session = getattr(candidate, "session", None)
            session_user_id = None
            if session is not None:
                session_user_id = session.get("_auth_user_id")
            if session_user_id:
                user_model = get_user_model()
                try:
                    user = user_model.objects.get(pk=session_user_id)
                except user_model.DoesNotExist:  # pragma: no cover - defensive
                    user = None
                else:
                    break
    if user is None or not getattr(user, "is_authenticated", False):
        raise PermissionDenied("Hard delete requires a service key or admin session")

    visibility_request = None
    for candidate in candidates:
        candidate_user = getattr(candidate, "user", None)
        if candidate_user is not None and getattr(
            candidate_user, "is_authenticated", False
        ):
            visibility_request = candidate
            break
    if visibility_request is None:
        visibility_request = candidates[-1] if candidates else request
    if not allow_extended_visibility(visibility_request):
        raise PermissionDenied("Extended visibility not permitted for this request")

    actor: dict[str, object] = {"user_id": user.pk}
    label = operator_label
    if not label:
        full_name = getattr(user, "get_full_name", None)
        if callable(full_name):
            label = (full_name() or "").strip() or None
    if not label:
        username = getattr(user, "get_username", None)
        if callable(username):
            label = (username() or "").strip() or None
    if label:
        actor["label"] = label
    return actor


def _prepare_request(request: Request):
    """
    Prepare a request by extracting, validating, and enriching context.

    This function provides request preparation for AI Core views including:
    - Header validation (tenant, case, key_alias formats)
    - Rate limiting enforcement
    - Default case bootstrap ("general")
    - Case active status check
    - ScopeContext building via normalize_request() (Pre-MVP ID Contract)
    - Request.META enrichment for downstream consumers
    - Log context binding

    Returns:
        tuple: (meta dict, error response or None)
            meta includes 'scope_context' with serialized ScopeContext and tool_context
    """
    from customers.tenant_context import TenantContext, TenantRequiredError

    tenant_header = request.headers.get(X_TENANT_ID_HEADER)
    if tenant_header is None:
        tenant_header = request.META.get(META_TENANT_ID_KEY)
    if tenant_header is None:
        tenant_header = request.META.get(X_TENANT_ID_HEADER)
    case_id = (
        request.headers.get(X_CASE_ID_HEADER)
        or request.META.get(META_CASE_ID_KEY)
        or ""
    ).strip()
    workflow_id = (
        request.headers.get(X_WORKFLOW_ID_HEADER)
        or request.META.get(META_WORKFLOW_ID_KEY)
        or ""
    ).strip()
    if not workflow_id:
        # Default workflow_id to case_id if available, otherwise generate UUID
        workflow_id = case_id or uuid4().hex
    key_alias_header = request.headers.get(X_KEY_ALIAS_HEADER)
    collection_header = request.headers.get(X_COLLECTION_ID_HEADER)
    if collection_header is None:
        collection_header = request.META.get(META_COLLECTION_ID_KEY)
    idempotency_header = request.headers.get(IDEMPOTENCY_KEY_HEADER)
    if not idempotency_header:
        idempotency_header = request.META.get(
            "HTTP_" + IDEMPOTENCY_KEY_HEADER.upper().replace("-", "_")
        )
    if not idempotency_header:
        idempotency_header = request.META.get(IDEMPOTENCY_KEY_HEADER)

    idempotency_key = None
    if idempotency_header is not None:
        candidate = idempotency_header.strip()
        if candidate:
            idempotency_key = candidate

    allow_header_resolution = bool(
        getattr(settings, "TESTING", False) or os.environ.get("PYTEST_CURRENT_TEST")
    )
    header_schema = request.headers.get(X_TENANT_SCHEMA_HEADER)
    if header_schema is None:
        header_schema = request.META.get(META_TENANT_SCHEMA_KEY)
    header_tenant = None
    if allow_header_resolution:
        if header_schema:
            header_tenant = TenantContext.resolve_identifier(header_schema)
        if header_tenant is None and tenant_header:
            header_tenant = TenantContext.resolve_identifier(
                tenant_header, allow_pk=True
            )

    try:
        tenant_obj = TenantContext.from_request(
            request,
            allow_headers=allow_header_resolution,
            require=True,
            use_connection_schema=False,
        )
    except TenantRequiredError as exc:
        if allow_header_resolution and header_tenant is not None:
            tenant_obj = header_tenant
        else:
            return None, _error_response(
                str(exc),
                "tenant_not_found",
                status.HTTP_403_FORBIDDEN,
            )

    if allow_header_resolution and header_tenant is not None:
        public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
        if (
            tenant_obj is None
            or getattr(tenant_obj, "schema_name", "") == public_schema
        ):
            tenant_obj = header_tenant

    tenant_schema = getattr(tenant_obj, "schema_name", "")
    if not tenant_schema:
        return None, _error_response(
            "Tenant schema could not be resolved from request context.",
            "tenant_not_found",
            status.HTTP_403_FORBIDDEN,
        )

    if tenant_header is None:
        return None, _error_response(
            "Tenant header is required for multi-tenant requests.",
            "invalid_tenant_header",
            status.HTTP_400_BAD_REQUEST,
        )

    tenant_id = tenant_header.strip()
    if not tenant_id or not TENANT_ID_RE.fullmatch(tenant_id):
        return None, _error_response(
            "Tenant header is required for multi-tenant requests.",
            "invalid_tenant_header",
            status.HTTP_400_BAD_REQUEST,
        )

    if tenant_id != tenant_schema:
        return None, _error_response(
            "Tenant header does not match the authenticated request tenant.",
            "tenant_mismatch",
            status.HTTP_403_FORBIDDEN,
        )

    schema_header = request.headers.get(X_TENANT_SCHEMA_HEADER)
    if schema_header is None:
        schema_header = request.META.get(META_TENANT_SCHEMA_KEY)
    if schema_header is not None:
        header_schema = schema_header.strip()
        if not header_schema:
            return None, _error_response(
                "Tenant schema header must not be empty.",
                "invalid_tenant_schema",
                status.HTTP_400_BAD_REQUEST,
            )
        if header_schema != tenant_schema:
            return None, _error_response(
                "Tenant schema header does not match resolved schema.",
                "tenant_schema_mismatch",
                status.HTTP_400_BAD_REQUEST,
            )

    if not case_id:
        case_id = DEFAULT_CASE_ID

    if case_id and not CASE_ID_RE.fullmatch(case_id):
        return None, _error_response(
            "Case header must use the documented format.",
            "invalid_case_header",
            status.HTTP_400_BAD_REQUEST,
        )

    if not rate_limit.check(tenant_id):
        return None, _error_response(
            "Rate limit exceeded for tenant.",
            "rate_limit_exceeded",
            status.HTTP_429_TOO_MANY_REQUESTS,
        )

    key_alias = None
    if key_alias_header is not None:
        key_alias = key_alias_header.strip()
        if not key_alias or not KEY_ALIAS_RE.fullmatch(key_alias):
            return None, _error_response(
                "Key alias header does not match the required format.",
                "invalid_key_alias",
                status.HTTP_400_BAD_REQUEST,
            )

    collection_id = None
    if collection_header is not None:
        candidate = collection_header.strip()
        if candidate:
            collection_id = candidate

    # Fix: Respect passed trace_id or generate new one
    trace_id_header = request.headers.get(X_TRACE_ID_HEADER)
    trace_id = (trace_id_header or "").strip()

    # [Telemetry Fix] Check OTel context first for valid trace
    if not trace_id:
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            trace_id = format_trace_id(ctx.trace_id)

    if not trace_id:
        trace_id = uuid4().hex
    if case_id == DEFAULT_CASE_ID:
        try:
            ensure_case(
                tenant_obj,
                case_id,
                title="General",
                reopen_closed=True,
            )
        except Exception:
            logger.exception(
                "case.default_bootstrap_failed",
                extra={"tenant_id": tenant_schema, "case_id": case_id},
            )
            return None, _error_response(
                "Failed to bootstrap default case.",
                "case_bootstrap_failed",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    if case_id:
        case_error = assert_case_active(
            request, case_id, tenant_identifier=tenant_schema
        )
        if case_error is not None:
            return None, case_error

    request.META[META_TRACE_ID_KEY] = trace_id
    request.META[META_CASE_ID_KEY] = case_id
    request.META[META_TENANT_ID_KEY] = tenant_id
    request.META[META_TENANT_SCHEMA_KEY] = tenant_schema
    if workflow_id:
        request.META[META_WORKFLOW_ID_KEY] = workflow_id
    else:
        request.META.pop(META_WORKFLOW_ID_KEY, None)
    if key_alias:
        request.META[META_KEY_ALIAS_KEY] = key_alias
    else:
        request.META.pop(META_KEY_ALIAS_KEY, None)
    if collection_id:
        request.META[META_COLLECTION_ID_KEY] = collection_id
    else:
        request.META.pop(META_COLLECTION_ID_KEY, None)
    if idempotency_key:
        request.META[META_IDEMPOTENCY_KEY] = idempotency_key
    else:
        request.META.pop(META_IDEMPOTENCY_KEY, None)

    # Build ScopeContext via normalize_request (Pre-MVP ID Contract)
    # This provides a standardized scope object for downstream consumers
    from ai_core.ids.http_scope import normalize_request

    scope_context = normalize_request(request)
    scope_payload = scope_context.model_dump(mode="json")
    scope_trace_id = scope_payload["trace_id"]
    scope_tenant_id = scope_payload["tenant_id"]
    scope_tenant_schema = scope_payload.get("tenant_schema") or tenant_schema

    # BREAKING CHANGE (Option A - Strict Separation):
    # Business IDs (case_id, workflow_id, collection_id) are NO LONGER in ScopeContext
    # They were already extracted from headers and set in META at lines 633-652
    # DO NOT overwrite them here - they remain in META for normalize_meta() to read

    # Update META with infrastructure IDs from ScopeContext
    request.META[META_TRACE_ID_KEY] = scope_trace_id
    if scope_tenant_id:
        request.META[META_TENANT_ID_KEY] = scope_tenant_id
    if scope_tenant_schema:
        request.META[META_TENANT_SCHEMA_KEY] = scope_tenant_schema

    # Business IDs (case_id, workflow_id, collection_id) remain in META from header extraction
    # normalize_meta() will build BusinessContext from these META values

    # BREAKING CHANGE (Option A - Strict Separation):
    # Build BusinessContext from extracted business domain IDs
    from ai_core.contracts.business import BusinessContext

    business_context = BusinessContext(
        case_id=case_id,
        workflow_id=workflow_id,
        collection_id=collection_id,
        # document_id and document_version_id extracted from headers if needed
    )

    tool_context = scope_context.to_tool_context(business=business_context)

    # Build meta dict from validated scope and additional view-specific fields
    meta = {
        "scope_context": scope_payload,
        "business_context": business_context.model_dump(mode="json", exclude_none=True),
        "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        "tenant_schema": scope_tenant_schema,
    }
    if key_alias:
        meta["key_alias"] = key_alias

    log_context = {
        "trace_id": scope_trace_id,
        "case_id": case_id,  # From header extraction (line 480)
        "workflow_id": workflow_id,  # From header extraction (line 481)
        "tenant_id": scope_tenant_id,
        "tenant": scope_tenant_id,
        "key_alias": key_alias,
        "collection_id": collection_id,  # From header extraction (line 588-592)
        # Pre-MVP ID Contract: identity tracking
        "user_id": scope_context.user_id,
        "invocation_id": scope_context.invocation_id,
    }
    request.log_context = log_context
    bind_log_context(**log_context)
    return meta, None


def _curl(command: str) -> dict[str, object]:
    """Return extensions embedding a curl code sample."""

    return curl_code_sample(command)


PING_RESPONSE_EXAMPLE = OpenApiExample(
    name="PingResponse",
    summary="Successful heartbeat",
    description="Minimal response confirming that AI Core is reachable.",
    value={"ok": True},
    response_only=True,
)

PING_CURL = _curl(
    " ".join(
        [
            "curl -i",
            '-H "X-Tenant-Schema: dev"',
            '-H "X-Tenant-Id: dev-tenant"',
            '-H "X-Case-Id: local"',
            "https://api.noesis.example/v1/ai/ping/",
        ]
    )
)

INTAKE_REQUEST_EXAMPLE = OpenApiExample(
    name="IntakeRequest",
    summary="Initial workflow context",
    description="Submit the first intake payload to capture tenant metadata and workflow scope.",
    value={
        "tenant_id": "acme",
        "trace_id": "f82c8a4f3f94484a8b1c9d03b1f65e10",
        "prompt": "Starte Intake für Projekt Kickoff",
        "metadata": {"project": "acme-kickoff", "initiator": "alex@example.com"},
        "scope": "kickoff",
        "needs_input": ["agenda", "timeline"],
    },
    request_only=True,
)

INTAKE_RESPONSE_EXAMPLE = OpenApiExample(
    name="IntakeResponse",
    summary="Recorded intake state",
    description="Echoes the tenant metadata after the intake step persisted the workflow.",
    value={
        "received": True,
        "tenant_id": "acme",
        "case_id": "crm-7421",
        "idempotent": False,
    },
    response_only=True,
)

INTAKE_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/v1/ai/intake/",
            '-H "Content-Type: application/json"',
            '-H "X-Tenant-Schema: acme_prod"',
            '-H "X-Tenant-Id: acme"',
            '-H "X-Case-Id: crm-7421"',
            '-H "Idempotency-Key: 1d1d8aa4-0f2e-4b94-8e41-44f96c42e01a"',
            '-d \'{"prompt": "Erstelle Meeting-Notizen"}\'',
        ]
    )
)


RAG_QUERY_REQUEST_EXAMPLE = OpenApiExample(
    name="RagQueryRequest",
    summary="Execute retrieval for a question",
    description="Submit a question, optional pre-composed query and metadata to the production retrieval graph.",
    value={
        "tenant_id": "acme",
        "trace_id": "6cdb89f6-8826-4f9b-8c82-1f14b3d4c21b",
        "question": "Welche Reisekosten gelten für Consultants?",
        "filters": {"doc_class": "policy", "process": "travel"},
        "visibility": "tenant",
    },
    request_only=True,
)

RAG_QUERY_RESPONSE_EXAMPLE = OpenApiExample(
    name="RagQueryResponse",
    summary="Composed retrieval answer",
    description="Returns the composed answer produced by the retrieval and compose nodes.",
    value={
        "answer": "Consultants nutzen das Travel-Policy-Template.",
        "prompt_version": "2024-05-01",
        "retrieval": {
            "alpha": 0.7,
            "min_sim": 0.15,
            "top_k_effective": 1,
            "matches_returned": 1,
            "max_candidates_effective": 50,
            "vector_candidates": 37,
            "lexical_candidates": 41,
            "deleted_matches_blocked": 0,
            "visibility_effective": "active",
            "took_ms": 42,
            "routing": {
                "profile": "standard",
                "vector_space_id": "rag/standard@v1",
            },
        },
        "snippets": [
            {
                "id": "doc-871#p3",
                "text": "Rücksendungen sind innerhalb von 30 Tagen möglich, sofern das Produkt unbenutzt ist.",
                "score": 0.82,
                "source": "policies/returns.md",
                "hash": "7f3d6a2c",
                "meta": {"page": 3, "language": "de"},
            }
        ],
    },
    response_only=True,
)

RAG_QUERY_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/v1/ai/rag/query/",
            '-H "Content-Type: application/json"',
            '-H "X-Tenant-Schema: acme_prod"',
            '-H "X-Tenant-Id: acme"',
            '-H "X-Case-Id: crm-7421"',
            '-H "Idempotency-Key: 6cdb89f6-8826-4f9b-8c82-1f14b3d4c21b"',
            '-d \'{"question": "Welche Reisekosten gelten für Consultants?", "filters": {"doc_class": "policy"}}\'',
        ]
    )
)

RAG_QUERY_REQUEST = inline_serializer(
    name="RagQueryRequest",
    fields={
        "tenant_id": serializers.CharField(required=True),
        "trace_id": serializers.CharField(required=True),
        "question": serializers.CharField(required=False),
        "query": serializers.CharField(required=False),
        "filters": serializers.JSONField(required=False),
        "process": serializers.CharField(required=False),
        "doc_class": serializers.CharField(required=False),
        "visibility": serializers.CharField(required=False),
        "visibility_override_allowed": serializers.BooleanField(required=False),
        "hybrid": serializers.JSONField(required=False),
    },
)

RAG_QUERY_RESPONSE = RagQueryResponseSerializer

RAG_QUERY_SCHEMA = {
    "request": RAG_QUERY_REQUEST,
    "responses": {200: RAG_QUERY_RESPONSE},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": (
        "Execute the production RAG graph. Headers are mapped to a ToolContext and the body is validated "
        "against the RetrieveInput contract to populate query, filters and related metadata."
    ),
    "examples": [RAG_QUERY_REQUEST_EXAMPLE, RAG_QUERY_RESPONSE_EXAMPLE],
    "extensions": RAG_QUERY_CURL,
}

RAG_UPLOAD_REQUEST_EXAMPLE = OpenApiExample(
    name="RagDocumentUploadRequest",
    summary="Document upload payload",
    description="Multipart form data containing the binary document and optional metadata JSON.",
    value={"file": "<binary>", "metadata": '{"tags":["handbook"]}'},
    request_only=True,
)

RAG_UPLOAD_RESPONSE_EXAMPLE = OpenApiExample(
    name="RagDocumentUploadResponse",
    summary="Document upload accepted",
    description="Signals that the document bytes were written to the tenant-scoped object store.",
    value={
        "status": "accepted",
        "document_id": "0f0a6d5e49e14e79bc2d0da52c5b2f4a",
        "trace_id": "f82c8a4f3f94484a8b1c9d03b1f65e10",
        "idempotent": False,
    },
    response_only=True,
)

RAG_UPLOAD_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/v1/rag/documents/upload/",
            '-H "X-Tenant-Schema: acme_prod"',
            '-H "X-Tenant-Id: acme"',
            '-H "X-Case-Id: crm-7421"',
            '-H "Idempotency-Key: 3f9d2c68-ffb0-4f7d-969e-1e6c5f8c1234"',
            '-F "file=@document.txt"',
            '-F \'metadata={"tags":["handbook"]}\'',
        ]
    )
)

RAG_UPLOAD_REQUEST = inline_serializer(
    name="RagDocumentUploadRequest",
    fields={
        "file": serializers.FileField(),
        "metadata": serializers.CharField(required=False),
    },
)

RAG_UPLOAD_RESPONSE = inline_serializer(
    name="RagDocumentUploadResponse",
    fields={
        "status": serializers.CharField(),
        "document_id": serializers.CharField(),
        "trace_id": serializers.CharField(),
        "idempotent": serializers.BooleanField(),
        "ingestion_run_id": serializers.CharField(required=False),
        "ingestion_status": serializers.CharField(required=False),
        "external_id": serializers.CharField(),
    },
)

RAG_UPLOAD_SCHEMA = {
    "request": RAG_UPLOAD_REQUEST,
    "responses": {202: RAG_UPLOAD_RESPONSE},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Upload a raw document to the tenant-scoped object store for later ingestion.",
    "examples": [RAG_UPLOAD_REQUEST_EXAMPLE, RAG_UPLOAD_RESPONSE_EXAMPLE],
    "extensions": RAG_UPLOAD_CURL,
}

RAG_INGESTION_RUN_REQUEST_EXAMPLE = OpenApiExample(
    name="RagIngestionRunRequest",
    summary="Queue ingestion run",
    description="Dispatches the ingestion pipeline for the provided document identifiers.",
    value={
        "document_ids": ["0f0a6d5e49e14e79bc2d0da52c5b2f4a"],
        "priority": "normal",
        "embedding_profile": "standard",
    },
    request_only=True,
)

RAG_INGESTION_RUN_RESPONSE_EXAMPLE = OpenApiExample(
    name="RagIngestionRunResponse",
    summary="Ingestion run queued",
    description="Confirms that the ingestion pipeline was scheduled for execution.",
    value={
        "status": "queued",
        "queued_at": "2024-01-01T12:00:00+00:00",
        "ingestion_run_id": "4b0b2834606e4e4eb3d3933e9a735cbc",
        "trace_id": "f82c8a4f3f94484a8b1c9d03b1f65e10",
        "idempotent": False,
    },
    response_only=True,
)

RAG_INGESTION_RUN_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/v1/rag/ingestion/run/",
            '-H "Content-Type: application/json"',
            '-H "X-Tenant-Schema: acme_prod"',
            '-H "X-Tenant-Id: acme"',
            '-H "X-Case-Id: crm-7421"',
            '-H "Idempotency-Key: 9c6d9b07-52c8-4fb2-8c49-0a8e3a8a1d2d"',
            '-d \'{"document_ids": ["0f0a6d5e49e14e79bc2d0da52c5b2f4a"], "priority": "normal", "embedding_profile": "standard"}\'',
        ]
    )
)

RAG_INGESTION_RUN_REQUEST = inline_serializer(
    name="RagIngestionRunRequest",
    fields={
        "document_ids": serializers.ListField(
            child=serializers.CharField(), allow_empty=False
        ),
        "priority": serializers.CharField(required=False),
        "embedding_profile": serializers.CharField(),
    },
)

RAG_INGESTION_RUN_RESPONSE = inline_serializer(
    name="RagIngestionRunResponse",
    fields={
        "status": serializers.CharField(),
        "queued_at": serializers.DateTimeField(),
        "ingestion_run_id": serializers.CharField(),
        "trace_id": serializers.CharField(),
        "idempotent": serializers.BooleanField(),
        "invalid_ids": serializers.ListField(
            child=serializers.CharField(), required=False
        ),
    },
)

RAG_INGESTION_RUN_SCHEMA = {
    "request": RAG_INGESTION_RUN_REQUEST,
    "responses": {202: RAG_INGESTION_RUN_RESPONSE},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Queue an ingestion run for previously uploaded documents.",
    "examples": [
        RAG_INGESTION_RUN_REQUEST_EXAMPLE,
        RAG_INGESTION_RUN_RESPONSE_EXAMPLE,
    ],
    "extensions": RAG_INGESTION_RUN_CURL,
}

RAG_INGESTION_STATUS_RESPONSE = inline_serializer(
    name="RagIngestionStatusResponse",
    fields={
        "run_id": serializers.CharField(),
        "status": serializers.CharField(),
        "queued_at": serializers.CharField(required=False),
        "started_at": serializers.CharField(required=False),
        "finished_at": serializers.CharField(required=False),
        "duration_ms": serializers.FloatField(required=False),
        "document_ids": serializers.ListField(
            child=serializers.CharField(), required=False
        ),
        "invalid_document_ids": serializers.ListField(
            child=serializers.CharField(), required=False
        ),
        "inserted_documents": serializers.IntegerField(required=False),
        "replaced_documents": serializers.IntegerField(required=False),
        "skipped_documents": serializers.IntegerField(required=False),
        "inserted_chunks": serializers.IntegerField(required=False),
        "trace_id": serializers.CharField(required=False),
        "embedding_profile": serializers.CharField(required=False),
        "source": serializers.CharField(required=False),
        "error": serializers.CharField(required=False),
    },
)

RAG_INGESTION_STATUS_SCHEMA = {
    "responses": {200: RAG_INGESTION_STATUS_RESPONSE},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": (
        "Return the latest ingestion run status for the current tenant and case."
    ),
}

RAG_HARD_DELETE_ADMIN_REQUEST_EXAMPLE = OpenApiExample(
    name="RagHardDeleteAdminRequest",
    summary="Trigger hard delete",
    description="Queues the hard delete Celery task for the specified documents.",
    value={
        "tenant_id": "2f0955c2-21ce-4f38-bfb0-3b690cd57834",
        "document_ids": [
            "3fbb07d0-2a5b-4b75-8ad4-5c5e8f3e1d21",
            "986cf6d5-2d8c-4b6c-98eb-3ac80f8aa84f",
        ],
        "reason": "cleanup",
        "ticket_ref": "TCK-1234",
    },
    request_only=True,
)

RAG_HARD_DELETE_ADMIN_RESPONSE_EXAMPLE = OpenApiExample(
    name="RagHardDeleteAdminResponse",
    summary="Hard delete queued",
    description="Confirms that the hard delete task was scheduled for execution.",
    value={
        "status": "queued",
        "job_id": "0d9f7ac1-0b07-4b7c-98b7-7237f8b9df5b",
        "trace_id": "c8b7e6c430864d6aa6c66de8f9ad6d47",
        "documents_requested": 2,
        "idempotent": True,
    },
    response_only=True,
)

RAG_HARD_DELETE_ADMIN_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/ai/rag/admin/hard-delete/",
            '-H "Content-Type: application/json"',
            '-H "X-Internal-Key: ops-service"',
            '-d \'{"tenant_id": "2f0955c2-21ce-4f38-bfb0-3b690cd57834", "document_ids": ["3fbb07d0-2a5b-4b75-8ad4-5c5e8f3e1d21"], "reason": "cleanup", "ticket_ref": "TCK-1234"}\'',
        ]
    )
)

RAG_HARD_DELETE_ADMIN_REQUEST = inline_serializer(
    name="RagHardDeleteAdminRequest",
    fields={
        "tenant_id": serializers.CharField(),
        "document_ids": serializers.ListField(
            child=serializers.CharField(), allow_empty=False
        ),
        "reason": serializers.CharField(),
        "ticket_ref": serializers.CharField(),
        "operator_label": serializers.CharField(required=False),
    },
)

RAG_HARD_DELETE_ADMIN_RESPONSE = inline_serializer(
    name="RagHardDeleteAdminResponse",
    fields={
        "status": serializers.CharField(),
        "job_id": serializers.CharField(),
        "trace_id": serializers.CharField(),
        "documents_requested": serializers.IntegerField(),
        "idempotent": serializers.BooleanField(),
    },
)

RAG_HARD_DELETE_ADMIN_SCHEMA = {
    "request": RAG_HARD_DELETE_ADMIN_REQUEST,
    "responses": {202: RAG_HARD_DELETE_ADMIN_RESPONSE},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Internal endpoint that schedules the rag.hard_delete task after an admin or service key authorisation.",
    "examples": [
        RAG_HARD_DELETE_ADMIN_REQUEST_EXAMPLE,
        RAG_HARD_DELETE_ADMIN_RESPONSE_EXAMPLE,
    ],
    "extensions": RAG_HARD_DELETE_ADMIN_CURL,
}


CRAWLER_RUN_REQUEST_EXAMPLE = OpenApiExample(
    name="CrawlerRunRequest",
    summary="Trigger live crawl with snapshot",
    description="Run the crawler LangGraph against one or more origins with optional snapshot capture.",
    value={
        "workflow_id": "crawler-demo",
        "mode": "live",
        "collection_id": "6d6fba7c-1c62-4f0a-8b8b-7efb4567a0aa",
        "tags": ["handbook", "hr"],
        "limits": {"max_document_bytes": 2048},
        "snapshot": {"enabled": True, "label": "debug"},
        "review": "required",
        "shadow_mode": True,
        "dry_run": True,
        "origins": [
            {
                "url": "https://example.com/docs/handbook",
                "title": "Employee Handbook",
                "language": "de",
                "provider": "web",
                "fetch": True,
            }
        ],
    },
    request_only=True,
)


CRAWLER_RUN_RESPONSE_EXAMPLE = OpenApiExample(
    name="CrawlerRunResponse",
    summary="Crawler graph outcome",
    description="Summarises the node decisions, control state, telemetry and snapshot outcome for each origin.",
    value={
        "workflow_id": "crawler-demo",
        "mode": "live",
        "collection_id": "6d6fba7c-1c62-4f0a-8b8b-7efb4567a0aa",
        "origins": [
            {
                "origin": "https://example.com/docs/handbook",
                "provider": "web",
                "document_id": "0f3b0d9c4f2a4bf6a4e46cf9f90c8d21",
                "result": {"decision": "upsert", "reason": "ingest_ready"},
                "control": {"shadow_mode": True, "review": "required"},
                "ingest_action": "upsert",
                "gating_score": 1.0,
                "graph_run_id": "2b6a9bb4f0dc4635a663274807c33139",
                "state": {
                    "workflow_id": "crawler-demo",
                    "document_id": "0f3b0d9c4f2a4bf6a4e46cf9f90c8d21",
                    "origin_uri": "https://example.com/docs/handbook",
                    "provider": "web",
                    "content_hash": "3c21bd0b8f1a4a1b9c44d0fd57a2d9ef",
                    "tags": ["handbook", "hr"],
                    "snapshot_requested": True,
                    "snapshot_label": "debug",
                },
                "collection_id": "6d6fba7c-1c62-4f0a-8b8b-7efb4567a0aa",
                "review": "required",
                "dry_run": True,
                "ingestion_run_id": "4b0b2834606e4e4eb3d3933e9a735cbc",
            }
        ],
        "transitions": [
            {
                "origin": "https://example.com/docs/handbook",
                "transitions": {
                    "crawler.fetch": {
                        "decision": "fetched",
                        "attributes": {"status_code": 200},
                    },
                    "crawler.ingest_decision": {"decision": "upsert"},
                },
            }
        ],
        "telemetry": [
            {
                "origin": "https://example.com/docs/handbook",
                "provider": "web",
                "fetch_used": True,
                "http_status": 200,
                "fetched_bytes": 1280,
                "media_type_effective": "text/html",
                "fetch_elapsed": 0.44,
                "fetch_retries": 1,
                "fetch_retry_reason": "retry_after_backoff",
                "fetch_backoff_total_ms": 200.0,
                "snapshot_requested": True,
                "snapshot_label": "debug",
                "tags": ["handbook", "hr"],
            }
        ],
        "errors": [],
        "idempotent": False,
    },
    response_only=True,
)


CRAWLER_RUN_REQUEST = inline_serializer(
    name="CrawlerRunRequest",
    fields={
        "workflow_id": serializers.CharField(required=False),
        "mode": serializers.ChoiceField(choices=["live", "manual"], required=False),
        "origins": serializers.ListField(child=serializers.JSONField(), required=False),
        "origin_url": serializers.CharField(required=False),
        "provider": serializers.CharField(required=False),
        "document_id": serializers.CharField(required=False),
        "title": serializers.CharField(required=False),
        "language": serializers.CharField(required=False),
        "content": serializers.CharField(required=False, allow_null=True),
        "content_type": serializers.CharField(required=False),
        "fetch": serializers.BooleanField(required=False),
        "collection_id": serializers.CharField(required=False),
        "limits": serializers.JSONField(required=False),
        "snapshot": serializers.JSONField(required=False),
        "snapshot_label": serializers.CharField(required=False, allow_null=True),
        "tags": serializers.ListField(child=serializers.CharField(), required=False),
        "shadow_mode": serializers.BooleanField(required=False),
        "dry_run": serializers.BooleanField(required=False),
        "review": serializers.CharField(required=False),
        "manual_review": serializers.CharField(required=False),
        "force_retire": serializers.BooleanField(required=False),
        "recompute_delta": serializers.BooleanField(required=False),
        "max_document_bytes": serializers.IntegerField(required=False),
    },
)

CRAWLER_RUN_RESPONSE = inline_serializer(
    name="CrawlerRunResponse",
    fields={
        "workflow_id": serializers.CharField(required=False, allow_null=True),
        "mode": serializers.CharField(),
        "collection_id": serializers.CharField(required=False, allow_null=True),
        "origins": serializers.ListField(child=serializers.JSONField(), required=True),
        "transitions": serializers.ListField(
            child=serializers.JSONField(), required=True
        ),
        "telemetry": serializers.ListField(
            child=serializers.JSONField(), required=True
        ),
        "errors": serializers.ListField(child=serializers.JSONField(), required=False),
        "idempotent": serializers.BooleanField(),
    },
)

CRAWLER_RUN_SCHEMA = {
    "request": CRAWLER_RUN_REQUEST,
    "responses": {200: CRAWLER_RUN_RESPONSE},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Execute the crawler ingestion LangGraph with manual or live fetch content for debugging.",
    "examples": [CRAWLER_RUN_REQUEST_EXAMPLE, CRAWLER_RUN_RESPONSE_EXAMPLE],
}


PING_SCHEMA = {
    "responses": {200: PingResponseSerializer},
    "error_statuses": RATE_LIMIT_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Simple heartbeat used by agents and operators to verify that AI Core is responsive.",
    "examples": [PING_RESPONSE_EXAMPLE],
    "extensions": PING_CURL,
}


INTAKE_SCHEMA = {
    "request": IntakeRequestSerializer,
    "responses": {200: IntakeResponseSerializer},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Primary entrypoint for LangGraph orchestrated intake conversations.",
    "examples": [INTAKE_REQUEST_EXAMPLE, INTAKE_RESPONSE_EXAMPLE],
    "extensions": INTAKE_CURL,
}


class _BaseAgentView(DeprecationHeadersMixin, APIView):
    # Authentication and permissions are inherited from REST_FRAMEWORK defaults
    # (SessionAuthentication + IsAuthenticated)
    pass


class _PingBase(_BaseAgentView):
    def get(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error
        response = Response({"ok": True})
        return apply_std_headers(response, meta)


class PingViewV1(_PingBase):
    """Lightweight endpoint used to verify AI Core availability."""

    @default_extend_schema(**PING_SCHEMA)
    def get(self, request: Request) -> Response:
        return super().get(request)


class _GraphView(_BaseAgentView):
    graph_name: str | None = None

    def post(self, request: Request) -> Response:
        if not (
            settings.DEBUG
            or getattr(settings, "TESTING", False)
            or os.environ.get("PYTEST_CURRENT_TEST")
        ):
            return _error_response(
                "Not found.",
                "graph_endpoint_disabled",
                status.HTTP_404_NOT_FOUND,
            )
        if not self.graph_name:
            raise NotImplementedError("graph_name must be configured on subclasses")
        meta, error = _prepare_request(request)
        if error:
            return error

        request.graph_name = self.graph_name
        response = _run_graph(request)
        return apply_std_headers(response, meta)


def _run_graph(request: Request) -> Response:  # type: ignore[no-untyped-def]
    """Compatibility wrapper used by tests to monkeypatch graph execution."""
    return services.execute_graph(request)


class IntakeViewV1(_GraphView):
    """Entry point for the agent intake workflow."""

    api_deprecated = True
    api_deprecation_id = "info-intake"
    graph_name = "info_intake"

    @default_extend_schema(**INTAKE_SCHEMA)
    def post(self, request: Request) -> Response:
        return super().post(request)


def _serialise_json_value(value: object) -> object:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Mapping):
        serialised: dict[str, object] = {}
        for key, val in value.items():
            if isinstance(key, uuid.UUID):
                key_str = str(key)
            elif isinstance(key, str):
                key_str = key
            else:
                key_str = str(key)
            serialised[key_str] = _serialise_json_value(val)
        return serialised
    if isinstance(value, (list, tuple, set)):
        return [_serialise_json_value(item) for item in value]
    return value


def _resolve_lifecycle_store() -> object | None:
    """Return the document lifecycle store used for baseline lookups."""

    store = globals().get("DOCUMENTS_LIFECYCLE_STORE")
    if store is not None:
        return store
    try:
        from documents import api as documents_api  # local import to avoid cycles
    except Exception:  # pragma: no cover - defensive import guard
        return None
    return getattr(documents_api, "DEFAULT_LIFECYCLE_STORE", None)


def _normalise_rag_response(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the payload projected onto the public RAG response contract."""

    allowed_top_level = {
        "answer",
        "prompt_version",
        "retrieval",
        "snippets",
        "diagnostics",
    }
    allowed_retrieval = {
        "alpha",
        "min_sim",
        "top_k_effective",
        "matches_returned",
        "max_candidates_effective",
        "vector_candidates",
        "lexical_candidates",
        "deleted_matches_blocked",
        "visibility_effective",
        "took_ms",
        "routing",
    }
    allowed_routing = {"profile", "vector_space_id"}

    projected: dict[str, object] = {}
    diagnostics: dict[str, object] = (
        dict(payload["diagnostics"])
        if isinstance(payload.get("diagnostics"), Mapping)
        else {}
    )

    top_level_extras = {
        key: value for key, value in payload.items() if key not in allowed_top_level
    }

    for key in allowed_top_level:
        if key not in payload:
            continue
        if key == "diagnostics":
            continue
        if key != "retrieval":
            projected[key] = _serialise_json_value(payload[key])
            continue

        value = payload[key]
        if not isinstance(value, Mapping):
            projected[key] = _serialise_json_value(value)
            continue

        retrieval_dict = dict(value)
        retrieval_projected: dict[str, object] = {}
        retrieval_extras: dict[str, object] = {}

        for retrieval_key, retrieval_value in retrieval_dict.items():
            if retrieval_key not in allowed_retrieval:
                retrieval_extras[retrieval_key] = retrieval_value
                continue

            if retrieval_key == "took_ms" and isinstance(retrieval_value, float):
                retrieval_projected[retrieval_key] = int(retrieval_value)
            else:
                retrieval_projected[retrieval_key] = retrieval_value

        if "routing" in retrieval_dict and isinstance(
            retrieval_dict["routing"], Mapping
        ):
            routing_dict = dict(retrieval_dict["routing"])
            routing_projected_nested: dict[str, object] = {}
            for routing_key, routing_value in routing_dict.items():
                if routing_key in allowed_routing:
                    routing_projected_nested[routing_key] = routing_value
                else:
                    if "routing" not in retrieval_extras:
                        retrieval_extras["routing"] = {}
                    retrieval_extras["routing"][routing_key] = routing_value
            retrieval_projected["routing"] = routing_projected_nested
        elif "routing" in retrieval_dict:
            retrieval_projected["routing"] = retrieval_dict["routing"]

        if retrieval_extras:
            diagnostics["retrieval"] = _serialise_json_value(retrieval_extras)

        projected[key] = _serialise_json_value(retrieval_projected)

    if top_level_extras:
        diagnostics["response"] = _serialise_json_value(top_level_extras)

    if diagnostics:
        projected["diagnostics"] = diagnostics

    return projected


@require_POST
def crawl_selected(request):
    """Handle crawl selected requests from the RAG tools page (HTMX).

    Acts as the L2 Proxy/Adapter to the L3 CrawlerManager.
    """
    try:
        # Wrap request to use shared preparation logic
        from rest_framework.request import Request as DRFRequest
        from django.http import HttpResponse

        # DRF Request requires parsers/authenticators to be set if we want full functionality,
        # but _prepare_request only uses headers/META/user/auth which proxies to underlying request.
        drf_request = DRFRequest(request)

        meta, error = _prepare_request(drf_request)
        if error:
            # For HTMX, ideally we return a 4xx HTML fragment, but simplistic text is fine for now
            return HttpResponse(
                f"Error: {error.data.get('detail', 'Unknown error')}",
                status=error.status_code,
            )

        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return HttpResponse("Invalid JSON body", status=400)

        urls = data.get("urls")
        collection_id = data.get("collection_id", "crawler-demo")
        mode = data.get("mode", "live")
        workflow_id = data.get("workflow_id", "crawler-demo")

        # BREAKING CHANGE (Option A - Strict Separation):
        # _prepare_request only builds scope_context, we need to add business_context
        # Extract business IDs from request headers
        case_id = drf_request.headers.get("X-Case-ID", "").strip() or None
        from ai_core.contracts.business import BusinessContext

        business_context = BusinessContext(
            case_id=case_id,
            workflow_id=workflow_id,
            collection_id=collection_id,
        )
        meta["business_context"] = business_context.model_dump(
            mode="json", exclude_none=True
        )
        tool_context = tool_context_from_meta(meta)
        meta["tool_context"] = tool_context.model_dump(mode="json", exclude_none=True)

        if not urls:
            return HttpResponse("No URLs provided", status=400)

        crawler_payload = {
            "workflow_id": workflow_id,
            "mode": mode,
            "origins": [{"url": url} for url in urls],
            "collection_id": collection_id,
        }

        try:
            request_model = CrawlerRunRequest.model_validate(crawler_payload)
        except ValidationError as exc:
            return HttpResponse(f"Validation Error: {exc}", status=400)

        # L2 -> L3 Dispatch
        from crawler.manager import CrawlerManager

        manager = CrawlerManager()

        try:
            result = manager.dispatch_crawl_request(request_model, meta)
        except Exception as exc:
            logger.exception("crawl_dispatch_failed")
            return HttpResponse(f"Dispatch Error: {str(exc)}", status=500)

        # Return HTML status for HTMX
        count = result.get("count", 0)

        # Simple HTML feedback
        html_response = f"""
        <div class="p-4 mb-4 text-sm text-green-800 rounded-lg bg-green-50 dark:bg-gray-800 dark:text-green-400" role="alert">
          <span class="font-medium">Success!</span> Queued {count} URL(s) for ingestion.
          <ul class="mt-1.5 list-disc list-inside">
             {"".join(f"<li>{task['url']}</li>" for task in result.get("tasks", []))}
          </ul>
        </div>
        """
        return HttpResponse(html_response)

    except Exception as e:
        logger.exception("crawl_selected unhandled exception")
        return HttpResponse(f"Internal Error: {str(e)}", status=500)


class RagQueryViewV1(_GraphView):
    """Execute the production retrieval augmented generation graph."""

    graph_name = "rag.default"

    @default_extend_schema(**RAG_QUERY_SCHEMA)
    def post(self, request: Request) -> Response:
        response = super().post(request)

        # Only validate successful responses returned from the graph runner.
        if 200 <= response.status_code < 300:
            graph_payload = response.data
            if isinstance(graph_payload, Mapping):
                graph_payload = _normalise_rag_response(graph_payload)
            serializer = RagQueryResponseSerializer(data=graph_payload)
            try:
                serializer.is_valid(raise_exception=True)
            except serializers.ValidationError:
                try:
                    logger.warning(
                        "rag.response.validation_failed",
                        extra={
                            "errors": getattr(serializer, "errors", None),
                            "keys": (
                                list(graph_payload.keys())
                                if isinstance(graph_payload, dict)
                                else None
                            ),
                        },
                    )
                except Exception:
                    pass
                details = (
                    serializer.errors
                    if isinstance(serializer.errors, Mapping)
                    else None
                )
                payload = build_tool_error_payload(
                    message="Response payload failed validation.",
                    status_code=status.HTTP_400_BAD_REQUEST,
                    code="invalid_response",
                    details=details,
                )
                return Response(payload, status=status.HTTP_400_BAD_REQUEST)
            except Exception:
                try:
                    logger.warning(
                        "rag.response.validation_failed",
                        extra={
                            "errors": getattr(serializer, "errors", None),
                            "keys": (
                                list(graph_payload.keys())
                                if isinstance(graph_payload, dict)
                                else None
                            ),
                        },
                    )
                except Exception:
                    pass
                raise
            response.data = serializer.validated_data
            bind_log_context(response_contract="rag.v2")
            if isinstance(getattr(request, "log_context", None), dict):
                request.log_context["response_contract"] = "rag.v2"

        return response


class RagUploadView(APIView):
    """Handle multipart document uploads for ingestion pipelines."""

    if settings.TESTING:
        authentication_classes: list = []
        permission_classes: list = []

    @default_extend_schema(**RAG_UPLOAD_SCHEMA)
    def post(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

        # BREAKING CHANGE (Option A - Strict Separation):
        # scope_context is infrastructure only (ScopeContext).
        # Business IDs like case_id should be in a separate business_context dict.
        # For now, check and set case_id in request.META instead.
        case_id_from_header = (
            request.headers.get("X-Case-ID") or request.META.get(META_CASE_ID_KEY) or ""
        ).strip()

        # Fix for Silent RAG Failure (Finding #22):
        # In DEV/DEBUG mode, default missing case_id to the dev default.
        # so that uploads from Workbench are visible to the RAG Chat.
        if settings.DEBUG and not case_id_from_header:
            logger.info(
                "view.rag_upload.defaulting_case_id",
                extra={"assigned_case_id": DEV_DEFAULT_CASE_ID},
            )
            case_id_from_header = DEV_DEFAULT_CASE_ID
            request.META["META_CASE_ID"] = case_id_from_header

        # Add business context to meta for handle_document_upload
        from ai_core.contracts.business import BusinessContext

        workflow_id_from_header = (
            request.headers.get("X-Workflow-ID")
            or request.META.get(META_WORKFLOW_ID_KEY)
            or ""
        ).strip()
        collection_id_from_header = (
            request.headers.get("X-Collection-ID")
            or request.META.get(META_COLLECTION_ID_KEY)
            or ""
        ).strip()

        business_context = BusinessContext(
            case_id=case_id_from_header,
            workflow_id=workflow_id_from_header or case_id_from_header,
            collection_id=collection_id_from_header or None,
        )
        meta["business_context"] = business_context.model_dump(
            mode="json", exclude_none=True
        )
        tool_context = tool_context_from_meta(meta)
        meta["tool_context"] = tool_context.model_dump(mode="json", exclude_none=True)

        content_type = request.headers.get("Content-Type", "")
        if content_type:
            content_type = content_type.split(";")[0].strip().lower()
        if not content_type.startswith("multipart/"):
            return _error_response(
                "Request payload must be encoded as multipart/form-data.",
                "unsupported_media_type",
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )

        upload = request.FILES.get("file")
        if upload is None:
            return _error_response(
                "File form part is required for document uploads.",
                "missing_file",
                status.HTTP_400_BAD_REQUEST,
            )

        metadata_raw = request.data.get("metadata")
        idempotency_key = request.headers.get(IDEMPOTENCY_KEY_HEADER)

        response = services.handle_document_upload(
            upload,
            metadata_raw,
            meta,
            idempotency_key,
        )

        return apply_std_headers(response, meta)


class RagIngestionRunView(APIView):
    """Queue ingestion runs for previously uploaded documents."""

    @default_extend_schema(**RAG_INGESTION_RUN_SCHEMA)
    def post(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

        # BREAKING CHANGE (Option A): Extract business IDs from business_context
        context = tool_context_from_meta(meta)

        # Fix for Silent RAG Failure (Finding #22): Default case_id for async ingestion too
        if settings.DEBUG and not context.business.case_id:
            updated_business = context.business.model_copy(
                update={"case_id": DEV_DEFAULT_CASE_ID}
            )
            context = context.model_copy(update={"business": updated_business})
            meta["business_context"] = updated_business.model_dump(
                mode="json", exclude_none=True
            )
            meta["tool_context"] = context.model_dump(mode="json", exclude_none=True)

        idempotency_key = request.headers.get(IDEMPOTENCY_KEY_HEADER)
        if isinstance(request.data, Mapping):
            payload = dict(request.data)
        else:
            payload = dict(getattr(request, "data", {}) or {})

        if (
            context.business.collection_id
            and not payload.get("collection_id")
            and payload.get("collection_ids") in (None, "")
        ):
            payload["collection_id"] = context.business.collection_id

        response = services.start_ingestion_run(payload, meta, idempotency_key)

        return apply_std_headers(response, meta)


class RagIngestionStatusView(APIView):
    """Expose status information about recent ingestion runs."""

    @default_extend_schema(**RAG_INGESTION_STATUS_SCHEMA)
    def get(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

        # BREAKING CHANGE (Option A): Extract business and infrastructure IDs
        context = tool_context_from_meta(meta)
        tenant_id = context.scope.tenant_id
        case_id = context.business.case_id
        latest = get_latest_ingestion_run(tenant_id, case_id)
        if not latest:
            response = _error_response(
                "No ingestion runs recorded for the current tenant/case.",
                "ingestion_status_not_found",
                status.HTTP_404_NOT_FOUND,
            )
            return apply_std_headers(response, meta)

        response_payload: dict[str, object] = {}
        allowed_keys = {
            "run_id",
            "status",
            "queued_at",
            "started_at",
            "finished_at",
            "duration_ms",
            "document_ids",
            "invalid_document_ids",
            "inserted_documents",
            "replaced_documents",
            "skipped_documents",
            "inserted_chunks",
            "trace_id",
            "embedding_profile",
            "source",
            "error",
        }
        for key in allowed_keys:
            value = latest.get(key)
            if value is None:
                continue
            if key in {"document_ids", "invalid_document_ids"} and not isinstance(
                value, list
            ):
                try:
                    value = list(value)
                except TypeError:
                    continue
            response_payload[key] = value

        if "run_id" not in response_payload or "status" not in response_payload:
            response = _error_response(
                "No ingestion runs recorded for the current tenant/case.",
                "ingestion_status_not_found",
                status.HTTP_404_NOT_FOUND,
            )
            return apply_std_headers(response, meta)

        from customers.tenant_context import TenantContext

        tenant_obj = TenantContext.resolve_identifier(context.scope.tenant_id)
        if tenant_obj is not None:
            # BREAKING CHANGE (Option A): case_id from business_context
            case_obj = Case.objects.filter(
                tenant=tenant_obj, external_id=context.business.case_id
            ).first()
            if case_obj is not None:
                response_payload["case_status"] = case_obj.status
                if case_obj.phase:
                    response_payload["case_phase"] = case_obj.phase
                latest_case_event = (
                    case_obj.events.filter(event_type__startswith="ingestion_run_")
                    .order_by("-created_at")
                    .first()
                )
                if latest_case_event is not None:
                    response_payload["latest_case_event"] = {
                        "event_type": latest_case_event.event_type,
                        "created_at": latest_case_event.created_at.isoformat(),
                    }
                    if latest_case_event.trace_id:
                        response_payload["latest_case_event"][
                            "trace_id"
                        ] = latest_case_event.trace_id
                    if latest_case_event.payload:
                        response_payload["latest_case_event"][
                            "payload"
                        ] = latest_case_event.payload
                    if latest_case_event.ingestion_run_id:
                        response_payload["latest_case_event"][
                            "ingestion_run_id"
                        ] = latest_case_event.ingestion_run.run_id

        response = Response(response_payload, status=status.HTTP_200_OK)
        processed_response = apply_std_headers(response, meta)
        idempotency_key_value = context.scope.idempotency_key
        header_idempotency = request.headers.get(IDEMPOTENCY_KEY_HEADER)
        if not header_idempotency:
            meta_header_key = "HTTP_" + IDEMPOTENCY_KEY_HEADER.upper().replace("-", "_")
            raw_meta_header = request.META.get(meta_header_key)
            if isinstance(raw_meta_header, str) and raw_meta_header.strip():
                header_idempotency = raw_meta_header
        resolved_idempotency = idempotency_key_value or (
            header_idempotency.strip() if isinstance(header_idempotency, str) else None
        )
        if resolved_idempotency:
            processed_response.headers[IDEMPOTENCY_KEY_HEADER] = resolved_idempotency
            header_key_lower = IDEMPOTENCY_KEY_HEADER.lower()
            processed_response._headers[header_key_lower] = (
                IDEMPOTENCY_KEY_HEADER,
                resolved_idempotency,
            )
            if not idempotency_key_value:
                updated_scope = context.scope.model_copy(
                    update={"idempotency_key": resolved_idempotency}
                )
                updated_context = context.model_copy(update={"scope": updated_scope})
                meta["scope_context"] = updated_scope.model_dump(
                    mode="json", exclude_none=True
                )
                meta["tool_context"] = updated_context.model_dump(
                    mode="json", exclude_none=True
                )
        return processed_response


class RagHardDeleteAdminView(APIView):
    """Trigger the asynchronous hard delete task for administrators."""

    authentication_classes = [SessionAuthentication]
    permission_classes: list = []

    @default_extend_schema(**RAG_HARD_DELETE_ADMIN_SCHEMA)
    def post(self, request: Request) -> Response:
        payload = request.data if isinstance(request.data, Mapping) else {}

        try:
            request_data = RagHardDeleteAdminRequest.model_validate(payload)
        except ValidationError as exc:
            error = exc.errors()[0] if exc.errors() else None
            message = error.get("msg") if error else str(exc)
            code = error.get("type") if error else "validation_error"
            if code not in {
                "invalid_tenant_id",
                "invalid_document_ids",
                "invalid_reason",
                "invalid_ticket_ref",
            }:
                code = "validation_error"
            return _error_response(message, code, status.HTTP_400_BAD_REQUEST)

        tenant_id = request_data.tenant_id
        document_ids = request_data.document_ids
        reason = request_data.reason
        ticket_ref = request_data.ticket_ref
        operator_label = request_data.operator_label

        tenant_schema = None
        header_schema = request.headers.get(X_TENANT_SCHEMA_HEADER)
        if isinstance(header_schema, str) and header_schema.strip():
            tenant_schema = header_schema.strip()
        elif request_data.tenant_schema:
            tenant_schema = request_data.tenant_schema

        trace_id = uuid4().hex
        bind_log_context(trace_id=trace_id, tenant_id=tenant_id)

        actor = _resolve_hard_delete_actor(request, operator_label)

        scope_context = ScopeContext.model_validate(
            {
                "tenant_id": tenant_id,
                "trace_id": trace_id,
                "invocation_id": uuid4().hex,
                "run_id": uuid4().hex,
                "tenant_schema": tenant_schema,
                "idempotency_key": request.headers.get(IDEMPOTENCY_KEY_HEADER),
            }
        )
        from ai_core.contracts.business import BusinessContext

        business_context = BusinessContext()
        tool_context = scope_context.to_tool_context(business=business_context)
        meta = {
            "scope_context": scope_context.model_dump(mode="json"),
            "business_context": business_context.model_dump(
                mode="json", exclude_none=True
            ),
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }
        state = {
            "tenant_id": tenant_id,
            "document_ids": document_ids,
            "reason": reason,
            "ticket_ref": ticket_ref,
            "tenant_schema": tenant_schema,
            "trace_id": trace_id,
        }

        async_result = hard_delete.delay(
            state,
            meta,
            actor=actor,
        )
        idempotent = bool(request.headers.get(IDEMPOTENCY_KEY_HEADER))
        response_payload = {
            "status": "queued",
            "job_id": getattr(async_result, "id", None),
            "trace_id": trace_id,
            "documents_requested": len(document_ids),
            "idempotent": idempotent,
        }
        response = Response(response_payload, status=status.HTTP_202_ACCEPTED)
        return apply_std_headers(response, meta)


class CrawlerIngestionRunnerView(APIView):
    """Expose the crawler ingestion LangGraph for manual debugging runs."""

    @default_extend_schema(**CRAWLER_RUN_SCHEMA)
    def post(self, request: Request) -> Response:
        if not (
            settings.DEBUG
            or getattr(settings, "TESTING", False)
            or os.environ.get("PYTEST_CURRENT_TEST")
        ):
            return _error_response(
                "Not found.",
                "crawler_runner_disabled",
                status.HTTP_404_NOT_FOUND,
            )
        meta, error = _prepare_request(request)
        if error:
            return error

        payload = request.data if isinstance(request.data, Mapping) else {}
        try:
            request_model = CrawlerRunRequest.model_validate(payload)
        except ValidationError as exc:
            return _error_response(
                services._format_validation_error(exc),
                "invalid_request",
                status.HTTP_400_BAD_REQUEST,
            )

        lifecycle_store = _resolve_lifecycle_store()
        # Note: graph_factory is always None now (Finding #5 fix removed module-level import)
        # Tests that need custom graphs should patch run_crawler_runner directly
        try:
            result = run_crawler_runner(
                meta=meta,
                request_model=request_model,
                lifecycle_store=lifecycle_store,
                graph_factory=None,
            )
        except CrawlerRunError as exc:
            payload = build_tool_error_payload(
                message=str(exc),
                status_code=exc.status_code,
                code=exc.code,
                details=exc.details,
            )
            response = Response(payload, status=exc.status_code)
            return apply_response_headers(response, meta)
        except ValueError as exc:
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )

        payload = result.payload
        if result.status_code >= 400 and isinstance(payload, Mapping):
            message = str(
                payload.get("reason") or payload.get("detail") or "Crawler failed."
            )
            code = str(payload.get("code") or "crawler_error")
            details = {
                k: v
                for k, v in payload.items()
                if k not in {"code", "detail", "reason"}
            }
            payload = build_tool_error_payload(
                message=message,
                status_code=result.status_code,
                code=code,
                details=details or None,
            )
        response = Response(payload, status=result.status_code)
        return apply_response_headers(response, meta, result.idempotency_key)


ping_v1 = PingViewV1.as_view()

intake_v1 = IntakeViewV1.as_view()

rag_query_v1 = RagQueryViewV1.as_view()
rag_query = rag_query_v1

rag_upload_v1 = RagUploadView.as_view()
rag_upload = rag_upload_v1

rag_ingestion_run_v1 = RagIngestionRunView.as_view()
rag_ingestion_run = rag_ingestion_run_v1

rag_ingestion_status_v1 = RagIngestionStatusView.as_view()
rag_ingestion_status = rag_ingestion_status_v1


rag_hard_delete_admin = RagHardDeleteAdminView.as_view()
crawler_runner = CrawlerIngestionRunnerView.as_view()
crawl_selected = crawl_selected

CHECKPOINTER = FileCheckpointer()
