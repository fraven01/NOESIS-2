from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from importlib import import_module
from uuid import uuid4

from django.conf import settings
from django.db import connection
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest

from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    META_CASE_ID_KEY,
    META_KEY_ALIAS_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    META_TRACE_ID_KEY,
    X_CASE_ID_HEADER,
    X_KEY_ALIAS_HEADER,
    X_TENANT_ID_HEADER,
    X_TENANT_SCHEMA_HEADER,
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
    NeedsResponseSerializer,
    PingResponseSerializer,
    RagQueryResponseSerializer,
    ScopeResponseSerializer,
    SysDescResponseSerializer,
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
from ai_core.graph.adapters import module_runner
from ai_core.graph.core import GraphRunner
from ai_core.graph.registry import get as get_graph_runner, register as register_graph
from ai_core.graphs import (
    info_intake,
    needs_mapping,
    scope_check,
    system_description,
)  # noqa: F401

# Import graphs so they are available via module globals for Legacy views.
# This enables tests to monkeypatch e.g. `views.info_intake` directly and
# allows _GraphView.get_graph to resolve from globals() without importing.
try:  # pragma: no cover - exercised indirectly via tests
    info_intake = import_module("ai_core.graphs.info_intake")
    scope_check = import_module("ai_core.graphs.scope_check")
    needs_mapping = import_module("ai_core.graphs.needs_mapping")
    system_description = import_module("ai_core.graphs.system_description")
except Exception:  # defensive: don't break module import if graphs change
    # Fallback to lazy import via _GraphView.get_graph when not present.
    pass


from . import services

# Re-export normalize_meta so tests can monkeypatch via ai_core.views
from ai_core.graph.schemas import normalize_meta as normalize_meta  # noqa: F401
from .ingestion import run_ingestion as run_ingestion  # re-export for tests
from .ingestion import partition_document_ids as partition_document_ids  # test hook
from .services import CHECKPOINTER as CHECKPOINTER  # re-export for tests
from .rag.ingestion_contracts import (
    resolve_ingestion_profile as resolve_ingestion_profile,  # test hook
)
from .infra import rate_limit
from .ingestion_status import (
    get_latest_ingestion_run,
)
from .rag.hard_delete import hard_delete
from .schemas import (
    RagHardDeleteAdminRequest,
)
from pydantic import ValidationError

from .infra.resp import apply_std_headers


logger = get_logger(__name__)

logger.info(
    "module_loaded",
    extra={"module": __name__, "path": str(Path(__file__).resolve())},
)


def assert_case_active(tenant: str, case_id: str) -> None:
    """Placeholder for future case activity checks."""
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
    import hashlib

    return hashlib.sha256(buffer).hexdigest()


def _resolve_tenant_id(request: HttpRequest) -> str | None:
    """Derive the active tenant schema for the current request.

    Resolution order (first non-public wins):
    1) Request-bound tenant object (django-tenants)
    2) Connection schema (django-tenants current schema)
    3) Explicit header-backed attribute populated by TenantSchemaMiddleware
    """

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")

    def _normalise(schema: object | None) -> str | None:
        if schema is None:
            return None
        if not isinstance(schema, str):
            schema = str(schema)
        candidate = schema.strip()
        if not candidate or candidate == public_schema:
            return None
        return candidate

    # 1) Tenant object from django-tenants (domain-based routing)
    tenant_obj = getattr(request, "tenant", None)
    resolved = _normalise(getattr(tenant_obj, "schema_name", None))
    if resolved:
        return resolved

    # 2) Fallback: current connection schema
    resolved = _normalise(getattr(connection, "schema_name", None))
    if resolved:
        return resolved

    # 3) Explicit header provided by TenantSchemaMiddleware
    return _normalise(getattr(request, "tenant_schema", None))


KEY_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _error_response(detail: str, code: str, status_code: int) -> Response:
    """Return a standardised error payload."""

    return Response({"detail": detail, "code": code}, status=status_code)


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
    tenant_header = request.headers.get(X_TENANT_ID_HEADER)
    case_id = (request.headers.get(X_CASE_ID_HEADER) or "").strip()
    key_alias_header = request.headers.get(X_KEY_ALIAS_HEADER)

    tenant_schema = _resolve_tenant_id(request)
    if not tenant_schema:
        return None, _error_response(
            "Tenant schema could not be resolved from headers.",
            "tenant_not_found",
            status.HTTP_400_BAD_REQUEST,
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

    schema_header = request.headers.get(X_TENANT_SCHEMA_HEADER)
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

    if not CASE_ID_RE.fullmatch(case_id):
        return None, _error_response(
            "Case header is required and must use the documented format.",
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

    trace_id = uuid4().hex
    assert_case_active(tenant_id, case_id)
    meta = {
        "tenant_id": tenant_id,
        "tenant_schema": tenant_schema,
        "case_id": case_id,
        "trace_id": trace_id,
    }
    if key_alias:
        meta["key_alias"] = key_alias

    request.META[META_TRACE_ID_KEY] = trace_id
    request.META[META_CASE_ID_KEY] = case_id
    request.META[META_TENANT_ID_KEY] = tenant_id
    request.META[META_TENANT_SCHEMA_KEY] = tenant_schema
    if key_alias:
        request.META[META_KEY_ALIAS_KEY] = key_alias
    else:
        request.META.pop(META_KEY_ALIAS_KEY, None)

    log_context = {
        "trace_id": trace_id,
        "case_id": case_id,
        "tenant": tenant_id,
        "key_alias": key_alias,
    }
    request.log_context = log_context
    bind_log_context(**log_context)
    return meta, None


LEGACY_DEPRECATION_ID = "ai-core-legacy"


def _legacy_schema_kwargs(base_kwargs: dict[str, object]) -> dict[str, object]:
    legacy_kwargs = dict(base_kwargs)
    legacy_kwargs["deprecated"] = True
    return legacy_kwargs


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

SCOPE_REQUEST_EXAMPLE = OpenApiExample(
    name="ScopeRequest",
    summary="Scope payload",
    description="Provide the captured scope inputs that should be validated.",
    value={
        "prompt": "Zeige den Projektstatus",
        "metadata": {"project": "acme-kickoff"},
        "scope": "discovery",
    },
    request_only=True,
)

SCOPE_RESPONSE_EXAMPLE = OpenApiExample(
    name="ScopeResponse",
    summary="Scope validation result",
    description="Lists missing prerequisites that must be provided before scope validation passes.",
    value={"missing": ["project_brief"], "idempotent": False},
    response_only=True,
)

SCOPE_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/v1/ai/scope/",
            '-H "Content-Type: application/json"',
            '-H "X-Tenant-Schema: acme_prod"',
            '-H "X-Tenant-Id: acme"',
            '-H "X-Case-Id: crm-7421"',
            '-H "Idempotency-Key: 1d1d8aa4-0f2e-4b94-8e41-44f96c42e01a"',
            '-d \'{"prompt": "Erstelle Meeting-Notizen"}\'',
        ]
    )
)

NEEDS_REQUEST_EXAMPLE = OpenApiExample(
    name="NeedsRequest",
    summary="Needs mapping payload",
    description="Submit captured inputs so the needs mapping node can derive tasks.",
    value={
        "metadata": {"project": "acme-kickoff"},
        "needs_input": ["stakeholder_alignment", "timeline"],
    },
    request_only=True,
)

NEEDS_RESPONSE_EXAMPLE = OpenApiExample(
    name="NeedsResponse",
    summary="Needs mapping outcome",
    description="Shows whether actionable needs were derived and highlights any missing context.",
    value={
        "mapped": True,
        "missing": ["stakeholder_alignment"],
        "idempotent": False,
    },
    response_only=True,
)

NEEDS_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/v1/ai/needs/",
            '-H "Content-Type: application/json"',
            '-H "X-Tenant-Schema: acme_prod"',
            '-H "X-Tenant-Id: acme"',
            '-H "X-Case-Id: crm-7421"',
            '-H "Idempotency-Key: 9c2ef7a8-7e6b-4a55-8cb3-bf6203d86016"',
            '-d \'{"metadata": {"project": "acme-kickoff"}}\'',
        ]
    )
)

SYSDESC_REQUEST_EXAMPLE = OpenApiExample(
    name="SysDescRequest",
    summary="System description payload",
    description="Provide the finalised scope so the system description can be generated.",
    value={
        "scope": "kickoff",
        "metadata": {"project": "acme-kickoff"},
    },
    request_only=True,
)

SYSDESC_RESPONSE_EXAMPLE = OpenApiExample(
    name="SysDescResponse",
    summary="System description",
    description="Outputs the deterministic system prompt once all workflow prerequisites are met.",
    value={
        "description": "You are the NOESIS kickoff agent.",
        "skipped": False,
        "missing": [],
        "idempotent": True,
    },
    response_only=True,
)

SYSDESC_CURL = _curl(
    " ".join(
        [
            "curl -X POST https://api.noesis.example/v1/ai/sysdesc/",
            '-H "Content-Type: application/json"',
            '-H "X-Tenant-Schema: acme_prod"',
            '-H "X-Tenant-Id: acme"',
            '-H "X-Case-Id: crm-7421"',
            '-H "Idempotency-Key: f2b0b0f4-3c4b-4b9c-a8b5-5a4a9c8796c1"',
            '-d \'{"scope": "kickoff"}\'',
        ]
    )
)

RAG_QUERY_REQUEST_EXAMPLE = OpenApiExample(
    name="RagQueryRequest",
    summary="Execute retrieval for a question",
    description="Submit a question, optional pre-composed query and metadata to the production retrieval graph.",
    value={
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
            "max_candidates_effective": 50,
            "vector_candidates": 37,
            "lexical_candidates": 41,
            "deleted_matches_blocked": 0,
            "visibility_effective": "active",
            "took_ms": 42,
            "routing": {
                "profile": "standard",
                "vector_space_id": "rag/global",
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
    "description": "Persist initial workflow context and return the recorded metadata.",
    "examples": [INTAKE_REQUEST_EXAMPLE, INTAKE_RESPONSE_EXAMPLE],
    "extensions": INTAKE_CURL,
}

SCOPE_SCHEMA = {
    "request": IntakeRequestSerializer,
    "responses": {200: ScopeResponseSerializer},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Check whether the current workflow state contains the required scope metadata.",
    "examples": [SCOPE_REQUEST_EXAMPLE, SCOPE_RESPONSE_EXAMPLE],
    "extensions": SCOPE_CURL,
}

NEEDS_SCHEMA = {
    "request": IntakeRequestSerializer,
    "responses": {200: NeedsResponseSerializer},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Map captured inputs to actionable needs and signal any missing prerequisites.",
    "examples": [NEEDS_REQUEST_EXAMPLE, NEEDS_RESPONSE_EXAMPLE],
    "extensions": NEEDS_CURL,
}

SYSDESC_SCHEMA = {
    "request": IntakeRequestSerializer,
    "responses": {200: SysDescResponseSerializer},
    "error_statuses": RATE_LIMIT_JSON_ERROR_STATUSES,
    "include_trace_header": True,
    "description": "Produce a deterministic system prompt when all prerequisites have been satisfied.",
    "examples": [SYSDESC_REQUEST_EXAMPLE, SYSDESC_RESPONSE_EXAMPLE],
    "extensions": SYSDESC_CURL,
}


RAG_DEMO_DEPRECATED_RESPONSE = inline_serializer(
    name="RagDemoDeprecatedResponse",
    fields={
        "detail": serializers.CharField(),
        "code": serializers.CharField(),
    },
)


class _BaseAgentView(DeprecationHeadersMixin, APIView):
    authentication_classes: list = []
    permission_classes: list = []


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


class LegacyPingView(_PingBase):
    """Legacy heartbeat endpoint served under the unversioned prefix."""

    api_deprecated = True
    api_deprecation_id = LEGACY_DEPRECATION_ID

    @default_extend_schema(**_legacy_schema_kwargs(PING_SCHEMA))
    def get(self, request: Request) -> Response:
        return super().get(request)


class _GraphView(_BaseAgentView):
    graph_name: str | None = None

    def get_graph(self) -> GraphRunner:  # pragma: no cover - trivial indirection
        if not self.graph_name:
            raise NotImplementedError("graph_name must be configured on subclasses")
        candidate = globals().get(self.graph_name)

        try:
            registered = get_graph_runner(self.graph_name)
        except KeyError:
            registered = None

        if candidate is not None:
            runner: GraphRunner | None = None
            if isinstance(candidate, ModuleType):
                if registered is None:
                    runner = module_runner(candidate)
            elif hasattr(candidate, "run"):
                if registered is None or registered is not candidate:
                    runner = candidate

            if runner is not None:
                logger.info(
                    "graph_runner_lazy_registered",
                    extra={
                        "graph": self.graph_name,
                        "source": getattr(candidate, "__name__", repr(candidate)),
                    },
                )
                register_graph(self.graph_name, runner)
                registered = runner

        if registered is None:
            raise KeyError(f"graph runner '{self.graph_name}' is not registered")

        return registered

    def post(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

        graph_runner = self.get_graph()
        request.graph_name = self.graph_name
        response = _run_graph(request, graph_runner)
        return apply_std_headers(response, meta)


def _run_graph(request: Request, graph_runner) -> Response:  # type: ignore[no-untyped-def]
    """Compatibility wrapper used by tests to monkeypatch graph execution."""
    return services.execute_graph(request, graph_runner)


class IntakeViewV1(_GraphView):
    """Entry point for the agent intake workflow."""

    graph_name = "info_intake"

    @default_extend_schema(**INTAKE_SCHEMA)
    def post(self, request: Request) -> Response:
        return super().post(request)


class LegacyIntakeView(_GraphView):
    """Deprecated intake endpoint retained for backwards compatibility."""

    api_deprecated = True
    api_deprecation_id = LEGACY_DEPRECATION_ID
    graph_name = "info_intake"

    @default_extend_schema(**_legacy_schema_kwargs(INTAKE_SCHEMA))
    def post(self, request: Request) -> Response:
        return super().post(request)


class ScopeViewV1(_GraphView):
    """Validate that a workflow has sufficient scope information."""

    graph_name = "scope_check"

    @default_extend_schema(**SCOPE_SCHEMA)
    def post(self, request: Request) -> Response:
        return super().post(request)


class LegacyScopeView(_GraphView):
    """Deprecated scope validation endpoint retained for clients on /ai/."""

    api_deprecated = True
    api_deprecation_id = LEGACY_DEPRECATION_ID
    graph_name = "scope_check"

    @default_extend_schema(**_legacy_schema_kwargs(SCOPE_SCHEMA))
    def post(self, request: Request) -> Response:
        return super().post(request)


class NeedsViewV1(_GraphView):
    """Derive concrete needs from the captured workflow state."""

    graph_name = "needs_mapping"

    @default_extend_schema(**NEEDS_SCHEMA)
    def post(self, request: Request) -> Response:
        return super().post(request)


class LegacyNeedsView(_GraphView):
    """Deprecated needs endpoint retained for clients on /ai/."""

    api_deprecated = True
    api_deprecation_id = LEGACY_DEPRECATION_ID
    graph_name = "needs_mapping"

    @default_extend_schema(**_legacy_schema_kwargs(NEEDS_SCHEMA))
    def post(self, request: Request) -> Response:
        return super().post(request)


class SysDescViewV1(_GraphView):
    """Generate a system description for downstream agents."""

    graph_name = "system_description"

    @default_extend_schema(**SYSDESC_SCHEMA)
    def post(self, request: Request) -> Response:
        return super().post(request)


class LegacySysDescView(_GraphView):
    """Deprecated system description endpoint retained for clients on /ai/."""

    api_deprecated = True
    api_deprecation_id = LEGACY_DEPRECATION_ID
    graph_name = "system_description"

    @default_extend_schema(**_legacy_schema_kwargs(SYSDESC_SCHEMA))
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


def _normalise_rag_response(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the payload projected onto the public RAG response contract."""

    allowed_top_level = {"answer", "prompt_version", "retrieval", "snippets"}
    allowed_retrieval = {
        "alpha",
        "min_sim",
        "top_k_effective",
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
    diagnostics: dict[str, object] = {}

    top_level_extras = {
        key: value for key, value in payload.items() if key not in allowed_top_level
    }

    for key in allowed_top_level:
        if key not in payload:
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

    @default_extend_schema(**RAG_UPLOAD_SCHEMA)
    def post(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

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

        idempotency_key = request.headers.get(IDEMPOTENCY_KEY_HEADER)
        response = services.start_ingestion_run(request.data, meta, idempotency_key)

        return apply_std_headers(response, meta)


class RagIngestionStatusView(APIView):
    """Expose status information about recent ingestion runs."""

    @default_extend_schema(**RAG_INGESTION_STATUS_SCHEMA)
    def get(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

        latest = get_latest_ingestion_run(meta["tenant_id"], meta["case_id"])
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

        response = Response(response_payload, status=status.HTTP_200_OK)
        return apply_std_headers(response, meta)


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
        bind_log_context(trace_id=trace_id, tenant=tenant_id)

        actor = _resolve_hard_delete_actor(request, operator_label)

        async_result = hard_delete.delay(
            tenant_id,
            document_ids,
            reason,
            ticket_ref,
            actor=actor,
            tenant_schema=tenant_schema,
            trace_id=trace_id,
        )

        idempotent = bool(request.headers.get(IDEMPOTENCY_KEY_HEADER))
        response_payload = {
            "status": "queued",
            "job_id": getattr(async_result, "id", None),
            "trace_id": trace_id,
            "documents_requested": len(document_ids),
            "idempotent": idempotent,
        }

        meta = {"trace_id": trace_id, "tenant_id": tenant_id}
        response = Response(response_payload, status=status.HTTP_202_ACCEPTED)
        return apply_std_headers(response, meta)


class RagDemoViewV1(_BaseAgentView):
    """Deprecated demo endpoint retained only for backwards compatibility."""

    api_deprecated = True
    api_deprecation_id = "rag-demo-mvp"

    @default_extend_schema(
        request=IntakeRequestSerializer,
        responses={410: RAG_DEMO_DEPRECATED_RESPONSE},
        error_statuses=RATE_LIMIT_JSON_ERROR_STATUSES,
        include_trace_header=True,
        description=(
            "This demo workflow has been removed from the MVP build. The endpoint "
            "returns HTTP 410 to signal permanent removal."
        ),
        examples=[
            OpenApiExample(
                name="RagDemoRemoved",
                summary="Deprecated",
                description="The demo endpoint has been removed and now returns HTTP 410.",
                value={
                    "detail": "The RAG demo endpoint has been removed.",
                    "code": "rag_demo_removed",
                },
            )
        ],
    )
    def post(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

        response = _error_response(
            "The RAG demo endpoint is deprecated and no longer available in the MVP build.",
            "rag_demo_removed",
            status.HTTP_410_GONE,
        )
        return apply_std_headers(response, meta)


ping_v1 = PingViewV1.as_view()
ping_legacy = LegacyPingView.as_view()
ping = ping_legacy

intake_v1 = IntakeViewV1.as_view()
intake_legacy = LegacyIntakeView.as_view()
intake = intake_legacy

scope_v1 = ScopeViewV1.as_view()
scope_legacy = LegacyScopeView.as_view()
scope = scope_legacy

needs_v1 = NeedsViewV1.as_view()
needs_legacy = LegacyNeedsView.as_view()
needs = needs_legacy

sysdesc_v1 = SysDescViewV1.as_view()
sysdesc_legacy = LegacySysDescView.as_view()
sysdesc = sysdesc_legacy

rag_demo_v1 = RagDemoViewV1.as_view()
rag_demo = rag_demo_v1

rag_query_v1 = RagQueryViewV1.as_view()
rag_query = rag_query_v1

rag_upload_v1 = RagUploadView.as_view()
rag_upload = rag_upload_v1

rag_ingestion_run_v1 = RagIngestionRunView.as_view()
rag_ingestion_run = rag_ingestion_run_v1

rag_ingestion_status_v1 = RagIngestionStatusView.as_view()
rag_ingestion_status = rag_ingestion_status_v1

rag_hard_delete_admin = RagHardDeleteAdminView.as_view()
