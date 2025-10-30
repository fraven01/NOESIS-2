from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from pathlib import Path
from types import ModuleType
from importlib import import_module
from urllib.parse import urlsplit
from uuid import uuid4, uuid5

from django.conf import settings
from django.db import connection
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest, HttpResponse

from common.constants import (
    IDEMPOTENCY_KEY_HEADER,
    META_CASE_ID_KEY,
    META_COLLECTION_ID_KEY,
    META_KEY_ALIAS_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    META_TRACE_ID_KEY,
    X_CASE_ID_HEADER,
    X_COLLECTION_ID_HEADER,
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
    PingResponseSerializer,
    RagQueryResponseSerializer,
)

# Crawler contracts and runtime structures used by the ingestion runner view.
from crawler.contracts import normalize_source
from crawler.errors import CrawlerError, ErrorClass
from crawler.fetcher import (
    FetchFailure,
    FetchRequest,
    FetchStatus,
    PolitenessContext,
)
from common.guardrails import FetcherLimits
from crawler.frontier import (
    CrawlSignals,
    FrontierAction,
    SourceDescriptor,
    decide_frontier_action,
)
from crawler.http_fetcher import HttpFetcher, HttpFetcherConfig
from documents.parsers import ParseStatus, ParserContent, compute_parser_stats
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
from ai_core.graph.adapters import module_runner
from ai_core.graph.core import GraphRunner
from ai_core.graph.registry import get as get_graph_runner, register as register_graph
from ai_core.graphs import (
    crawler_ingestion_graph,
    info_intake,
)  # noqa: F401
from ai_core.middleware import guardrails as guardrails_middleware
from ai_core.rag.guardrails import GuardrailLimits, GuardrailSignals

# Re-export normalize_meta so tests can monkeypatch via ai_core.views
from ai_core.graph.schemas import normalize_meta as normalize_meta  # noqa: F401
from ai_core.infra.observability import emit_event, record_span
from pydantic import ValidationError

from . import services
from .infra import object_store, rate_limit
from .infra.resp import apply_std_headers
from .ingestion import partition_document_ids as partition_document_ids  # test hook
from .ingestion import run_ingestion as run_ingestion  # re-export for tests
from .ingestion_status import (
    get_latest_ingestion_run,
)
from .rag.hard_delete import hard_delete
from .rag.ingestion_contracts import (
    resolve_ingestion_profile as resolve_ingestion_profile,  # test hook
)
from .schemas import CrawlerRunRequest, RagHardDeleteAdminRequest
from .services import CHECKPOINTER as CHECKPOINTER  # re-export for tests


GuardrailStatus = guardrails_middleware.GuardrailStatus
GuardrailErrorCategory = guardrails_middleware.GuardrailErrorCategory

# Import graphs so they are available via module globals for Legacy views.
# This enables tests to monkeypatch e.g. `views.info_intake` directly and
# allows _GraphView.get_graph to resolve from globals() without importing.
try:  # pragma: no cover - exercised indirectly via tests
    info_intake = import_module("ai_core.graphs.info_intake")
except Exception:  # defensive: don't break module import if graphs change
    # Fallback to lazy import via _GraphView.get_graph when not present.
    pass


# Optional hooks for tests to provide lifecycle stores without
# importing heavy dependencies at module import time.
DOCUMENTS_LIFECYCLE_STORE: object | None = None

if TYPE_CHECKING:  # pragma: no cover - type checking only
    pass


logger = get_logger(__name__)

logger.info(
    "module_loaded",
    extra={"module": __name__, "path": str(Path(__file__).resolve())},
)


@dataclass(slots=True)
class CrawlerStateBuild:
    """Result bundle returned by :func:`_build_crawler_state`."""

    origin: str
    provider: str
    document_id: str
    state: dict[str, object]
    fetch_used: bool
    http_status: int | None
    fetched_bytes: int | None
    media_type_effective: str | None
    fetch_elapsed: float | None
    fetch_retries: int | None
    fetch_retry_reason: str | None
    fetch_backoff_total_ms: float | None
    snapshot_path: str | None
    snapshot_sha256: str | None
    tags: tuple[str, ...]
    collection_id: str | None
    snapshot_requested: bool
    snapshot_label: str | None
    review: str | None
    dry_run: bool


class CrawlerRunError(RuntimeError):
    """Raised when crawler state preparation cannot proceed."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = dict(details or {})


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


def _build_header_mapping(fetch_result) -> dict[str, str]:
    """Return a header mapping reconstructed from fetch metadata."""

    headers: dict[str, str] = {}
    metadata = fetch_result.metadata
    if metadata.content_type:
        headers["Content-Type"] = metadata.content_type
    if metadata.etag:
        headers["ETag"] = metadata.etag
    if metadata.last_modified:
        headers["Last-Modified"] = metadata.last_modified
    if metadata.content_length is not None:
        headers["Content-Length"] = str(metadata.content_length)
    return headers


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


def _merge_origin_tags(
    global_tags: Sequence[str] | None, origin_tags: Sequence[str] | None
) -> list[str]:
    """Combine global and origin scoped tags while preserving order."""

    combined: list[str] = []
    seen: set[str] = set()
    for tag_list in (global_tags or []), (origin_tags or []):
        if not tag_list:
            continue
        for tag in tag_list:
            if not tag:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            combined.append(tag)
    return combined


def _map_failure(error: ErrorClass | None, reason: str | None) -> FetchFailure | None:
    """Translate :class:`CrawlerError` metadata into a fetch failure contract."""

    if error is None:
        return None
    if error is ErrorClass.TIMEOUT:
        return FetchFailure(reason=reason or "timeout", temporary=True)
    if error is ErrorClass.TRANSIENT_NETWORK:
        return FetchFailure(reason=reason or "network_error", temporary=True)
    if error is ErrorClass.RATE_LIMIT:
        return FetchFailure(reason=reason or "rate_limited", temporary=True)
    return FetchFailure(reason=reason or error.value, temporary=False)


def _map_fetch_error_response(result) -> tuple[int, str]:
    """Return HTTP status code and error identifier for fetch failures."""

    error = result.error
    if error is None:
        return status.HTTP_502_BAD_GATEWAY, "crawler_fetch_failed"

    error_class = getattr(error, "error_class", None)
    if error_class is ErrorClass.TIMEOUT:
        return status.HTTP_504_GATEWAY_TIMEOUT, "crawler_fetch_timeout"
    if error_class is ErrorClass.RATE_LIMIT:
        return status.HTTP_429_TOO_MANY_REQUESTS, "crawler_fetch_rate_limited"
    if error_class is ErrorClass.NOT_FOUND:
        return status.HTTP_404_NOT_FOUND, "crawler_fetch_not_found"
    if error_class is ErrorClass.GONE:
        return status.HTTP_410_GONE, "crawler_fetch_gone"
    if error_class is ErrorClass.POLICY_DENY:
        return status.HTTP_403_FORBIDDEN, "crawler_fetch_policy_denied"
    if error_class is ErrorClass.UPSTREAM_429:
        return status.HTTP_429_TOO_MANY_REQUESTS, "crawler_fetch_upstream_429"
    if error_class is ErrorClass.TRANSIENT_NETWORK:
        return status.HTTP_503_SERVICE_UNAVAILABLE, "crawler_fetch_transient_error"
    return status.HTTP_502_BAD_GATEWAY, "crawler_fetch_failed"


def _write_snapshot(
    *,
    tenant: str,
    case: str,
    payload: bytes,
) -> tuple[str, str]:
    """Persist the crawler payload as HTML snapshot and return metadata."""

    sha256 = hashlib.sha256(payload).hexdigest()
    tenant_safe = object_store.sanitize_identifier(tenant)
    case_safe = object_store.sanitize_identifier(case)
    relative = "/".join((tenant_safe, case_safe, "crawler", f"{sha256}.html"))
    object_store.write_bytes(relative, payload)
    absolute = str(object_store.BASE_PATH / relative)
    return absolute, sha256


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
    collection_header = request.headers.get(X_COLLECTION_ID_HEADER)
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

    collection_id = None
    if collection_header is not None:
        candidate = collection_header.strip()
        if candidate:
            collection_id = candidate

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
    if collection_id:
        meta["collection_id"] = collection_id
    if idempotency_key:
        meta["idempotency_key"] = idempotency_key
    print("PREPARE_META_IDEMPOTENCY", idempotency_key)

    request.META[META_TRACE_ID_KEY] = trace_id
    request.META[META_CASE_ID_KEY] = case_id
    request.META[META_TENANT_ID_KEY] = tenant_id
    request.META[META_TENANT_SCHEMA_KEY] = tenant_schema
    if key_alias:
        request.META[META_KEY_ALIAS_KEY] = key_alias
    else:
        request.META.pop(META_KEY_ALIAS_KEY, None)
    if collection_id:
        request.META[META_COLLECTION_ID_KEY] = collection_id
    else:
        request.META.pop(META_COLLECTION_ID_KEY, None)

    log_context = {
        "trace_id": trace_id,
        "case_id": case_id,
        "tenant": tenant_id,
        "key_alias": key_alias,
        "collection_id": collection_id,
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
            "matches_returned": 1,
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


def _resolve_document_uuid(identifier: object) -> uuid.UUID | None:
    """Best-effort conversion mirroring :mod:`documents.api` behaviour."""

    if isinstance(identifier, uuid.UUID):
        return identifier
    if identifier is None:
        return None
    try:
        candidate = str(identifier).strip()
    except Exception:  # pragma: no cover - defensive
        candidate = str(identifier)
    if not candidate:
        return None
    try:
        return uuid.UUID(candidate)
    except (TypeError, ValueError):
        return uuid5(uuid.NAMESPACE_URL, candidate)


def _load_baseline_context(
    tenant_id: object,
    workflow_id: object,
    document_identifier: object,
    repository: object | None,
    lifecycle_store: object | None,
) -> tuple[dict[str, object], str | None]:
    """Fetch baseline metadata for the crawler graph state."""

    baseline: dict[str, object] = {}
    previous_status: str | None = None

    tenant: str | None = None
    if tenant_id is not None:
        tenant_candidate = str(tenant_id).strip()
        if tenant_candidate:
            tenant = tenant_candidate
    if not tenant:
        return baseline, previous_status

    document_uuid = _resolve_document_uuid(document_identifier)
    if document_uuid is None:
        return baseline, previous_status

    workflow: str | None = None
    if workflow_id is not None:
        workflow_candidate = str(workflow_id).strip()
        if workflow_candidate:
            workflow = workflow_candidate

    if repository is not None and hasattr(repository, "get"):
        try:
            existing = repository.get(  # type: ignore[attr-defined]
                tenant,
                document_uuid,
                prefer_latest=True,
                workflow_id=workflow,
            )
        except NotImplementedError:
            existing = None
        except Exception:  # pragma: no cover - best effort logging
            logger.debug(
                "crawler.baseline.repository_lookup_failed",
                extra={"tenant_id": tenant, "document_id": str(document_identifier)},
                exc_info=True,
            )
            existing = None

        if existing is not None:
            checksum = getattr(existing, "checksum", None)
            if checksum:
                checksum_str = str(checksum)
                baseline.setdefault("checksum", checksum_str)
                baseline.setdefault("content_hash", checksum_str)
            ref = getattr(existing, "ref", None)
            if ref is not None:
                document_ref_id = getattr(ref, "document_id", None)
                if document_ref_id is not None:
                    baseline.setdefault("document_id", str(document_ref_id))
                collection_id = getattr(ref, "collection_id", None)
                if collection_id is not None:
                    baseline.setdefault("collection_id", str(collection_id))
                version = getattr(ref, "version", None)
                if version:
                    baseline.setdefault("version", version)
            lifecycle_state = getattr(existing, "lifecycle_state", None)
            if lifecycle_state:
                lifecycle_text = str(lifecycle_state)
                baseline.setdefault("lifecycle_state", lifecycle_text)
                if previous_status is None:
                    previous_status = lifecycle_text

    if lifecycle_store is not None:
        getter = getattr(lifecycle_store, "get_document_state", None)
        if callable(getter):
            try:
                record = getter(  # type: ignore[misc]
                    tenant_id=tenant,
                    document_id=document_uuid,
                    workflow_id=workflow,
                )
            except Exception:  # pragma: no cover - best effort logging
                logger.debug(
                    "crawler.baseline.lifecycle_lookup_failed",
                    extra={
                        "tenant_id": tenant,
                        "document_id": str(document_identifier),
                    },
                    exc_info=True,
                )
                record = None

            if record is not None:
                state_value = getattr(record, "state", None)
                if state_value:
                    state_text = str(state_value)
                    baseline.setdefault("lifecycle_state", state_text)
                    previous_status = state_text
                reason_value = getattr(record, "reason", None)
                if reason_value:
                    baseline.setdefault("previous_reason", str(reason_value))
                events = getattr(record, "policy_events", None)
                if events:
                    baseline.setdefault("policy_events", tuple(events))

    return baseline, previous_status


def _normalise_rag_response(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the payload projected onto the public RAG response contract."""

    allowed_top_level = {"answer", "prompt_version", "retrieval", "snippets"}
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


def _build_crawler_state(
    meta: Mapping[str, object], request_data: CrawlerRunRequest
) -> list[CrawlerStateBuild]:
    """Compose crawler graph state objects for each requested origin."""

    workflow_default = getattr(settings, "CRAWLER_DEFAULT_WORKFLOW_ID", None)
    workflow_id = request_data.workflow_id or workflow_default or meta.get("tenant_id")
    if not workflow_id:
        raise ValueError("workflow_id could not be resolved for the crawler run")

    try:
        repository = services._get_documents_repository()
    except Exception:
        repository = None
    lifecycle_store = _resolve_lifecycle_store()

    builds: list[CrawlerStateBuild] = []
    for origin in request_data.origins or []:
        provider = origin.provider or request_data.provider
        try:
            source = normalize_source(provider, origin.url, None)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(str(exc)) from exc

        parsed = urlsplit(source.canonical_source)
        host = parsed.hostname or parsed.netloc
        if not host:
            raise ValueError("origin URL must include a valid host component")
        path_component = parsed.path or "/"

        descriptor = SourceDescriptor(
            host=host, path=path_component, provider=source.provider
        )
        frontier_input = {"descriptor": descriptor, "signals": CrawlSignals()}

        politeness = PolitenessContext(host=descriptor.host)
        fetch_request = FetchRequest(
            canonical_source=source.canonical_source, politeness=politeness
        )

        document_id = origin.document_id or request_data.document_id or uuid4().hex
        tags = tuple(_merge_origin_tags(request_data.tags, origin.tags))

        limit_bytes = None
        if origin.limits and origin.limits.max_document_bytes is not None:
            limit_bytes = origin.limits.max_document_bytes
        elif request_data.max_document_bytes is not None:
            limit_bytes = request_data.max_document_bytes
        limits = GuardrailLimits(max_document_bytes=limit_bytes)

        snapshot_options = origin.snapshot
        if snapshot_options is None and (
            request_data.snapshot.enabled or request_data.snapshot.label
        ):
            snapshot_options = request_data.snapshot
        snapshot_requested = bool(snapshot_options and snapshot_options.enabled)
        snapshot_label = snapshot_options.label if snapshot_options else None

        dry_run = bool(
            origin.dry_run if origin.dry_run is not None else request_data.dry_run
        )
        review = origin.review or request_data.review or request_data.manual_review

        need_fetch = bool(origin.fetch or origin.content is None)
        body_bytes: bytes = b""
        effective_content_type = _normalize_media_type_value(
            origin.content_type or request_data.content_type
        )
        fetch_input: dict[str, object]
        fetch_used = False
        http_status: int | None = None
        fetched_bytes: int | None = None
        fetch_elapsed: float | None = None
        fetch_retries: int | None = None
        fetch_retry_reason: str | None = None
        fetch_backoff_total_ms: float | None = None
        snapshot_path: str | None = None
        snapshot_sha256: str | None = None

        if need_fetch:
            decision = decide_frontier_action(descriptor, CrawlSignals())
            if decision.action is not FrontierAction.ENQUEUE:
                emit_event(
                    "crawler_robots_blocked",
                    {
                        "host": descriptor.host,
                        "reason": decision.reason,
                        "policy_events": list(decision.policy_events),
                    },
                )
                raise CrawlerRunError(
                    "Frontier denied the crawl due to robots or scheduling policies.",
                    code="crawler_robots_blocked",
                    status_code=status.HTTP_403_FORBIDDEN,
                    details={
                        "fetch_used": False,
                        "http_status": None,
                        "fetched_bytes": None,
                        "media_type_effective": None,
                        "fetch_elapsed": None,
                        "fetch_retries": None,
                        "fetch_retry_reason": None,
                        "fetch_backoff_total_ms": None,
                    },
                )

            emit_event(
                "fetch_started",
                {"origin": source.canonical_source, "provider": source.provider},
            )
            fetch_limits = None
            if limits.max_document_bytes is not None:
                fetch_limits = FetcherLimits(max_bytes=limits.max_document_bytes)
            config = HttpFetcherConfig(limits=fetch_limits)
            fetcher = HttpFetcher(config)
            fetch_result = fetcher.fetch(fetch_request)
            record_span(
                "crawler.fetch",
                attributes={
                    "crawler.fetch.status": fetch_result.status.value,
                    "crawler.fetch.bytes": fetch_result.telemetry.bytes_downloaded,
                    "crawler.fetch.retry_reason": fetch_result.telemetry.retry_reason,
                },
            )
            emit_event(
                "fetch_finished",
                {
                    "status": fetch_result.status.value,
                    "status_code": fetch_result.metadata.status_code,
                    "bytes": fetch_result.telemetry.bytes_downloaded,
                },
            )
            fetch_elapsed = fetch_result.telemetry.latency
            fetch_retries = fetch_result.telemetry.retries
            fetch_retry_reason = fetch_result.telemetry.retry_reason
            fetch_backoff_total_ms = fetch_result.telemetry.backoff_total_ms

            if fetch_result.status is not FetchStatus.FETCHED:
                status_code, code = _map_fetch_error_response(fetch_result)
                emit_event(code, {"origin": source.canonical_source})
                details = {
                    "fetch_used": True,
                    "http_status": fetch_result.metadata.status_code,
                    "fetched_bytes": fetch_result.telemetry.bytes_downloaded,
                    "media_type_effective": fetch_result.metadata.content_type,
                    "fetch_elapsed": fetch_elapsed,
                    "fetch_retries": fetch_retries,
                    "fetch_retry_reason": fetch_retry_reason,
                    "fetch_backoff_total_ms": fetch_backoff_total_ms,
                }
                raise CrawlerRunError(
                    "Fetching the origin URL failed.",
                    code=code,
                    status_code=status_code,
                    details=details,
                )

            fetch_used = True
            http_status = fetch_result.metadata.status_code
            payload_bytes = getattr(fetch_result, "payload", None)
            if payload_bytes is None:
                payload_bytes = getattr(fetch_result, "body", None)
            body_bytes = payload_bytes or b""
            fetched_bytes = len(body_bytes)
            effective_content_type = _normalize_media_type_value(
                fetch_result.metadata.content_type
            )
            fetch_input = {
                "request": fetch_request,
                "status_code": fetch_result.metadata.status_code,
                "body": body_bytes,
                "headers": _build_header_mapping(fetch_result),
                "elapsed": fetch_result.telemetry.latency,
                "retries": fetch_result.telemetry.retries,
                "retry_reason": fetch_result.telemetry.retry_reason,
                "downloaded_bytes": fetch_result.telemetry.bytes_downloaded,
                "backoff_total_ms": fetch_result.telemetry.backoff_total_ms,
            }
            if fetch_limits is not None:
                fetch_input["limits"] = fetch_limits
            failure = _map_failure(
                getattr(fetch_result.error, "error_class", None),
                getattr(fetch_result.error, "reason", None),
            )
            if failure is not None:
                fetch_input["failure"] = failure
        else:
            if origin.content is None:
                raise ValueError(
                    "Manual crawler runs require inline content. Provide content or enable remote fetching."
                )
            body_bytes = origin.content.encode("utf-8")
            fetched_bytes = len(body_bytes)
            fetch_elapsed = 0.05
            fetch_retries = 0
            fetch_retry_reason = None
            fetch_backoff_total_ms = 0.0
            manual_content_type = effective_content_type or "application/octet-stream"
            fetch_input = {
                "request": fetch_request,
                "status_code": 200,
                "body": body_bytes,
                "headers": {"Content-Type": manual_content_type},
                "elapsed": 0.05,
            }

        if effective_content_type is None:
            effective_content_type = "application/octet-stream"

        guardrail_signals = GuardrailSignals(
            tenant_id=str(meta.get("tenant_id")),
            provider=source.provider,
            canonical_source=source.canonical_source,
            host=descriptor.host,
            document_bytes=len(body_bytes),
            mime_type=effective_content_type,
        )
        guardrail_decision = guardrails_middleware.enforce_guardrails(
            limits=limits,
            signals=guardrail_signals,
            error_builder=_build_guardrail_error,
        )
        if guardrail_decision.status is GuardrailStatus.DENY:
            emit_event(
                "crawler_guardrail_denied",
                {
                    "reason": guardrail_decision.reason,
                    "policy_events": list(guardrail_decision.policy_events),
                },
            )
            raise CrawlerRunError(
                "Crawler guardrails denied the document.",
                code="crawler_guardrail_denied",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                details={
                    "fetch_used": fetch_used,
                    "http_status": http_status,
                    "fetched_bytes": fetched_bytes,
                    "media_type_effective": effective_content_type,
                    "fetch_elapsed": fetch_elapsed,
                    "fetch_retries": fetch_retries,
                    "fetch_retry_reason": fetch_retry_reason,
                    "fetch_backoff_total_ms": fetch_backoff_total_ms,
                },
            )

        try:
            decoded = body_bytes.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            decoded = body_bytes.decode("latin-1", errors="replace")
        decoded = _sanitize_primary_text(decoded)

        parse_content = ParserContent(
            media_type=effective_content_type,
            primary_text=decoded,
            title=origin.title or request_data.title,
            content_language=origin.language or request_data.language,
        )
        parse_stats = compute_parser_stats(
            primary_text=decoded,
            extraction_path="crawler.manual",
        )
        parse_input = {
            "status": ParseStatus.PARSED,
            "content": parse_content,
            "stats": parse_stats,
        }

        normalize_input = {
            "source": source,
            "document_id": document_id,
            "tags": tags,
        }

        gating_input = {"limits": limits, "signals": guardrail_signals}

        if snapshot_requested and body_bytes:
            tenant_id = str(meta.get("tenant_id"))
            case_id = str(meta.get("case_id"))
            snapshot_path, snapshot_sha256 = _write_snapshot(
                tenant=tenant_id,
                case=case_id,
                payload=body_bytes,
            )
        else:
            snapshot_path = None
            snapshot_sha256 = None

        state: dict[str, object] = {
            "tenant_id": meta.get("tenant_id"),
            "case_id": meta.get("case_id"),
            "workflow_id": workflow_id,
            "external_id": source.external_id,
            "origin_uri": source.canonical_source,
            "provider": source.provider,
            "frontier_input": frontier_input,
            "fetch_input": fetch_input,
            "parse_input": parse_input,
            "normalize_input": normalize_input,
            "delta_input": {},
            "gating_input": gating_input,
            "document_id": document_id,
            "collection_id": request_data.collection_id,
        }
        control: dict[str, object] = {
            "snapshot": snapshot_requested,
            "snapshot_label": snapshot_label,
            "fetch": fetch_used,
            "tags": list(tags),
            "shadow_mode": bool(request_data.shadow_mode or dry_run),
            "dry_run": dry_run,
        }
        if review:
            control["review"] = review
            control["manual_review"] = review
        if request_data.force_retire:
            control["force_retire"] = True
        if request_data.recompute_delta:
            control["recompute_delta"] = True
        state["control"] = control

        baseline_data, previous_status = _load_baseline_context(
            meta.get("tenant_id"),
            workflow_id,
            document_id,
            repository,
            lifecycle_store,
        )
        state["baseline"] = baseline_data
        if previous_status:
            state["previous_status"] = previous_status

        builds.append(
            CrawlerStateBuild(
                origin=source.canonical_source,
                provider=source.provider,
                document_id=document_id,
                state=state,
                fetch_used=fetch_used,
                http_status=http_status,
                fetched_bytes=fetched_bytes,
                media_type_effective=effective_content_type,
                fetch_elapsed=fetch_elapsed,
                fetch_retries=fetch_retries,
                fetch_retry_reason=fetch_retry_reason,
                fetch_backoff_total_ms=fetch_backoff_total_ms,
                snapshot_path=snapshot_path,
                snapshot_sha256=snapshot_sha256,
                tags=tags,
                collection_id=request_data.collection_id,
                snapshot_requested=snapshot_requested,
                snapshot_label=snapshot_label,
                review=review,
                dry_run=dry_run,
            )
        )

    return builds


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
        if isinstance(request.data, Mapping):
            payload = dict(request.data)
        else:
            payload = dict(getattr(request, "data", {}) or {})

        if (
            meta.get("collection_id")
            and not payload.get("collection_id")
            and payload.get("collection_ids") in (None, "")
        ):
            payload["collection_id"] = meta["collection_id"]

        response = services.start_ingestion_run(payload, meta, idempotency_key)

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
        processed_response = apply_std_headers(response, meta)
        idempotency_key_value = meta.get("idempotency_key")
        header_idempotency = request.headers.get(IDEMPOTENCY_KEY_HEADER)
        print("DEBUG_IDEMPOTENCY", header_idempotency, idempotency_key_value)
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
                meta["idempotency_key"] = resolved_idempotency
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


class CrawlerIngestionRunnerView(APIView):
    """Expose the crawler ingestion LangGraph for manual debugging runs."""

    @default_extend_schema(**CRAWLER_RUN_SCHEMA)
    def post(self, request: Request) -> Response:
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

        if request_model.collection_id:
            meta["collection_id"] = request_model.collection_id

        try:
            state_builds = _build_crawler_state(meta, request_model)
        except CrawlerRunError as exc:
            payload = {"detail": str(exc), "code": exc.code}
            payload.update(exc.details)
            response = Response(payload, status=exc.status_code)
            return apply_std_headers(response, meta)
        except ValueError as exc:
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )

        if not state_builds:
            return _error_response(
                "No origins resolved for crawler run.",
                "missing_origins",
                status.HTTP_400_BAD_REQUEST,
            )

        origin_keys = sorted(build.origin for build in state_builds)
        workflow_id = state_builds[0].state.get("workflow_id")
        fingerprint_components = [
            str(meta.get("tenant_id", "")),
            str(meta.get("case_id", "")),
            str(workflow_id or ""),
            request_model.mode,
            "|".join(origin_keys),
        ]
        fingerprint = hashlib.sha256(
            "::".join(fingerprint_components).encode("utf-8")
        ).hexdigest()
        fingerprint_path: str | None = None
        fingerprint_match = False
        try:
            tenant_key = object_store.sanitize_identifier(str(meta.get("tenant_id")))
            case_key = object_store.sanitize_identifier(str(meta.get("case_id")))
            fingerprint_path = (
                f"{tenant_key}/{case_key}/crawler_runner_idempotency.json"
            )
        except Exception:
            fingerprint_path = None
        if fingerprint_path:
            try:
                existing = object_store.read_json(fingerprint_path)
            except FileNotFoundError:
                existing = None
            except Exception:
                existing = None
            if (
                isinstance(existing, dict)
                and existing.get("fingerprint") == fingerprint
            ):
                fingerprint_match = True
            else:
                try:
                    object_store.write_json(
                        fingerprint_path,
                        {
                            "fingerprint": fingerprint,
                            "workflow_id": workflow_id,
                            "mode": request_model.mode,
                            "origins": origin_keys,
                        },
                    )
                except Exception:
                    pass

        origins_payload: list[dict[str, object]] = []
        transitions_payload: list[dict[str, object]] = []
        telemetry_payload: list[dict[str, object]] = []
        errors_payload: list[dict[str, object]] = []

        raw_header_idempotency = request.headers.get(IDEMPOTENCY_KEY_HEADER)
        if not raw_header_idempotency:
            raw_header_idempotency = request.META.get(
                "HTTP_" + IDEMPOTENCY_KEY_HEADER.upper().replace("-", "_")
            )
        if not raw_header_idempotency:
            raw_header_idempotency = request.META.get(IDEMPOTENCY_KEY_HEADER)
        resolved_idempotency = meta.get("idempotency_key")
        if not resolved_idempotency and isinstance(raw_header_idempotency, str):
            candidate_key = raw_header_idempotency.strip()
            if candidate_key:
                resolved_idempotency = candidate_key
                meta["idempotency_key"] = candidate_key
        header_idempotent = bool(resolved_idempotency)

        for build in state_builds:
            graph = crawler_ingestion_graph.build_graph()

            def _upsert_handler(decision):  # type: ignore[no-untyped-def]
                try:
                    document_id: object | None = None
                    chunk_meta: object | None = None
                    attributes = getattr(decision, "attributes", None)
                    if isinstance(attributes, Mapping):
                        chunk_meta = attributes.get("chunk_meta")
                        if chunk_meta is not None:
                            if hasattr(chunk_meta, "document_id"):
                                document_id = getattr(chunk_meta, "document_id", None)
                            elif isinstance(chunk_meta, Mapping):
                                document_id = chunk_meta.get("document_id")
                        if document_id is None:
                            document_id = attributes.get("document_id")
                    if document_id is None and hasattr(decision, "payload"):
                        payload = getattr(decision, "payload")
                        document_id = getattr(payload, "document_id", None)
                        if document_id is None and hasattr(payload, "chunk_meta"):
                            chunk_meta = getattr(payload, "chunk_meta", None)
                            if getattr(chunk_meta, "document_id", None):
                                document_id = getattr(chunk_meta, "document_id")
                    if not document_id:
                        document_id = build.document_id
                    if not document_id:
                        return {"status": "skipped", "reason": "missing_document_id"}
                    request_data = {
                        "document_ids": [str(document_id)],
                        "embedding_profile": getattr(
                            settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard"
                        ),
                    }
                    collection_scope = (
                        build.collection_id
                        or request_model.collection_id
                        or meta.get("collection_id")
                    )
                    ingest_meta = dict(meta)
                    if collection_scope:
                        request_data["collection_id"] = collection_scope
                        ingest_meta["collection_id"] = collection_scope
                    resp = services.start_ingestion_run(
                        request_data,
                        ingest_meta,
                        idempotency_key=resolved_idempotency,
                    )
                    body = getattr(resp, "data", {}) or {}
                    run_id = body.get("ingestion_run_id")
                    status_label = body.get("status") or "queued"
                    result = {"status": status_label}
                    if run_id:
                        result["ingestion_run_id"] = run_id
                    return result
                except Exception as exc:  # pragma: no cover - defensive
                    return {"status": "error", "error": str(exc)}

            try:
                graph.upsert_handler = _upsert_handler  # type: ignore[attr-defined]
            except Exception:
                pass

            try:
                state = graph.start_crawl(build.state)
            except Exception as exc:
                errors_payload.append(
                    {"origin": build.origin, "error": str(exc), "stage": "start_crawl"}
                )
                continue

            graph_meta = {
                "tenant_id": meta["tenant_id"],
                "case_id": meta["case_id"],
                "workflow_id": state.get("workflow_id"),
            }

            try:
                result_state, result = graph.run(state, graph_meta)
            except Exception as exc:
                errors_payload.append(
                    {"origin": build.origin, "error": str(exc), "stage": "run"}
                )
                continue

            artifacts = result_state.get("artifacts", {}) or {}
            upsert_result = (
                artifacts.get("upsert_result") if isinstance(artifacts, dict) else {}
            )
            ingestion_run_id = None
            if isinstance(upsert_result, dict):
                ingestion_run_id = upsert_result.get("ingestion_run_id")

            origin_snapshot = {
                "workflow_id": result_state.get("workflow_id"),
                "document_id": result_state.get("normalize_input", {}).get(
                    "document_id"
                )
                or result_state.get("document_id"),
                "origin_uri": result_state.get("origin_uri"),
                "provider": result_state.get("provider"),
                "content_hash": result_state.get("content_hash"),
                "tags": list(result_state.get("normalize_input", {}).get("tags", ())),
                "snapshot_requested": build.snapshot_requested,
                "snapshot_label": build.snapshot_label,
            }
            if build.snapshot_path:
                origin_snapshot["snapshot_path"] = build.snapshot_path
            if build.snapshot_sha256:
                origin_snapshot["snapshot_sha256"] = build.snapshot_sha256

            origin_entry: dict[str, object] = {
                "origin": build.origin,
                "provider": build.provider,
                "document_id": build.document_id,
                "result": services._make_json_safe(result),
                "control": services._make_json_safe(result_state.get("control", {})),
                "ingest_action": result_state.get("ingest_action"),
                "gating_score": result_state.get("gating_score"),
                "graph_run_id": result.get("graph_run_id"),
                "state": services._make_json_safe(origin_snapshot),
                "collection_id": build.collection_id,
                "review": build.review,
                "dry_run": build.dry_run,
            }
            if ingestion_run_id:
                origin_entry["ingestion_run_id"] = ingestion_run_id

            origins_payload.append(origin_entry)

            transitions_payload.append(
                {
                    "origin": build.origin,
                    "transitions": services._make_json_safe(
                        result_state.get("transitions", {})
                    ),
                }
            )

            telemetry_entry: dict[str, object] = {
                "origin": build.origin,
                "provider": build.provider,
                "fetch_used": build.fetch_used,
                "http_status": build.http_status,
                "fetched_bytes": build.fetched_bytes,
                "media_type_effective": build.media_type_effective,
                "fetch_elapsed": build.fetch_elapsed,
                "fetch_retries": build.fetch_retries,
                "fetch_retry_reason": build.fetch_retry_reason,
                "fetch_backoff_total_ms": build.fetch_backoff_total_ms,
                "snapshot_requested": build.snapshot_requested,
                "snapshot_label": build.snapshot_label,
                "tags": list(build.tags),
            }
            if build.snapshot_path:
                telemetry_entry["snapshot_path"] = build.snapshot_path
            if build.snapshot_sha256:
                telemetry_entry["snapshot_sha256"] = build.snapshot_sha256
            telemetry_payload.append(telemetry_entry)

        response_payload = {
            "workflow_id": workflow_id,
            "mode": request_model.mode,
            "collection_id": request_model.collection_id,
            "origins": origins_payload,
            "transitions": transitions_payload,
            "telemetry": telemetry_payload,
            "errors": errors_payload,
            "idempotent": bool(fingerprint_match or header_idempotent),
        }
        response_body = json.dumps(response_payload)
        response = HttpResponse(response_body, status=status.HTTP_200_OK)
        response["Content-Type"] = "application/json"
        response = apply_std_headers(response, meta)
        if resolved_idempotency:
            response[IDEMPOTENCY_KEY_HEADER] = resolved_idempotency
        return response


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
crawler_runner = CrawlerIngestionRunnerView.as_view()
