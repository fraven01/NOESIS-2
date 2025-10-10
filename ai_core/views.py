from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from types import ModuleType
from importlib import import_module
from uuid import uuid4

from django.conf import settings
from django.db import connection
from django.contrib.auth import get_user_model
from django.core.exceptions import PermissionDenied
from django.http import HttpRequest
from django.utils import timezone

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
from common.logging import bind_log_context
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
from ai_core.graph.core import FileCheckpointer, GraphContext, GraphRunner
from ai_core.graph.registry import get as get_graph_runner, register as register_graph
from ai_core.graph.schemas import ToolContext, merge_state, normalize_meta
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


from .infra import object_store, rate_limit
from .ingestion import partition_document_ids, run_ingestion
from .ingestion_status import (
    get_latest_ingestion_run,
    record_ingestion_run_queued,
)
from .ingestion_utils import make_fallback_external_id
from .rag.hard_delete import hard_delete
from ai_core.tools import InputError
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.tool_contracts import (
    InputError as ToolInputError,
    NotFoundError as ToolNotFoundError,
    RateLimitedError as ToolRateLimitedError,
    TimeoutError as ToolTimeoutError,
    UpstreamServiceError as ToolUpstreamServiceError,
)
from .schemas import (
    InfoIntakeRequest,
    NeedsMappingRequest,
    RagHardDeleteAdminRequest,
    RagIngestionRunRequest,
    RagQueryRequest,
    RagUploadMetadata,
    ScopeCheckRequest,
    SystemDescriptionRequest,
)
from pydantic import ValidationError

from .rag.ingestion_contracts import (
    map_ingestion_error_to_status,
    resolve_ingestion_profile,
)
from .infra.resp import apply_std_headers


logger = logging.getLogger(__name__)


CHECKPOINTER = FileCheckpointer()


GRAPH_REQUEST_MODELS = {
    "info_intake": InfoIntakeRequest,
    "scope_check": ScopeCheckRequest,
    "needs_mapping": NeedsMappingRequest,
    "system_description": SystemDescriptionRequest,
    "rag.default": RagQueryRequest,
}


def assert_case_active(tenant: str, case_id: str) -> None:
    """Placeholder for future case activity checks."""
    return None


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


def _format_validation_error(error: ValidationError) -> str:
    """Return a compact textual representation of a Pydantic validation error."""

    messages: list[str] = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue.get("loc", ()))
        message = issue.get("msg", "Invalid input")
        if location:
            messages.append(f"{location}: {message}")
        else:
            messages.append(message)
    return "; ".join(messages)


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


def _run_graph(request: Request, graph_runner: GraphRunner) -> Response:
    meta, error = _prepare_request(request)
    if error:
        return error

    try:
        normalized_meta = normalize_meta(request)
    except ValueError as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)

    tool_context_data = normalized_meta.get("tool_context")
    tool_context: ToolContext | None = None
    if isinstance(tool_context_data, ToolContext):
        tool_context = tool_context_data
    elif isinstance(tool_context_data, Mapping):
        try:
            tool_context = ToolContext(**tool_context_data)
        except TypeError:
            tool_context = None
    if tool_context is not None:
        setattr(request, "tool_context", tool_context)
        if hasattr(request, "_request") and request._request is not request:
            setattr(request._request, "tool_context", tool_context)

    context = GraphContext(
        tenant_id=normalized_meta["tenant_id"],
        case_id=normalized_meta["case_id"],
        trace_id=normalized_meta["trace_id"],
        graph_name=normalized_meta["graph_name"],
        graph_version=normalized_meta["graph_version"],
    )

    try:
        state = CHECKPOINTER.load(context)
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)

    raw_body = getattr(request, "body", b"")
    if not raw_body and hasattr(request, "_request"):
        raw_body = getattr(request._request, "body", b"")

    content_type_header = request.headers.get("Content-Type")
    normalized_content_type = ""
    if content_type_header:
        normalized_content_type = content_type_header.split(";")[0].strip().lower()

    incoming_state = None

    if raw_body:
        if normalized_content_type and not (
            normalized_content_type == "application/json"
            or normalized_content_type.endswith("+json")
        ):
            return _error_response(
                "Request payload must be encoded as application/json.",
                "unsupported_media_type",
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )
        try:
            payload = json.loads(raw_body)
            if isinstance(payload, dict):
                incoming_state = payload
        except json.JSONDecodeError:
            return _error_response(
                "Request payload contained invalid JSON.",
                "invalid_json",
                status.HTTP_400_BAD_REQUEST,
            )

    request_model = GRAPH_REQUEST_MODELS.get(context.graph_name)
    if request_model is not None:
        data = incoming_state or {}
        try:
            validated = request_model.model_validate(data)
        except ValidationError as exc:
            return _error_response(
                _format_validation_error(exc),
                "invalid_request",
                status.HTTP_400_BAD_REQUEST,
            )
        incoming_state = validated.model_dump(exclude_none=True)

    merged_state = merge_state(state, incoming_state)

    runner_meta = dict(normalized_meta)
    if normalized_meta.get("tenant_schema"):
        runner_meta["tenant_schema"] = normalized_meta["tenant_schema"]
    if normalized_meta.get("key_alias"):
        runner_meta["key_alias"] = normalized_meta["key_alias"]

    try:
        new_state, result = graph_runner.run(merged_state, runner_meta)
    except InputError as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)
    except ValueError as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)
    except ToolNotFoundError as exc:
        logger.info("tool.not_found")
        detail = str(exc) or "No matching documents were found."
        return _error_response(detail, "rag_no_matches", status.HTTP_404_NOT_FOUND)
    except ToolInputError as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)
    except ToolRateLimitedError as _exc:
        logger.warning("tool.rate_limited")
        return _error_response(
            "Tool rate limited.", "llm_rate_limited", status.HTTP_429_TOO_MANY_REQUESTS
        )
    except ToolTimeoutError as _exc:
        logger.warning("tool.timeout")
        return _error_response(
            "Upstream tool timeout.", "llm_timeout", status.HTTP_504_GATEWAY_TIMEOUT
        )
    except ToolUpstreamServiceError as _exc:
        logger.warning("tool.upstream_error")
        return _error_response(
            "Upstream tool error.", "llm_error", status.HTTP_502_BAD_GATEWAY
        )
    except RateLimitError as exc:  # LLM proxy signalled rate limiting
        try:
            extra = {
                "status": getattr(exc, "status", None),
                "code": getattr(exc, "code", None),
            }
            logger.warning("llm.rate_limited", extra=extra)
        except Exception:
            pass
        status_code = (
            int(getattr(exc, "status", 429))
            if str(getattr(exc, "status", "")).isdigit()
            else status.HTTP_429_TOO_MANY_REQUESTS
        )
        detail = getattr(exc, "detail", None) or "LLM rate limited."
        return _error_response(detail, "llm_rate_limited", status_code)
    except LlmClientError as exc:  # Upstream LLM error (4xx/5xx)
        try:
            extra = {
                "status": getattr(exc, "status", None),
                "code": getattr(exc, "code", None),
            }
            logger.warning("llm.client_error", extra=extra)
        except Exception:
            pass
        raw_status = getattr(exc, "status", None)
        # Map all upstream client/provider errors to Bad Gateway to avoid
        # suggesting a client input error for consumers of this API.
        status_code = status.HTTP_502_BAD_GATEWAY
        # Preserve a 429 if it slipped through without specific type
        try:
            if isinstance(raw_status, int) and raw_status == 429:
                status_code = status.HTTP_429_TOO_MANY_REQUESTS
        except Exception:
            pass
        detail = getattr(exc, "detail", None) or "Upstream LLM error."
        return _error_response(detail, "llm_error", status_code)
    except Exception:
        return _error_response(
            "Service temporarily unavailable.",
            "service_unavailable",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        CHECKPOINTER.save(context, new_state)
    except (TypeError, ValueError) as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)

    response = Response(result)
    return apply_std_headers(response, meta)


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
        "idempotent": False,
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

RAG_QUERY_RESPONSE = inline_serializer(
    name="RagQueryResponse",
    fields={
        "answer": serializers.CharField(),
        "prompt_version": serializers.CharField(),
        "idempotent": serializers.BooleanField(),
    },
)

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
        graph_runner = self.get_graph()
        request.graph_name = self.graph_name
        return _run_graph(request, graph_runner)


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


class RagQueryViewV1(_GraphView):
    """Execute the production retrieval augmented generation graph."""

    graph_name = "rag.default"

    @default_extend_schema(**RAG_QUERY_SCHEMA)
    def post(self, request: Request) -> Response:
        return super().post(request)


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

        metadata_obj: dict[str, object] | None = None
        metadata_raw = request.data.get("metadata")
        if metadata_raw not in (None, ""):
            if isinstance(metadata_raw, (bytes, bytearray)):
                try:
                    metadata_text = metadata_raw.decode("utf-8")
                except UnicodeDecodeError:
                    return _error_response(
                        "Metadata must be valid JSON.",
                        "invalid_metadata",
                        status.HTTP_400_BAD_REQUEST,
                    )
            else:
                metadata_text = str(metadata_raw)

            metadata_text = metadata_text.strip()
            if metadata_text:
                try:
                    metadata_obj = json.loads(metadata_text)
                except json.JSONDecodeError:
                    return _error_response(
                        "Metadata must be valid JSON.",
                        "invalid_metadata",
                        status.HTTP_400_BAD_REQUEST,
                    )

        if metadata_obj is not None and not isinstance(metadata_obj, dict):
            return _error_response(
                "Metadata must be a JSON object when provided.",
                "invalid_metadata",
                status.HTTP_400_BAD_REQUEST,
            )

        if metadata_obj is None:
            metadata_obj = {}

        try:
            metadata_model = RagUploadMetadata.model_validate(metadata_obj)
        except ValidationError as exc:
            return _error_response(
                str(exc), "invalid_metadata", status.HTTP_400_BAD_REQUEST
            )

        metadata_obj = metadata_model.model_dump()
        if metadata_obj.get("external_id") is None:
            metadata_obj.pop("external_id")

        original_name = getattr(upload, "name", "") or "upload.bin"
        try:
            safe_name = object_store.safe_filename(original_name)
        except ValueError:
            safe_name = object_store.safe_filename("upload.bin")

        try:
            tenant_segment = object_store.sanitize_identifier(meta["tenant_id"])
            case_segment = object_store.sanitize_identifier(meta["case_id"])
        except ValueError:
            return _error_response(
                "Request metadata was invalid.",
                "invalid_request",
                status.HTTP_400_BAD_REQUEST,
            )

        document_id = uuid4().hex
        storage_prefix = f"{tenant_segment}/{case_segment}/uploads"
        object_path = f"{storage_prefix}/{document_id}_{safe_name}"

        file_bytes = upload.read()
        if not isinstance(file_bytes, (bytes, bytearray)):
            file_bytes = bytes(file_bytes)

        object_store.write_bytes(object_path, file_bytes)

        supplied_external = metadata_obj.get("external_id")
        if isinstance(supplied_external, str):
            supplied_external = supplied_external.strip()
        else:
            supplied_external = None

        if supplied_external:
            external_id = supplied_external
        else:
            external_id = make_fallback_external_id(
                original_name,
                getattr(upload, "size", None) or len(file_bytes),
                file_bytes,
            )

        metadata_obj["external_id"] = external_id

        object_store.write_json(
            f"{storage_prefix}/{document_id}.meta.json", metadata_obj
        )

        try:
            profile_binding = resolve_ingestion_profile(
                getattr(settings, "RAG_DEFAULT_EMBEDDING_PROFILE", "standard")
            )
        except InputError as exc:
            error_code = getattr(exc, "code", "invalid_ingestion_profile")
            logger.exception(
                "Failed to resolve default ingestion profile after upload",
                extra={
                    "tenant_id": meta["tenant_id"],
                    "case_id": meta["case_id"],
                    "error": str(exc),
                },
            )
            return _error_response(
                "Default ingestion profile is not configured correctly.",
                error_code,
                map_ingestion_error_to_status(error_code),
            )

        resolved_profile_id = profile_binding.profile_id
        ingestion_run_id = uuid4().hex
        queued_at = timezone.now().isoformat()
        document_ids = [document_id]

        try:
            run_ingestion.delay(
                meta["tenant_id"],
                meta["case_id"],
                document_ids,
                resolved_profile_id,
                tenant_schema=meta["tenant_schema"],
                run_id=ingestion_run_id,
                trace_id=meta["trace_id"],
                idempotency_key=request.headers.get(IDEMPOTENCY_KEY_HEADER),
            )
        except Exception:  # pragma: no cover - defensive path
            logger.exception(
                "Failed to dispatch ingestion run after upload",
                extra={
                    "tenant_id": meta["tenant_id"],
                    "case_id": meta["case_id"],
                    "document_id": document_id,
                    "run_id": ingestion_run_id,
                },
            )
            return _error_response(
                "Failed to queue ingestion run for uploaded document.",
                "ingestion_dispatch_failed",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        record_ingestion_run_queued(
            meta["tenant_id"],
            meta["case_id"],
            ingestion_run_id,
            document_ids,
            queued_at=queued_at,
            trace_id=meta["trace_id"],
            embedding_profile=resolved_profile_id,
            source="upload",
        )

        idempotent = bool(request.headers.get(IDEMPOTENCY_KEY_HEADER))
        response_payload = {
            "status": "accepted",
            "document_id": document_id,
            "trace_id": meta["trace_id"],
            "idempotent": idempotent,
            "external_id": external_id,
            "ingestion_run_id": ingestion_run_id,
            "ingestion_status": "queued",
        }

        response = Response(response_payload, status=status.HTTP_202_ACCEPTED)
        return apply_std_headers(response, meta)


class RagIngestionRunView(APIView):
    """Queue ingestion runs for previously uploaded documents."""

    @default_extend_schema(**RAG_INGESTION_RUN_SCHEMA)
    def post(self, request: Request) -> Response:
        meta, error = _prepare_request(request)
        if error:
            return error

        try:
            request_data = RagIngestionRunRequest.model_validate(request.data)
        except ValidationError as exc:
            # NOTE: Consider a more detailed error mapping for production
            return _error_response(
                str(exc), "validation_error", status.HTTP_400_BAD_REQUEST
            )

        try:
            profile_binding = resolve_ingestion_profile(request_data.embedding_profile)
        except InputError as exc:
            return _error_response(
                exc.message,
                exc.code,
                map_ingestion_error_to_status(exc.code),
            )
        resolved_profile_id = profile_binding.profile_id

        ingestion_run_id = uuid4().hex
        # Tests monkeypatch django.utils.timezone.now, so keep using the module
        # import instead of a local alias to ensure the override is observed.
        queued_at = timezone.now().isoformat()

        valid_document_ids, invalid_document_ids = partition_document_ids(
            meta["tenant_id"], meta["case_id"], request_data.document_ids
        )

        # Always enqueue the task. If at least one valid ID is known, only
        # dispatch those; otherwise pass through the original list and let the
        # task perform validation/no-op. This satisfies both observability and
        # test expectations in empty/non-empty setups.
        to_dispatch = (
            valid_document_ids if valid_document_ids else request_data.document_ids
        )
        run_ingestion.delay(
            meta["tenant_id"],
            meta["case_id"],
            to_dispatch,
            resolved_profile_id,
            tenant_schema=meta["tenant_schema"],
            run_id=ingestion_run_id,
            trace_id=meta["trace_id"],
            idempotency_key=request.headers.get(IDEMPOTENCY_KEY_HEADER),
            # Pass priority to the task if the task supports it.
            # priority=request_data.priority,
        )

        record_ingestion_run_queued(
            meta["tenant_id"],
            meta["case_id"],
            ingestion_run_id,
            to_dispatch,
            queued_at=queued_at,
            trace_id=meta["trace_id"],
            embedding_profile=resolved_profile_id,
            source="manual",
            invalid_document_ids=invalid_document_ids,
        )

        idempotent = bool(request.headers.get(IDEMPOTENCY_KEY_HEADER))
        response_payload = {
            "status": "queued",
            "queued_at": queued_at,
            "ingestion_run_id": ingestion_run_id,
            "trace_id": meta["trace_id"],
            "idempotent": idempotent,
        }

        if invalid_document_ids:
            response_payload["invalid_ids"] = invalid_document_ids
        else:
            response_payload["invalid_ids"] = []

        response = Response(response_payload, status=status.HTTP_202_ACCEPTED)
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
