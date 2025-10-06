from __future__ import annotations

import json
import logging
import re
from types import ModuleType
from importlib import import_module
from uuid import uuid4

from django.conf import settings
from django.db import connection
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
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView


from ai_core.graph.adapters import module_runner
from ai_core.graph.core import FileCheckpointer, GraphContext, GraphRunner
from ai_core.graph.registry import get as get_graph_runner, register as register_graph
from ai_core.graph.schemas import merge_state, normalize_meta
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
from .ingestion_utils import make_fallback_external_id
from .rag.ingestion_contracts import (
    IngestionContractError,
    map_ingestion_error_to_status,
    resolve_ingestion_profile,
)
from .infra.resp import apply_std_headers


logger = logging.getLogger(__name__)


CHECKPOINTER = FileCheckpointer()


def assert_case_active(tenant: str, case_id: str) -> None:
    """Placeholder for future case activity checks."""
    return None


def _resolve_tenant_id(request: HttpRequest) -> str | None:
    """Derive the active tenant identifier for the current request."""

    tenant_obj = getattr(request, "tenant", None)
    schema_name = getattr(tenant_obj, "schema_name", None)
    if not schema_name:
        schema_name = getattr(connection, "schema_name", None)

    if not schema_name:
        return None

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    if schema_name == public_schema:
        return None

    return schema_name


KEY_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _error_response(detail: str, code: str, status_code: int) -> Response:
    """Return a standardised error payload."""

    return Response({"detail": detail, "code": code}, status=status_code)


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
        "tenant": tenant_id,
        "tenant_schema": tenant_schema,
        "case": case_id,
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

    merged_state = merge_state(state, incoming_state)

    runner_meta = dict(normalized_meta)
    runner_meta["tenant"] = normalized_meta["tenant_id"]
    runner_meta["case"] = normalized_meta["case_id"]
    if normalized_meta.get("tenant_schema"):
        runner_meta["tenant_schema"] = normalized_meta["tenant_schema"]
    if normalized_meta.get("key_alias"):
        runner_meta["key_alias"] = normalized_meta["key_alias"]

    try:
        new_state, result = graph_runner.run(merged_state, runner_meta)
    except ValueError as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)
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
        "tenant": "acme",
        "case": "crm-7421",
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

        original_name = getattr(upload, "name", "") or "upload.bin"
        try:
            safe_name = object_store.safe_filename(original_name)
        except ValueError:
            safe_name = object_store.safe_filename("upload.bin")

        try:
            tenant_segment = object_store.sanitize_identifier(meta["tenant"])
            case_segment = object_store.sanitize_identifier(meta["case"])
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

        idempotent = bool(request.headers.get(IDEMPOTENCY_KEY_HEADER))
        response_payload = {
            "status": "accepted",
            "document_id": document_id,
            "trace_id": meta["trace_id"],
            "idempotent": idempotent,
            "external_id": external_id,
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

        payload = request.data if isinstance(request.data, dict) else {}

        document_ids = payload.get("document_ids")
        if not isinstance(document_ids, list) or not document_ids:
            return _error_response(
                "document_ids must be a non-empty list.",
                "invalid_document_ids",
                status.HTTP_400_BAD_REQUEST,
            )

        normalized_document_ids = []
        for value in document_ids:
            if not isinstance(value, str) or not value.strip():
                return _error_response(
                    "document_ids must contain non-empty strings.",
                    "invalid_document_ids",
                    status.HTTP_400_BAD_REQUEST,
                )
            normalized_document_ids.append(value)

        priority = payload.get("priority", "normal")
        if priority is None:
            priority = "normal"
        if not isinstance(priority, str) or not priority.strip():
            return _error_response(
                "priority must be a non-empty string when provided.",
                "invalid_priority",
                status.HTTP_400_BAD_REQUEST,
            )
        priority = priority.strip()

        try:
            profile_binding = resolve_ingestion_profile(
                payload.get("embedding_profile")
            )
        except IngestionContractError as exc:
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
            meta["tenant"], meta["case"], normalized_document_ids
        )

        # Always enqueue the task. If at least one valid ID is known, only
        # dispatch those; otherwise pass through the original list and let the
        # task perform validation/no-op. This satisfies both observability and
        # test expectations in empty/non-empty setups.
        to_dispatch = (
            valid_document_ids if valid_document_ids else normalized_document_ids
        )
        run_ingestion.delay(
            meta["tenant"],
            meta["case"],
            to_dispatch,
            resolved_profile_id,
            tenant_schema=meta["tenant_schema"],
            run_id=ingestion_run_id,
            trace_id=meta["trace_id"],
            idempotency_key=request.headers.get(IDEMPOTENCY_KEY_HEADER),
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

rag_upload_v1 = RagUploadView.as_view()
rag_upload = rag_upload_v1

rag_ingestion_run_v1 = RagIngestionRunView.as_view()
rag_ingestion_run = rag_ingestion_run_v1
