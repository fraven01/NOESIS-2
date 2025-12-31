"""
This module contains the business logic service layer for the AI Core.

The service layer is responsible for orchestrating complex operations, separating the
concerns of the HTTP view layer from the underlying capabilities (e.g., graphs,
tasks, data access).

Pattern:
- Views in `views.py` should be thin. They handle HTTP request/response concerns.
- Views call functions in this `services` module.
- Service functions contain the actual business logic, calling graphs, tasks, etc.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import time
from inspect import signature
from collections.abc import Iterable, Mapping
from importlib import import_module
from typing import Any
from uuid import UUID, uuid4

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone

from pydantic import ValidationError
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from ai_core.infra.observability import (
    emit_event,
    observe_span,
    update_observation,
    start_trace as lf_start_trace,
    end_trace as lf_end_trace,
    tracing_enabled as lf_tracing_enabled,
)

from common.constants import (
    COLLECTION_ID_HEADER_CANDIDATES,
    META_COLLECTION_ID_KEY,
)

from ai_core.graph.core import FileCheckpointer, GraphContext, GraphRunner
from ai_core.graph.schemas import merge_state
from ai_core.tool_contracts import ToolContext
from ai_core.graph.schemas import normalize_meta as _base_normalize_meta
from ai_core.graphs.technical.cost_tracking import coerce_cost_value, track_ledger_costs
from ai_core.llm.client import LlmClientError, RateLimitError
from celery import current_app, exceptions as celery_exceptions
from common.celery import with_scope_apply_async
from ai_core.tool_contracts import (
    InconsistentMetadataError as ToolInconsistentMetadataError,
)
from ai_core.tool_contracts import InputError as ToolInputError
from ai_core.tool_contracts import NotFoundError as ToolNotFoundError
from ai_core.tool_contracts import RateLimitedError as ToolRateLimitedError
from ai_core.tool_contracts import TimeoutError as ToolTimeoutError
from ai_core.tool_contracts import UpstreamServiceError as ToolUpstreamServiceError
from ai_core.tool_contracts import ContextError as ToolContextError
from ai_core.tools import InputError

from ai_core.graphs.transition_contracts import (
    GraphTransition,
    StandardTransitionResult,
)

# NOTE: UniversalIngestionError and build_universal_ingestion_graph are NOT imported at module level
# to prevent OOM in tests. Import them lazily inside functions that need them.
# See: ai_core/graphs/technical/universal_ingestion_graph.py
from ai_core.rag.vector_client import get_default_client
from customers.tenant_context import TenantContext
from documents.domain_service import DocumentDomainService
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
from documents.models import DocumentCollection
from documents.repository import DocumentsRepository, InMemoryDocumentsRepository
from ai_core.adapters.db_documents_repository import DbDocumentsRepository
from cases.models import Case
from ai_core.rag.collections import (
    MANUAL_COLLECTION_LABEL,
    MANUAL_COLLECTION_SLUG,
    ensure_manual_collection,
    manual_collection_uuid,
)

from ..case_events import emit_ingestion_case_event
from ..infra import object_store
from ..ingestion import partition_document_ids, run_ingestion
from ..ingestion_status import record_ingestion_run_queued
from ..ingestion_utils import make_fallback_external_id
from ..rag.ingestion_contracts import (
    map_ingestion_error_to_status,
    resolve_ingestion_profile as _base_resolve_ingestion_profile,
)
from ..schemas import (
    InfoIntakeRequest,
    RagIngestionRunRequest,
    RagQueryRequest,
    RagUploadMetadata,
)


logger = logging.getLogger(__name__)


CHECKPOINTER = FileCheckpointer()
ASYNC_GRAPH_NAMES = frozenset(
    getattr(settings, "GRAPH_WORKER_GRAPHS", ("rag.default",))
)


def _should_enqueue_graph(graph_name: str) -> bool:
    return graph_name in ASYNC_GRAPH_NAMES


# Lightweight ledger shim so tests can monkeypatch services.ledger.record(...)
class _LedgerShim:
    def record(self, meta):  # type: ignore[no-untyped-def]
        # No-op by default; tests replace this with a spy.
        try:
            _ = dict(meta)  # force materialization for safety
        except Exception:
            pass
        return None


ledger = _LedgerShim()


_DOCUMENTS_REPOSITORY: DocumentsRepository | None = None


# Helper: make arbitrary payloads JSON-serialisable (UUIDs → strings)
def _make_json_safe(value):  # type: ignore[no-untyped-def]
    """Return a structure that json.dumps can serialise.

    - Converts uuid.UUID instances to str
    - Recurses into mappings and sequences
    - Normalises datetime/date instances to ISO strings
    """
    import uuid as _uuid
    from collections.abc import Mapping as _Mapping
    from dataclasses import asdict, is_dataclass
    from datetime import date as _date, datetime as _datetime

    try:
        from documents.parsers import ParsedResult

        if isinstance(value, ParsedResult):
            return _make_json_safe(asdict(value))
    except (ImportError, TypeError):
        pass

    if hasattr(value, "model_dump"):
        try:
            return _make_json_safe(value.model_dump())
        except Exception:  # pragma: no cover - defensive conversion
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return _make_json_safe(asdict(value))
    if isinstance(value, _uuid.UUID):
        return str(value)
    if isinstance(value, (_datetime, _date)):
        return value.isoformat()
    if isinstance(value, _Mapping):
        return {
            (str(k) if isinstance(k, _uuid.UUID) else k): _make_json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(
        value, (str, bytes, bytearray, type)
    ):
        return _make_json_safe(vars(value))
    return value


# Allow tests to monkeypatch the run_ingestion task via ai_core.views.run_ingestion
# while keeping a sane default binding for production code paths.
RUN_INGESTION = run_ingestion


def _get_run_ingestion_task():  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        task = getattr(views, "run_ingestion", None)
        if task is not None:
            return task
    except Exception:
        pass
    return RUN_INGESTION


def _task_accepts_state(task: object) -> bool:
    run_fn = getattr(task, "run", None)
    if run_fn is None:
        return False
    try:
        params = signature(run_fn).parameters
    except (TypeError, ValueError):
        return False
    return "state" in params


def _enqueue_ingestion_task(
    task: object,
    *,
    state: Mapping[str, object],
    meta: Mapping[str, object],
    legacy_args: tuple[object, ...],
    legacy_kwargs: Mapping[str, object],
) -> None:
    if _task_accepts_state(task):
        # task.delay(state, meta)
        signature = task.s(state, meta)
        scope = meta.get("scope_context")
        scope_dict = dict(scope) if isinstance(scope, Mapping) else {}
        with_scope_apply_async(signature, scope_dict)
        return
    task.delay(*legacy_args, **legacy_kwargs)


def _get_partition_document_ids():  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        fn = getattr(views, "partition_document_ids", None)
        if callable(fn):
            return fn
    except Exception:
        pass
    return partition_document_ids


def _get_checkpointer():  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        cp = getattr(views, "CHECKPOINTER", None)
        if cp is not None:
            return cp
    except Exception:
        pass
    return CHECKPOINTER


def _build_documents_repository() -> DocumentsRepository:
    """Build the appropriate documents repository."""
    if settings.TESTING:
        return InMemoryDocumentsRepository()

    # Check for explicit repository class configuration
    repository_class_path = getattr(settings, "DOCUMENTS_REPOSITORY_CLASS", None)
    if repository_class_path:
        logger.info(
            "documents_repository_configured",
            extra={"repository_class": repository_class_path},
        )
        try:
            module_path, class_name = repository_class_path.rsplit(".", 1)
            module = import_module(module_path)
            repository_class = getattr(module, class_name)
            return repository_class()
        except Exception:
            logger.exception(
                "documents_repository_instantiation_failed",
                extra={"repository_class": repository_class_path},
            )
            # Don't silently fall back - raise the error
            raise RuntimeError(
                f"Failed to instantiate repository class: {repository_class_path}"
            )

    # Default: Use DB repository for all non-test environments
    logger.info(
        "documents_repository_default",
        extra={"repository_class": "DbDocumentsRepository"},
    )
    return DbDocumentsRepository()


def _get_documents_repository() -> DocumentsRepository:
    try:
        views = import_module("ai_core.views")
        repo = getattr(views, "DOCUMENTS_REPOSITORY", None)
        if isinstance(repo, DocumentsRepository):
            return repo
    except Exception:
        pass

    global _DOCUMENTS_REPOSITORY
    if _DOCUMENTS_REPOSITORY is None:
        _DOCUMENTS_REPOSITORY = _build_documents_repository()
    return _DOCUMENTS_REPOSITORY


class DocumentComponents:
    """Container for document processing pipeline components."""

    def __init__(self, storage, captioner):
        self.storage = storage
        self.captioner = captioner


def get_document_components() -> DocumentComponents:
    """Return default document processing components.

    Provides storage and captioner instances used by the document processing graph.
    """
    from documents.storage import ObjectStoreStorage
    from documents.captioning import DeterministicCaptioner

    return DocumentComponents(
        storage=ObjectStoreStorage,
        captioner=DeterministicCaptioner,
    )


def _extract_initial_cost(meta: Mapping[str, Any]) -> float | None:
    cost_block = meta.get("cost")
    if isinstance(cost_block, Mapping):
        for key in ("total_usd", "usd", "total"):
            cost_value = cost_block.get(key)
            coerced = coerce_cost_value(cost_value)
            if coerced is not None:
                return coerced
    for key in ("cost_total_usd", "cost_usd", "total_cost_usd"):
        if key in meta:
            coerced = coerce_cost_value(meta[key])
            if coerced is not None:
                return coerced
    return None


def _extract_ledger_identifier(meta: Mapping[str, Any]) -> str | None:
    direct = meta.get("ledger_id") or meta.get("ledgerId")
    if direct:
        return str(direct)
    ledger_block = meta.get("ledger")
    if isinstance(ledger_block, Mapping):
        candidate = ledger_block.get("id") or ledger_block.get("ledger_id")
        if candidate:
            return str(candidate)
    return None


def _normalize_meta(request):  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        fn = getattr(views, "normalize_meta", None)
        if callable(fn):
            return fn(request)
    except Exception:
        pass
    return _base_normalize_meta(request)


def _resolve_ingestion_profile(profile: str):  # type: ignore[no-untyped-def]
    try:
        views = import_module("ai_core.views")
        fn = getattr(views, "resolve_ingestion_profile", None)
        if callable(fn):
            return fn(profile)
    except Exception:
        pass
    return _base_resolve_ingestion_profile(profile)


def _apply_collection_header_bridge(
    request: Request, payload: Mapping[str, object] | None
) -> dict[str, object]:
    data = dict(payload or {})

    header_value: str | None = None
    headers = getattr(request, "headers", None)
    if isinstance(headers, Mapping):
        for candidate_key in COLLECTION_ID_HEADER_CANDIDATES:
            candidate = headers.get(candidate_key)
            if candidate is None:
                continue
            if not isinstance(candidate, str):
                candidate = str(candidate)
            candidate = candidate.strip()
            if candidate:
                header_value = candidate
                break
    if header_value is None:
        meta = getattr(request, "META", None)
        if isinstance(meta, Mapping):
            candidate = meta.get(META_COLLECTION_ID_KEY)
            if isinstance(candidate, str):
                header_value = candidate.strip() or None

    if not header_value:
        return data

    body_value = data.get("collection_id")
    body_present = False
    if isinstance(body_value, str):
        if body_value.strip():
            body_present = True
        else:
            body_value = None
    elif body_value not in (None, ""):
        body_present = True

    filters_value = data.get("filters")
    filter_has_list = False
    collection_scope = "none"
    if isinstance(filters_value, Mapping):
        candidates = filters_value.get("collection_ids")
        if candidates:
            filter_has_list = True
            collection_scope = "list"
        single_filter = filters_value.get("collection_id")
        if single_filter is not None:
            try:
                if str(single_filter).strip():
                    body_present = True
                    if not filter_has_list:
                        collection_scope = "single"
            except Exception:
                body_present = True
                if not filter_has_list:
                    collection_scope = "single"

    if not filter_has_list and body_present:
        collection_scope = "single"

    if not body_present and not filter_has_list:
        data["collection_id"] = header_value
    else:
        reason = "filter_list_present" if filter_has_list else "body_present"
        logger.debug(
            "collection header ignored due to %s (collection_scope=%s)",
            reason,
            collection_scope,
            extra={
                "reason": reason,
                "header_present": True,
                "collection_scope": collection_scope,
            },
        )

    return data


def _persist_collection_scope(
    tenant_id: str, case_id: str, document_ids: Iterable[str], collection_id: str
) -> None:
    tenant_segment = object_store.sanitize_identifier(tenant_id)
    case_segment = object_store.sanitize_identifier(case_id or "uncased")
    for document_id in document_ids:
        if not document_id:
            continue
        path = f"{tenant_segment}/{case_segment}/uploads/{document_id}.meta.json"
        try:
            metadata = object_store.read_json(path)
        except FileNotFoundError:
            continue
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["collection_id"] = collection_id
        object_store.write_json(path, metadata)


def _ensure_collection_with_warning(
    service: DocumentDomainService,
    tenant,
    identifier: object,
    *,
    embedding_profile: str | None,
    scope: str | None,
) -> DocumentCollection | None:
    """Ensure a collection exists; create missing IDs with a warning (review later)."""

    try:
        collection_uuid = UUID(str(identifier))
    except Exception:
        collection_uuid = None

    if collection_uuid is not None:
        exists = DocumentCollection.objects.filter(
            tenant=tenant, collection_id=collection_uuid
        ).exists()
        if not exists:
            logger.warning(
                "documents.collection_missing_created",
                extra={
                    "tenant_id": str(getattr(tenant, "id", tenant)),
                    "collection_id": str(collection_uuid),
                    "reason": "missing_reference",
                },
            )

    return service.ensure_collection(
        tenant=tenant,
        key=str(identifier),
        embedding_profile=embedding_profile,
        scope=scope,
        collection_id=collection_uuid,
    )


def _ensure_document_collection(
    *,
    collection_id: object,
    tenant_identifier: object,
    case_identifier: object | None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    try:
        collection_uuid = UUID(str(collection_id))
    except (TypeError, ValueError):
        logger.debug(
            "collection_model.invalid_collection_id",
            extra={"collection_id": collection_id},
        )
        return

    from customers.tenant_context import TenantContext

    try:
        tenant = TenantContext.resolve_identifier(tenant_identifier, allow_pk=True)
    except Exception:
        logger.exception(
            "collection_model.tenant_resolution_failed",
            extra={"collection_id": str(collection_uuid)},
        )
        return

    if tenant is None:
        logger.info(
            "collection_model.missing_tenant",
            extra={"collection_id": str(collection_uuid)},
        )
        return

    try:
        from cases.models import Case
        from documents.models import DocumentCollection
    except Exception:
        logger.exception(
            "collection_model.import_failed",
            extra={"collection_id": str(collection_uuid)},
        )
        return

    meta_payload = metadata if isinstance(metadata, Mapping) else {}

    def _coerce_label(*keys: str) -> str | None:
        for key in keys:
            value = meta_payload.get(key)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
        return None

    name = _coerce_label("collection_name", "name", "label", "title")
    key = _coerce_label("collection_key", "key", "slug")
    fallback_label = str(collection_uuid)
    if not name:
        name = fallback_label
    if not key:
        key = name

    case_obj = None
    if case_identifier:
        case_identifier_str = str(case_identifier)
        case_obj = Case.objects.filter(
            external_id=case_identifier_str, tenant=tenant
        ).first()

        if case_obj is None:
            try:
                case_uuid = UUID(case_identifier_str)
            except Exception:
                logger.debug(
                    "collection_model.case_resolution_failed",
                    extra={
                        "collection_id": str(collection_uuid),
                        "case_identifier": case_identifier,
                    },
                )
            else:
                case_obj = Case.objects.filter(id=case_uuid, tenant=tenant).first()

    try:
        DocumentCollection.objects.get_or_create(
            id=collection_uuid,
            defaults={
                "tenant": tenant,
                "case": case_obj,
                "collection_id": collection_uuid,
                "name": name,
                "key": key,
                "metadata": meta_payload,
                "type": "",
                "visibility": "",
            },
        )
    except Exception:
        logger.exception(
            "collection_model.get_or_create_failed",
            extra={
                "collection_id": str(collection_uuid),
                "tenant_id": getattr(tenant, "schema_name", str(tenant)),
            },
        )


def _coerce_transition_result(
    transition: object,
) -> StandardTransitionResult | None:
    if isinstance(transition, GraphTransition):
        return transition.result
    if isinstance(transition, StandardTransitionResult):
        return transition
    if isinstance(transition, Mapping):
        try:
            return StandardTransitionResult.model_validate(transition)
        except ValidationError:
            return None
    return None


def _map_upload_graph_skip(
    decision: str, transitions: Mapping[str, object]
) -> Response:
    """Translate upload graph skip decisions into HTTP error responses."""

    accept_transition = _coerce_transition_result(transitions.get("accept_upload"))
    guardrail_transition = _coerce_transition_result(
        transitions.get("delta_and_guardrails")
    )
    guardrail_section = guardrail_transition.guardrail if guardrail_transition else None

    if decision == "skip_guardrail":
        policy_events: tuple[str, ...]
        if guardrail_section is not None:
            policy_events = guardrail_section.policy_events
        else:
            policy_events = ()
        if policy_events:
            policy_text = ", ".join(str(event) for event in policy_events)
            detail = f"Upload blocked by guardrails: {policy_text}."
        else:
            detail = "Upload blocked by guardrails."
        return _error_response(
            detail,
            "upload_blocked",
            status.HTTP_403_FORBIDDEN,
        )

    if decision == "skip_duplicate":
        return _error_response(
            "Duplicate upload detected for this document.",
            "duplicate_upload",
            status.HTTP_409_CONFLICT,
        )

    if decision == "skip_disallowed_mime":
        mime = None
        if accept_transition is not None:
            context = accept_transition.context
            if isinstance(context, Mapping):
                candidate = context.get("mime")
                if candidate is not None:
                    mime = str(candidate)
        if mime:
            detail = f"MIME type '{mime}' is not allowed for uploads."
        else:
            detail = "MIME type is not allowed for uploads."
        return _error_response(
            detail,
            "mime_not_allowed",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    if decision == "skip_oversize":
        max_bytes: int | None = None
        if accept_transition is not None:
            context = accept_transition.context
            if isinstance(context, Mapping):
                candidate = context.get("max_bytes")
                if isinstance(candidate, (int, float)):
                    max_bytes = int(candidate)
        if max_bytes is not None:
            detail = f"Uploaded file exceeds the allowed size of {max_bytes} bytes."
        else:
            detail = "Uploaded file exceeds the allowed size."
        return _error_response(
            detail,
            "file_too_large",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    if decision == "skip_invalid_input":
        reason = accept_transition.reason if accept_transition else "unknown"
        return _error_response(
            f"Upload payload is invalid: {reason}",
            "invalid_upload_payload",
            status.HTTP_400_BAD_REQUEST,
        )

    return _error_response(
        "Upload rejected due to validation failure.",
        "upload_rejected",
        status.HTTP_400_BAD_REQUEST,
    )


GRAPH_REQUEST_MODELS = {
    "info_intake": InfoIntakeRequest,
    "rag.default": RagQueryRequest,
}


def _log_graph_response_payload(payload: object, context: GraphContext) -> None:
    """Emit diagnostics about the response payload produced by a graph run."""

    try:
        payload_json = json.dumps(_make_json_safe(payload), ensure_ascii=False)
    except TypeError:
        logger.exception(
            "graph.response_payload_serialization_error",
            extra={
                "graph": context.graph_name,
                "tenant_id": context.tenant_id,
                "case_id": context.case_id,
                "payload_type": type(payload).__name__,
            },
        )
        raise

    logger.info(
        "graph.response_payload",
        extra={
            "graph": context.graph_name,
            "tenant_id": context.tenant_id,
            "case_id": context.case_id,
            "payload_json": payload_json,
        },
    )


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


@observe_span(name="graph.execute")
def execute_graph(request: Request, graph_runner: GraphRunner) -> Response:
    """
    Orchestrates the execution of a graph, handling context, state, and errors.

    This function contains the business logic that was previously in the
    `_run_graph` function in the view layer.
    """
    try:
        normalized_meta = _normalize_meta(request)
    except ValueError as exc:
        error_msg = str(exc)
        # Use specific error code for case header validation errors
        error_code = (
            "invalid_case_header" if "Case header" in error_msg else "invalid_request"
        )
        return _error_response(error_msg, error_code, status.HTTP_400_BAD_REQUEST)

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

    scope_context = normalized_meta["scope_context"]
    # BREAKING CHANGE (Option A): Extract business IDs from business_context
    business_context = normalized_meta.get("business_context", {})
    run_id = uuid4().hex
    workflow_id = business_context.get("workflow_id") or business_context.get("case_id")

    context = GraphContext(
        tenant_id=scope_context["tenant_id"],
        case_id=business_context.get("case_id"),
        trace_id=scope_context["trace_id"],
        workflow_id=workflow_id,
        run_id=run_id,
        graph_name=normalized_meta["graph_name"],
        graph_version=normalized_meta["graph_version"],
    )

    ledger_identifier = _extract_ledger_identifier(normalized_meta)
    initial_cost_total = _extract_initial_cost(normalized_meta)
    base_observation_metadata: dict[str, Any] = {
        "trace_id": context.trace_id,
        "tenant.id": context.tenant_id,
        "case.id": context.case_id,
        "graph.version": context.graph_version,
        "workflow.id": context.workflow_id,
        "run.id": context.run_id,
    }
    if ledger_identifier:
        base_observation_metadata["ledger.id"] = ledger_identifier
    if initial_cost_total is not None:
        base_observation_metadata["cost.total_usd"] = initial_cost_total
    observation_kwargs = {
        "tags": [
            "graph",
            f"graph:{context.graph_name}",
            f"version:{context.graph_version}",
        ],
        "user_id": str(context.tenant_id),
        "session_id": str(context.case_id),
        "metadata": dict(base_observation_metadata),
    }

    req_started = time.monotonic()
    try:
        logger.info(
            "graph.request.start",
            extra={
                "graph": context.graph_name,
                "tenant_id": context.tenant_id,
                "case_id": context.case_id,
                "trace_id": context.trace_id,
            },
        )
    except Exception:
        pass

    # Start a root trace (best-effort) only when tracing is enabled
    trace_started = False
    if lf_tracing_enabled():
        try:
            lf_start_trace(
                name=f"graph:{context.graph_name}",
                user_id=str(context.tenant_id),
                session_id=str(context.case_id),
                metadata={
                    "trace_id": context.trace_id,
                    "version": context.graph_version,
                    "workflow_id": context.workflow_id,
                    "run_id": context.run_id,
                },
            )
            trace_started = True
        except Exception:
            pass
        if trace_started:
            try:
                update_observation(**observation_kwargs)
            except Exception:
                pass

    cost_summary: dict[str, Any] | None = None

    try:
        # Attach graph context to observation (no request body content)
        try:
            update_observation(**observation_kwargs)
        except Exception:
            pass

        try:
            state = _get_checkpointer().load(context)
        except (TypeError, ValueError) as exc:
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )

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

        data = _apply_collection_header_bridge(request, incoming_state)

        request_model = GRAPH_REQUEST_MODELS.get(context.graph_name)
        if request_model is not None:
            try:
                validated = request_model.model_validate(data)
            except ValidationError as exc:
                return _error_response(
                    _format_validation_error(exc),
                    "invalid_request",
                    status.HTTP_400_BAD_REQUEST,
                )
            incoming_state = validated.model_dump(exclude_none=True)
        else:
            incoming_state = data

        merged_state = merge_state(state, incoming_state)

        runner_meta = dict(normalized_meta)
        if normalized_meta.get("tenant_schema"):
            runner_meta["tenant_schema"] = normalized_meta["tenant_schema"]
        if normalized_meta.get("key_alias"):
            runner_meta["key_alias"] = normalized_meta["key_alias"]

        try:
            t0 = time.monotonic()
            try:
                logger.info(
                    "graph.run.start",
                    extra={
                        "graph": context.graph_name,
                        "tenant_id": context.tenant_id,
                        "case_id": context.case_id,
                    },
                )
            except Exception:
                pass

            if _should_enqueue_graph(context.graph_name):
                signature = current_app.signature(
                    "llm_worker.tasks.run_graph",
                    kwargs={
                        "graph_name": context.graph_name,
                        "state": merged_state,
                        "meta": runner_meta,
                        "ledger_identifier": ledger_identifier,
                        "initial_cost_total": initial_cost_total,
                    },
                    queue="agents",
                )
                scope = {
                    "tenant_id": context.tenant_id,
                    "case_id": context.case_id,
                    "trace_id": context.trace_id,
                }
                async_result = with_scope_apply_async(signature, scope)

                try:
                    # Wait for the task with timeout
                    # Controlled via settings.GRAPH_WORKER_TIMEOUT_S
                    timeout_s = getattr(settings, "GRAPH_WORKER_TIMEOUT_S", 45)
                    task_payload = async_result.get(timeout=timeout_s, propagate=True)
                    new_state = task_payload["state"]
                    result = task_payload["result"]
                    cost_summary = task_payload.get("cost_summary")
                    # Round total_usd to 4 decimal places to reduce noise in logs/traces
                    if cost_summary and "total_usd" in cost_summary:
                        cost_summary["total_usd"] = round(cost_summary["total_usd"], 4)
                except celery_exceptions.TimeoutError:
                    # Task did not complete within timeout - return 202 Accepted
                    logger.warning(
                        "graph.worker_timeout",
                        extra={
                            "graph": context.graph_name,
                            "tenant_id": context.tenant_id,
                            "case_id": context.case_id,
                            "task_id": async_result.id,
                            "timeout_s": timeout_s,
                        },
                    )
                    return Response(
                        {
                            "status": "queued",
                            "task_id": async_result.id,
                            "graph": context.graph_name,
                            "tenant_id": context.tenant_id,
                            "case_id": context.case_id,
                            "trace_id": context.trace_id,
                        },
                        status=status.HTTP_202_ACCEPTED,
                    )
            else:
                with track_ledger_costs(initial_cost_total) as tracker:
                    runner_meta["ledger_logger"] = tracker.record_ledger_meta
                    try:
                        new_state, result = graph_runner.run(merged_state, runner_meta)
                    finally:
                        runner_meta.pop("ledger_logger", None)
                cost_summary = tracker.summary(ledger_identifier)
                # Round total_usd to 4 decimal places to reduce noise in logs/traces
                if cost_summary and "total_usd" in cost_summary:
                    cost_summary["total_usd"] = round(cost_summary["total_usd"], 4)

            try:
                dt_ms = int((time.monotonic() - t0) * 1000)
                logger.info(
                    "graph.run.end",
                    extra={
                        "graph": context.graph_name,
                        "tenant_id": context.tenant_id,
                        "case_id": context.case_id,
                        "duration_ms": dt_ms,
                    },
                )
            except Exception:
                pass
        except InputError as exc:
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )
        except ToolContextError as exc:
            # Invalid execution context (e.g., router/tenant/case issues) should be
            # surfaced to clients as a 400 rather than a transient 503.
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )
        except ValueError as exc:
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )
        except ToolNotFoundError as exc:
            logger.info("tool.not_found")
            detail = str(exc) or "No matching documents were found."
            return _error_response(detail, "rag_no_matches", status.HTTP_404_NOT_FOUND)
        except ToolInconsistentMetadataError as exc:
            logger.warning("tool.inconsistent_metadata")
            payload = {
                "detail": str(exc) or "reindex required",
                "code": "retrieval_inconsistent_metadata",
            }
            context = getattr(exc, "context", None)
            if context:
                payload["context"] = context
            return Response(payload, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except ToolInputError as exc:
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )
        except ToolRateLimitedError as _exc:
            logger.warning("tool.rate_limited")
            return _error_response(
                "Tool rate limited.",
                "llm_rate_limited",
                status.HTTP_429_TOO_MANY_REQUESTS,
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
            # Log unexpected exceptions with execution context for diagnostics.
            try:
                logger.exception(
                    "graph.execution_failed",
                    extra={
                        "graph": context.graph_name,
                        "tenant_id": context.tenant_id,
                        "case_id": context.case_id,
                    },
                )
            except Exception:
                pass
            return _error_response(
                "Service temporarily unavailable.",
                "service_unavailable",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            _get_checkpointer().save(context, _make_json_safe(new_state))
        except (TypeError, ValueError) as exc:
            return _error_response(
                str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST
            )

        try:
            _log_graph_response_payload(result, context)
        except TypeError:
            raise
        except Exception:
            logger.exception(
                "graph.response_payload_logging_failed",
                extra={
                    "graph": context.graph_name,
                    "tenant_id": context.tenant_id,
                    "case_id": context.case_id,
                },
            )

        try:
            response = Response(_make_json_safe(result))
        except TypeError:
            logger.exception(
                "graph.response_serialization_error",
                extra={
                    "graph": context.graph_name,
                    "tenant_id": context.tenant_id,
                    "case_id": context.case_id,
                    "payload_type": type(result).__name__,
                },
            )
            raise

        if cost_summary:
            final_metadata = dict(base_observation_metadata)
            final_metadata["cost.total_usd"] = cost_summary["total_usd"]
            try:
                update_observation(metadata=final_metadata)
            except Exception:
                pass
            event_payload = {
                "total_usd": cost_summary["total_usd"],
                "components": cost_summary["components"],
                "tenant_id": context.tenant_id,
                "case_id": context.case_id,
                "graph_name": context.graph_name,
                "graph_version": context.graph_version,
            }
            if "reconciliation" in cost_summary:
                event_payload["reconciliation"] = cost_summary["reconciliation"]
            try:
                emit_event("cost.summary", event_payload)
            except Exception:
                pass

        return response
    finally:
        try:
            lf_end_trace()
        except Exception:
            pass
        try:
            dt_total_ms = int((time.monotonic() - req_started) * 1000)
            logger.info(
                "graph.request.end",
                extra={
                    "graph": context.graph_name,
                    "tenant_id": context.tenant_id,
                    "case_id": context.case_id,
                    "duration_ms": dt_total_ms,
                },
            )
        except Exception:
            pass


def start_ingestion_run(
    request_data: dict, meta: dict, idempotency_key: str | None
) -> Response:
    """
    Orchestrates the start of a RAG ingestion run.

    This function contains the business logic that was previously in the
    `RagIngestionRunView`.
    """
    try:
        validated_data = RagIngestionRunRequest.model_validate(request_data)
    except ValidationError as exc:
        # NOTE: Consider a more detailed error mapping for production
        return _error_response(
            str(exc), "validation_error", status.HTTP_400_BAD_REQUEST
        )

    scope_context = meta["scope_context"]
    # BREAKING CHANGE (Option A): Extract business IDs from business_context
    business_context = meta.get("business_context", {})
    tenant_schema = scope_context.get("tenant_schema") or meta.get("tenant_schema")

    collection_scope = getattr(validated_data, "collection_id", None)
    # BREAKING CHANGE (Option A): collection_id goes to business_context, not scope_context
    if collection_scope and isinstance(business_context, dict):
        business_context["collection_id"] = collection_scope

    if collection_scope:
        _ensure_document_collection(
            collection_id=collection_scope,
            tenant_identifier=tenant_schema or scope_context.get("tenant_id"),
            case_identifier=business_context.get("case_id"),
            metadata=request_data if isinstance(request_data, Mapping) else None,
        )

    try:
        normalized_profile = (
            str(validated_data.embedding_profile).strip()
            if hasattr(validated_data, "embedding_profile")
            else str(validated_data.get("embedding_profile", "")).strip()
        )
        profile_binding = _resolve_ingestion_profile(normalized_profile)
    except InputError as exc:
        return _error_response(
            exc.message,
            exc.code,
            map_ingestion_error_to_status(exc.code),
        )
    resolved_profile_id = profile_binding.profile_id

    ingestion_run_id = uuid4().hex
    queued_at = timezone.now().isoformat()

    # BREAKING CHANGE (Option A): case_id from business_context
    valid_document_ids, invalid_document_ids = _get_partition_document_ids()(
        scope_context["tenant_id"],
        business_context.get("case_id"),
        validated_data.document_ids,
    )

    to_dispatch = (
        valid_document_ids if valid_document_ids else validated_data.document_ids
    )
    if collection_scope:
        _persist_collection_scope(
            scope_context["tenant_id"],
            business_context.get("case_id"),
            to_dispatch,
            collection_scope,
        )
    if isinstance(scope_context, dict):
        scope_context.setdefault("ingestion_run_id", ingestion_run_id)
    state_payload: dict[str, object] = {
        "tenant_id": scope_context["tenant_id"],
        "case_id": business_context.get("case_id"),
        "document_ids": to_dispatch,
        "embedding_profile": resolved_profile_id,
        "tenant_schema": tenant_schema,
        "run_id": ingestion_run_id,
        "trace_id": scope_context["trace_id"],
    }
    if collection_scope:
        state_payload["collection_id"] = collection_scope
    # BREAKING CHANGE (Option A): case_id from business_context
    _enqueue_ingestion_task(
        _get_run_ingestion_task(),
        state=state_payload,
        meta=dict(meta),
        legacy_args=(
            scope_context["tenant_id"],
            business_context.get("case_id"),
            to_dispatch,
            resolved_profile_id,
        ),
        legacy_kwargs={
            "tenant_schema": tenant_schema,
            "run_id": ingestion_run_id,
            "trace_id": scope_context["trace_id"],
            "idempotency_key": idempotency_key,
        },
    )

    record_ingestion_run_queued(
        tenant_id=scope_context["tenant_id"],
        case=business_context.get("case_id"),
        run_id=ingestion_run_id,
        document_ids=to_dispatch,
        queued_at=queued_at,
        trace_id=scope_context["trace_id"],
        embedding_profile=validated_data.embedding_profile,
        source=validated_data.source,
    )
    emit_ingestion_case_event(
        scope_context["tenant_id"],
        business_context.get("case_id"),
        run_id=ingestion_run_id,
        context="queued",
    )

    idempotent = bool(idempotency_key)
    response_payload = {
        "status": "queued",
        "queued_at": queued_at,
        "ingestion_run_id": ingestion_run_id,
        "trace_id": scope_context["trace_id"],
        "idempotent": idempotent,
    }

    if invalid_document_ids:
        response_payload["invalid_ids"] = invalid_document_ids
    else:
        response_payload["invalid_ids"] = []

    return Response(response_payload, status=status.HTTP_202_ACCEPTED)


def _derive_workflow_id(
    scope_context: Mapping[str, object],
    metadata: Mapping[str, object],
    business_context: Mapping[str, object] | None = None,
) -> str:
    """Derive workflow ID from metadata or business context.

    BREAKING CHANGE (Option A): case_id is now in business_context, not scope_context.
    """
    candidate = metadata.get("workflow_id")
    if isinstance(candidate, str):
        candidate = candidate.strip()
        if candidate:
            return candidate.replace(":", "_")

    # BREAKING CHANGE (Option A): use business_context only
    case_id = ""
    if business_context:
        case_id = str(business_context.get("case_id") or "").strip()

    if not case_id:
        return "upload"
    return case_id.replace(":", "_")


def _infer_media_type(upload: UploadedFile) -> str:
    content_type = getattr(upload, "content_type", None)
    if isinstance(content_type, str):
        content_type = content_type.split(";")[0].strip().lower()
    if not content_type:
        return "application/octet-stream"
    return content_type


def _build_document_meta(
    scope_context: Mapping[str, object],
    metadata_obj: Mapping[str, object],
    external_id: str,
    *,
    media_type: str | None = None,
    business_context: Mapping[str, object] | None = None,
) -> DocumentMeta:
    """Build DocumentMeta from scope and metadata.

    BREAKING CHANGE (Option A): business_context parameter added for workflow_id derivation.
    """
    workflow_id = _derive_workflow_id(scope_context, metadata_obj, business_context)
    payload: dict[str, object] = {
        "tenant_id": str(scope_context["tenant_id"]),
        "workflow_id": workflow_id,
    }

    optional_fields = (
        "title",
        "language",
        "tags",
        "origin_uri",
        "crawl_timestamp",
        "external_ref",
        "parse_stats",
        "pipeline_config",
    )
    for field in optional_fields:
        if field in metadata_obj:
            payload[field] = metadata_obj[field]

    external_ref = dict(payload.get("external_ref") or {})
    if external_id:
        external_ref.setdefault("external_id", external_id)
    if media_type:
        external_ref.setdefault("media_type", media_type)
    if external_ref:
        payload["external_ref"] = external_ref

    return DocumentMeta(**payload)


def _ensure_document_collection_record(
    *,
    scope_context: dict,
    collection_id: UUID | str,
    source: str | None = None,
    key: str | None = None,
    label: str | None = None,
    business_context: dict | None = None,
) -> None:
    """Create or update a DocumentCollection row for manual developer uploads.

    BREAKING CHANGE (Option A): case_id is now in business_context, not scope_context.
    """

    if not collection_id:
        return
    try:
        collection_uuid = (
            collection_id
            if isinstance(collection_id, UUID)
            else UUID(str(collection_id))
        )
    except (TypeError, ValueError, AttributeError):
        extra_payload = {"scope_context": scope_context}
        extra_payload["collection_id"] = collection_id
        logger.warning(
            "document_collection.ensure.invalid_id",
            extra=extra_payload,
        )
        return

    tenant_identifier = scope_context.get("tenant_id")
    tenant_obj = None
    if tenant_identifier:
        try:
            tenant_obj = TenantContext.resolve_identifier(
                tenant_identifier, allow_pk=True
            )
        except Exception:
            logger.exception(
                "document_collection.ensure_tenant_resolution_failed",
                extra={"tenant_id": tenant_identifier},
            )
            return

    if tenant_obj is None:
        logger.warning(
            "document_collection.ensure_missing_tenant",
            extra={"tenant_id": tenant_identifier},
        )
        return

    case_obj = None
    # BREAKING CHANGE (Option A): use business_context only
    case_id = business_context.get("case_id") if business_context else None
    if case_id:
        case_obj = Case.objects.filter(
            tenant=tenant_obj,
            external_id=str(case_id).strip(),
        ).first()

    collection = DocumentCollection.objects.filter(
        tenant=tenant_obj, collection_id=collection_uuid
    ).first()
    if collection is None and key:
        collection = DocumentCollection.objects.filter(
            tenant=tenant_obj, key=key
        ).first()

    metadata_payload = {"source": source} if source else {}

    if collection is None:
        try:
            DocumentCollection.objects.create(
                tenant=tenant_obj,
                case=case_obj,
                name=label or key or "Manual Collection",
                key=key or str(collection_uuid),
                collection_id=collection_uuid,
                type="manual",
                visibility="tenant",
                metadata=metadata_payload,
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "document_collection.ensure_record_failed",
                extra={
                    "tenant_id": tenant_identifier,
                    "collection_id": str(collection_uuid),
                },
                exc_info=True,
            )
        return

    updates: dict[str, object] = {}
    if collection.collection_id != collection_uuid:
        updates["collection_id"] = collection_uuid
    if case_obj is not None and collection.case_id != case_obj.id:
        updates["case"] = case_obj
    if key and collection.key != key:
        updates["key"] = key
    if label and collection.name != label:
        updates["name"] = label
    if not collection.type:
        updates["type"] = "manual"
    if not collection.visibility:
        updates["visibility"] = "tenant"
    if metadata_payload:
        merged_metadata = dict(collection.metadata or {})
        if merged_metadata.get("source") != source:
            merged_metadata["source"] = source
            updates["metadata"] = merged_metadata

    if updates:
        for field, value in updates.items():
            setattr(collection, field, value)
        try:
            collection.save(update_fields=list(updates))
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "document_collection.ensure_record_update_failed",
                extra={
                    "tenant_id": tenant_identifier,
                    "tenant_schema": getattr(tenant_obj, "schema_name", None),
                    "collection_id": str(collection_uuid),
                },
                exc_info=True,
            )


def handle_document_upload(
    upload: UploadedFile,
    metadata_raw: str | bytes | None,
    meta: dict,
    idempotency_key: str | None,
) -> Response:
    """
    Orchestrates the upload of a document, its metadata, and immediate ingestion.

    This function contains the business logic that was previously in the
    `RagUploadView`.
    """
    metadata_obj: dict[str, object] | None = None
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

    # BREAKING CHANGE (Option A - Strict Separation):
    # scope_context is infrastructure only, business_context has business IDs
    scope_context = meta["scope_context"]
    business_context = meta.get("business_context", {})

    header_collection = business_context.get("collection_id")
    if header_collection and not metadata_obj.get("collection_id"):
        metadata_obj["collection_id"] = header_collection

    try:
        metadata_model = RagUploadMetadata.model_validate(metadata_obj)
    except ValidationError as exc:
        return _error_response(
            str(exc), "invalid_metadata", status.HTTP_400_BAD_REQUEST
        )

    metadata_obj = metadata_model.model_dump()

    tenant_identifier = scope_context.get("tenant_id")
    tenant_obj = None
    if tenant_identifier:
        try:
            tenant_obj = TenantContext.resolve_identifier(
                tenant_identifier, allow_pk=True
            )
        except Exception:
            pass

    if tenant_obj is None:
        return _error_response(
            "Tenant could not be resolved for upload.",
            "tenant_not_found",
            status.HTTP_400_BAD_REQUEST,
        )

    manual_collection_scope = str(manual_collection_uuid(tenant_obj))

    manual_scope_assigned = False

    existing_scope = metadata_obj.get("collection_id")
    if existing_scope:
        existing_scope_str = str(existing_scope)
        metadata_obj["collection_id"] = existing_scope_str
        if existing_scope_str == manual_collection_scope:
            manual_scope_assigned = True

    if not metadata_obj.get("collection_id"):
        manual_scope_assigned = True
        try:
            ensure_manual_collection(tenant_obj)
        except Exception:  # pragma: no cover - defensive guard
            # BREAKING CHANGE (Option A): case_id from business_context
            logger.warning(
                "upload.ensure_manual_collection_failed",
                extra={
                    "tenant_id": scope_context.get("tenant_id"),
                    "case_id": business_context.get("case_id"),
                },
                exc_info=True,
            )
        metadata_obj["collection_id"] = manual_collection_scope

    # BREAKING CHANGE (Option A): Pass business_context to _ensure_document_collection_record
    if manual_scope_assigned and metadata_obj.get("collection_id"):
        _ensure_document_collection_record(
            scope_context=scope_context,
            collection_id=metadata_obj["collection_id"],
            source="upload",
            key=MANUAL_COLLECTION_SLUG,
            label=MANUAL_COLLECTION_LABEL,
            business_context=business_context,
        )

    tenant = tenant_obj
    domain_service = (
        DocumentDomainService(vector_store=get_default_client()) if tenant else None
    )
    ensured_collection = None
    if metadata_obj.get("collection_id") and domain_service and tenant:
        if manual_scope_assigned:
            ensured_collection = domain_service.ensure_collection(
                tenant=tenant,
                key=MANUAL_COLLECTION_SLUG,
                collection_id=UUID(manual_collection_scope),
                embedding_profile=metadata_obj.get("embedding_profile"),
                scope=metadata_obj.get("scope"),
            )
        else:
            ensured_collection = _ensure_collection_with_warning(
                domain_service,
                tenant,
                metadata_obj["collection_id"],
                embedding_profile=metadata_obj.get("embedding_profile"),
                scope=metadata_obj.get("scope"),
            )
        if ensured_collection is not None:
            metadata_obj["collection_id"] = str(ensured_collection.collection_id)

    document_uuid = uuid4()

    file_bytes = upload.read()
    if not isinstance(file_bytes, (bytes, bytearray)):
        file_bytes = bytes(file_bytes)

    original_name = getattr(upload, "name", "") or "upload.bin"

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

    checksum = hashlib.sha256(file_bytes).hexdigest()
    encoded_blob = base64.b64encode(file_bytes).decode("ascii")

    # Improved MIME type detection
    detected_mime = _infer_media_type(upload)
    if detected_mime == "application/octet-stream":
        guessed_type, _ = mimetypes.guess_type(original_name)
        if guessed_type:
            detected_mime = guessed_type

    document_metadata_payload = dict(metadata_obj)
    document_metadata_payload.setdefault("filename", original_name)
    document_metadata_payload.setdefault("content_hash", checksum)
    document_metadata_payload.setdefault("content_type", detected_mime)

    audit_meta = {
        "created_by_user_id": scope_context.get("user_id"),
        "initiated_by_user_id": scope_context.get("user_id"),
        "last_hop_service_id": scope_context.get("service_id"),
    }
    audit_meta = {key: value for key, value in audit_meta.items() if value}

    if domain_service and tenant:
        ingest_result = domain_service.ingest_document(
            tenant=tenant,
            source=document_metadata_payload.get("origin_uri")
            or document_metadata_payload.get("external_id")
            or original_name,
            content_hash=checksum,
            metadata=document_metadata_payload,
            audit_meta=audit_meta,
            collections=(
                (ensured_collection,) if ensured_collection is not None else ()
            ),
            embedding_profile=metadata_obj.get("embedding_profile"),
            scope=metadata_obj.get("scope"),
            dispatcher=lambda *_: None,
        )
        document_uuid = ingest_result.document.id
        collection_ids = ingest_result.collection_ids
    else:
        document_uuid = uuid4()
        collection_ids: tuple[UUID, ...] = ()

    if not metadata_obj.get("title"):
        metadata_obj["title"] = original_name

    # BREAKING CHANGE (Option A): Pass business_context to _build_document_meta
    document_meta = _build_document_meta(
        scope_context,
        metadata_obj,
        external_id,
        media_type=detected_mime,
        business_context=business_context,
    )
    if isinstance(business_context, dict):
        business_context.setdefault("workflow_id", document_meta.workflow_id)

    print(
        f"DEBUG: collection_ids={collection_ids!r} metadata_obj_collection_id={metadata_obj.get('collection_id')!r}"
    )
    ref_payload: dict[str, object] = {
        "tenant_id": document_meta.tenant_id,
        "workflow_id": document_meta.workflow_id,
        "document_id": document_uuid,
        "collection_id": (
            metadata_obj.get("collection_id")
            if metadata_obj.get("collection_id")
            else (collection_ids[0] if collection_ids else None)
        ),
        "version": metadata_obj.get("version"),
    }
    document_ref = DocumentRef(**ref_payload)

    blob = InlineBlob(
        type="inline",
        media_type=detected_mime,
        base64=encoded_blob,
        sha256=checksum,
        size=len(file_bytes),
    )

    normalized_document = NormalizedDocument(
        ref=document_ref,
        meta=document_meta,
        blob=blob,
        checksum=checksum,
        created_at=timezone.now(),
        source="upload",
    )

    ingestion_run_id = uuid4().hex

    try:
        repository = _get_documents_repository()

        # Prepare input for Universal Ingestion Graph
        # We pre-build the NormalizedDocument here to handle Django-specific file handling
        graph_input = {
            "normalized_document": normalized_document.model_dump(),
        }

        # BREAKING CHANGE (Option A - Full ToolContext Migration):
        # Universal ingestion graph now expects nested ToolContext structure
        from ai_core.contracts.business import BusinessContext

        # Build BusinessContext from business_context dict
        business = BusinessContext(
            case_id=business_context.get("case_id"),
            workflow_id=business_context.get("workflow_id"),
            collection_id=(
                str(ensured_collection.collection_id)
                if ensured_collection
                else str(metadata_obj.get("collection_id"))
            ),
        )

        graph_state = {
            "input": graph_input,
            "context": {
                "scope": {
                    **scope_context,
                    "invocation_id": scope_context.get("invocation_id") or uuid4().hex,
                    "ingestion_run_id": ingestion_run_id,
                },
                "business": business.model_dump(mode="json"),
                "metadata": {
                    # Runtime dependencies passed in metadata
                    "runtime_repository": repository,
                    "audit_meta": audit_meta,
                },
            },
        }

        # Invoke Universal Ingestion Graph
        # Imports are lazy to avoid circular dependency issues during module load if any
        from ai_core.graphs.technical.universal_ingestion_graph import (
            build_universal_ingestion_graph,
            UniversalIngestionError as UploadIngestionError,
        )

        universal_graph = build_universal_ingestion_graph()
        result_state = universal_graph.invoke(graph_state)

        output = result_state.get("output") or {}
        decision = output.get("decision", "error")
        reason = output.get("reason", "unknown")

        # P1 Fix: Handle both 'error' and 'failed' decisions from the graph
        if decision in ("error", "failed"):
            # P2 Fix: Distinguish user validation errors from server errors
            # Validate/Normalize nodes return "Missing...", "Unsupported...", "Normalization failed..."
            is_user_error = any(
                x in reason
                for x in (
                    "Missing",
                    "Unsupported",
                    "Normalization failed",
                    "Could not verify",
                )
            )

            if is_user_error:
                logger.warning(
                    "Upload ingestion validation failed",
                    extra={
                        "reason": reason,
                        "tenant_id": scope_context.get("tenant_id"),
                    },
                )
                return _error_response(
                    reason,
                    "invalid_request",
                    status.HTTP_400_BAD_REQUEST,
                )

            raise UploadIngestionError(reason)

        # Map telemetry and IDs back
        graph_result = {
            "decision": decision,
            "reason": reason,
            "document_id": output.get("document_id"),
            "ingestion_run_id": output.get("ingestion_run_id"),
            "telemetry": output.get("telemetry", {}),
            # Transitions are legacy concepts, but we might need to mock them if downstream expects them
            "transitions": {},
        }

    except UploadIngestionError as exc:
        reason = str(exc)
        if reason == "document_persistence_failed":
            return _error_response(
                "Failed to persist uploaded document.",
                "document_persistence_failed",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        # ... (rest of error handling remains similar but simplified) ...
        # BREAKING CHANGE (Option A): case_id from business_context
        logger.exception(
            "Upload ingestion graph failed",
            extra={
                "tenant_id": scope_context.get("tenant_id"),
                "case_id": business_context.get("case_id"),
                "reason": reason,
            },
        )
        return _error_response(
            "Failed to process uploaded document.",
            "upload_graph_failed",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        # BREAKING CHANGE (Option A): case_id from business_context
        logger.exception(
            "Unexpected error while running upload ingestion graph",
            extra={
                "tenant_id": scope_context.get("tenant_id"),
                "case_id": business_context.get("case_id"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        return _error_response(
            "Failed to process uploaded document.",
            "upload_graph_failed",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    if decision.startswith("skip") or decision == "rejected":
        logger.info(
            "Upload ingestion skipped",
            extra={"decision": decision, "reason": graph_result.get("reason")},
        )
        return Response(
            {
                "status": "skipped",
                "decision": decision,
                "reason": graph_result.get("reason"),
            },
            status=status.HTTP_200_OK,
        )

    document_id = graph_result.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        # BREAKING CHANGE (Option A): case_id from business_context
        logger.error(
            "Upload ingestion graph completed without persisting document",
            extra={
                "tenant_id": scope_context.get("tenant_id"),
                "case_id": business_context.get("case_id"),
                "graph_decision": decision,
            },
        )
        return _error_response(
            "Failed to persist uploaded document.",
            "document_persistence_failed",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Post-processing: Log success
    # Universal Ingestion Graph has already completed persistence and processing.
    # No need to dispatch a separate task.
    ingestion_run_id = graph_result.get("ingestion_run_id") or uuid4().hex

    # BREAKING CHANGE (Option A): case_id and workflow_id from business_context
    logger.info(
        "Upload ingestion completed synchronously",
        extra={
            "tenant_id": scope_context.get("tenant_id"),
            "case_id": business_context.get("case_id"),
            "document_id": document_id,
            "run_id": ingestion_run_id,
        },
    )

    response_payload: dict[str, object] = {
        "trace_id": scope_context.get("trace_id"),
        "workflow_id": business_context.get("workflow_id"),
    }
    if document_id:
        response_payload["document_id"] = document_id
    if ingestion_run_id:
        response_payload["ingestion_run_id"] = ingestion_run_id
    if metadata_obj.get("collection_id"):
        response_payload["collection_id"] = str(metadata_obj["collection_id"])

    return Response(response_payload, status=status.HTTP_202_ACCEPTED)
