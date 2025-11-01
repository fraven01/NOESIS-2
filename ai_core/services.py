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
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from importlib import import_module
from typing import Any
from uuid import uuid4

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone
from django.utils.module_loading import import_string
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
from ai_core.graph.schemas import ToolContext, merge_state
from ai_core.graph.schemas import normalize_meta as _base_normalize_meta
from ai_core.llm.client import LlmClientError, RateLimitError
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

from ai_core.graphs.upload_ingestion_graph import (
    UploadIngestionError,
    UploadIngestionGraph,
)
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
from documents.repository import DocumentsRepository, InMemoryDocumentsRepository

from .infra import object_store
from .ingestion import partition_document_ids, run_ingestion
from .ingestion_status import record_ingestion_run_queued
from .ingestion_utils import make_fallback_external_id
from .rag.ingestion_contracts import (
    map_ingestion_error_to_status,
    resolve_ingestion_profile as _base_resolve_ingestion_profile,
)
from .schemas import (
    InfoIntakeRequest,
    RagIngestionRunRequest,
    RagQueryRequest,
    RagUploadMetadata,
)


logger = logging.getLogger(__name__)


CHECKPOINTER = FileCheckpointer()


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
    """
    import uuid as _uuid
    from collections.abc import Mapping as _Mapping

    if isinstance(value, _uuid.UUID):
        return str(value)
    if isinstance(value, _Mapping):
        return {
            (str(k) if isinstance(k, _uuid.UUID) else k): _make_json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(v) for v in value]
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
    repository_setting = getattr(settings, "DOCUMENTS_REPOSITORY", None)
    if isinstance(repository_setting, DocumentsRepository):
        return repository_setting
    if callable(repository_setting):
        candidate = repository_setting()
        if isinstance(candidate, DocumentsRepository):
            return candidate

    repository_class_setting = getattr(settings, "DOCUMENTS_REPOSITORY_CLASS", None)
    if repository_class_setting:
        try:
            repository_class = import_string(repository_class_setting)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("invalid_documents_repository_class") from exc
        candidate = repository_class()
        if not isinstance(candidate, DocumentsRepository):
            raise TypeError("documents_repository_invalid_instance")
        return candidate

    return InMemoryDocumentsRepository()


def _get_documents_repository() -> DocumentsRepository:
    try:
        views = import_module("ai_core.views")
        repo = getattr(views, "DOCUMENTS_REPOSITORY", None)
        if isinstance(repo, DocumentsRepository):
            return repo
        if callable(repo):
            candidate = repo()
            if isinstance(candidate, DocumentsRepository):
                return candidate
    except Exception:
        pass

    global _DOCUMENTS_REPOSITORY
    if _DOCUMENTS_REPOSITORY is None:
        _DOCUMENTS_REPOSITORY = _build_documents_repository()
    return _DOCUMENTS_REPOSITORY


def _coerce_cost_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_initial_cost(meta: Mapping[str, Any]) -> float | None:
    cost_block = meta.get("cost")
    if isinstance(cost_block, Mapping):
        for key in ("total_usd", "usd", "total"):
            cost_value = cost_block.get(key)
            coerced = _coerce_cost_value(cost_value)
            if coerced is not None:
                return coerced
    for key in ("cost_total_usd", "cost_usd", "total_cost_usd"):
        if key in meta:
            coerced = _coerce_cost_value(meta[key])
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


class _GraphCostTracker:
    def __init__(self, initial_total: float | None = None) -> None:
        self._total_usd = 0.0
        self._components: list[dict[str, Any]] = []
        self._reconciliation_ids: set[str] = set()
        if initial_total is not None:
            initial = _coerce_cost_value(initial_total)
            if initial and initial > 0:
                self.add_component(
                    source="meta",
                    usd=initial,
                    kind="initial",
                )

    @property
    def total_usd(self) -> float:
        return self._total_usd

    @property
    def components(self) -> list[dict[str, Any]]:
        return list(self._components)

    def add_component(self, *, source: str, usd: float, **extra: Any) -> None:
        amount = _coerce_cost_value(usd)
        if amount is None:
            return
        component: dict[str, Any] = {"source": source, "usd": float(amount)}
        for key, value in extra.items():
            if value is None:
                continue
            component[key] = (
                value if isinstance(value, (str, int, float, bool)) else str(value)
            )
        ledger_entry_id = component.get("ledger_entry_id")
        if ledger_entry_id:
            self._reconciliation_ids.add(str(ledger_entry_id))
        self._components.append(component)
        self._total_usd += float(amount)

    def record_ledger_meta(self, meta: Any) -> None:
        if not isinstance(meta, Mapping):
            return
        usage = meta.get("usage")
        usd = None
        if isinstance(usage, Mapping):
            cost_block = usage.get("cost")
            if isinstance(cost_block, Mapping):
                usd = cost_block.get("usd") or cost_block.get("total")
        if usd is None:
            usd = meta.get("cost_usd") or meta.get("usd")
        coerced = _coerce_cost_value(usd)
        if coerced is None:
            return
        source = str(meta.get("source") or meta.get("label") or "ledger")
        component: dict[str, Any] = {
            "label": meta.get("label"),
            "model": meta.get("model"),
            "trace_id": meta.get("trace_id"),
        }
        entry_id = meta.get("id") or meta.get("ledger_id")
        if entry_id:
            component["ledger_entry_id"] = str(entry_id)
        cache_hit = meta.get("cache_hit")
        if cache_hit is not None:
            component["cache_hit"] = bool(cache_hit)
        latency = meta.get("latency_ms")
        if latency is not None:
            component["latency_ms"] = latency
        self.add_component(source=source, usd=float(coerced), **component)

    def summary(self, ledger_id: str | None = None) -> dict[str, Any] | None:
        if not self._components:
            return None
        summary: dict[str, Any] = {
            "total_usd": self.total_usd,
            "components": self.components,
        }
        reconciliation: dict[str, Any] = {}
        if ledger_id:
            reconciliation["ledger_id"] = ledger_id
        if self._reconciliation_ids:
            reconciliation["entry_ids"] = sorted(self._reconciliation_ids)
        if reconciliation:
            summary["reconciliation"] = reconciliation
        return summary


@contextmanager
def _track_ledger_costs(initial_total: float | None = None):  # type: ignore[no-untyped-def]
    tracker = _GraphCostTracker(initial_total)
    yield tracker


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
    case_segment = object_store.sanitize_identifier(case_id)
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


def _map_upload_graph_skip(
    decision: str, transitions: Mapping[str, Mapping[str, Any]]
) -> Response:
    """Translate upload graph skip decisions into HTTP error responses."""

    diagnostics = transitions.get("accept_upload", {}).get("diagnostics", {})
    guardrail_diag = transitions.get("delta_and_guardrails", {}).get(
        "diagnostics", {}
    )

    if decision == "skip_guardrail":
        policy_events = guardrail_diag.get("policy_events") or ()
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
        mime = diagnostics.get("mime")
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
        max_bytes = diagnostics.get("max_bytes")
        if isinstance(max_bytes, (int, float)):
            detail = (
                "Uploaded file exceeds the allowed size of "
                f"{int(max_bytes)} bytes."
            )
        else:
            detail = "Uploaded file exceeds the allowed size."
        return _error_response(
            detail,
            "file_too_large",
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    if decision == "skip_invalid_input":
        return _error_response(
            "Upload payload is invalid.",
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

    ledger_identifier = _extract_ledger_identifier(normalized_meta)
    initial_cost_total = _extract_initial_cost(normalized_meta)
    base_observation_metadata: dict[str, Any] = {
        "trace_id": context.trace_id,
        "tenant.id": context.tenant_id,
        "case.id": context.case_id,
        "graph.version": context.graph_version,
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

            with _track_ledger_costs(initial_cost_total) as tracker:
                runner_meta["ledger_logger"] = tracker.record_ledger_meta
                try:
                    new_state, result = graph_runner.run(merged_state, runner_meta)
                finally:
                    runner_meta.pop("ledger_logger", None)

            cost_summary = tracker.summary(ledger_identifier)

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
            response = Response(result)
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

    collection_scope = getattr(validated_data, "collection_id", None)
    if collection_scope:
        meta["collection_id"] = collection_scope

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

    valid_document_ids, invalid_document_ids = _get_partition_document_ids()(
        meta["tenant_id"], meta["case_id"], validated_data.document_ids
    )

    to_dispatch = (
        valid_document_ids if valid_document_ids else validated_data.document_ids
    )
    if collection_scope:
        _persist_collection_scope(
            meta["tenant_id"], meta["case_id"], to_dispatch, collection_scope
        )
    _get_run_ingestion_task().delay(
        meta["tenant_id"],
        meta["case_id"],
        to_dispatch,
        resolved_profile_id,
        tenant_schema=meta["tenant_schema"],
        run_id=ingestion_run_id,
        trace_id=meta["trace_id"],
        idempotency_key=idempotency_key,
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

    idempotent = bool(idempotency_key)
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

    return Response(response_payload, status=status.HTTP_202_ACCEPTED)


def _derive_workflow_id(
    meta: Mapping[str, object], metadata: Mapping[str, object]
) -> str:
    candidate = metadata.get("workflow_id")
    if isinstance(candidate, str):
        candidate = candidate.strip()
        if candidate:
            return candidate.replace(":", "_")

    case_id = str(meta.get("case_id") or "").strip()
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
    meta: Mapping[str, object],
    metadata_obj: Mapping[str, object],
    external_id: str,
    *,
    media_type: str | None = None,
) -> DocumentMeta:
    workflow_id = _derive_workflow_id(meta, metadata_obj)
    payload: dict[str, object] = {
        "tenant_id": str(meta["tenant_id"]),
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

    header_collection = meta.get("collection_id")
    if header_collection and not metadata_obj.get("collection_id"):
        metadata_obj["collection_id"] = header_collection

    try:
        metadata_model = RagUploadMetadata.model_validate(metadata_obj)
    except ValidationError as exc:
        return _error_response(
            str(exc), "invalid_metadata", status.HTTP_400_BAD_REQUEST
        )

    metadata_obj = metadata_model.model_dump()
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

    document_meta = _build_document_meta(
        meta, metadata_obj, external_id, media_type=_infer_media_type(upload)
    )

    ref_payload: dict[str, object] = {
        "tenant_id": document_meta.tenant_id,
        "workflow_id": document_meta.workflow_id,
        "document_id": document_uuid,
        "collection_id": metadata_obj.get("collection_id"),
        "version": metadata_obj.get("version"),
    }
    document_ref = DocumentRef(**ref_payload)

    blob = InlineBlob(
        type="inline",
        media_type=_infer_media_type(upload),
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

    graph_payload = {
        "tenant_id": meta["tenant_id"],
        "uploader_id": str(meta.get("key_alias") or meta["case_id"]),
        "case_id": meta["case_id"],
        "request_id": meta["trace_id"],
        "workflow_id": document_meta.workflow_id,
        "file_bytes": file_bytes,
        "filename": original_name,
        "declared_mime": blob.media_type,
        "visibility": metadata_obj.get("visibility"),
        "tags": metadata_obj.get("tags"),
        "source_key": metadata_obj.get("external_id"),
        "origin_uri": metadata_obj.get("origin_uri"),
    }

    graph_context: dict[str, object] = {}

    def _persist_via_repository(_: Mapping[str, object]) -> dict[str, object]:
        try:
            repository = _get_documents_repository()
            repository.upsert(normalized_document)
        except Exception:
            logger.exception(
                "Failed to persist uploaded document via repository",
                extra={
                    "tenant_id": meta.get("tenant_id"),
                    "case_id": meta.get("case_id"),
                },
            )
            raise UploadIngestionError("document_persistence_failed")

        document_identifier = str(document_ref.document_id)
        graph_context["document_id"] = document_identifier
        if document_ref.version is not None:
            graph_context["version"] = document_ref.version
        return {"document_id": document_identifier, "version": document_ref.version}

    try:
        graph = UploadIngestionGraph(persistence_handler=_persist_via_repository)
        graph_result = graph.run(graph_payload, run_until="persist_complete")
    except UploadIngestionError as exc:
        reason = str(exc)
        if reason == "document_persistence_failed":
            return _error_response(
                "Failed to persist uploaded document.",
                "document_persistence_failed",
                status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        if reason.startswith("input_missing"):
            return _error_response(
                "Upload payload is invalid.",
                "invalid_upload_payload",
                status.HTTP_400_BAD_REQUEST,
            )
        logger.exception(
            "Upload ingestion graph failed",
            extra={
                "tenant_id": meta.get("tenant_id"),
                "case_id": meta.get("case_id"),
                "reason": reason,
            },
        )
        return _error_response(
            "Failed to process uploaded document.",
            "upload_graph_failed",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception:
        logger.exception(
            "Unexpected error while running upload ingestion graph",
            extra={
                "tenant_id": meta.get("tenant_id"),
                "case_id": meta.get("case_id"),
            },
        )
        return _error_response(
            "Failed to process uploaded document.",
            "upload_graph_failed",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    decision = str(graph_result.get("decision") or "")
    transitions = graph_result.get("transitions") or {}
    if decision.startswith("skip"):
        logger.info(
            "Upload ingestion graph skipped document",
            extra={
                "tenant_id": meta.get("tenant_id"),
                "case_id": meta.get("case_id"),
                "decision": decision,
                "reason": graph_result.get("reason"),
            },
        )
        return _map_upload_graph_skip(decision, transitions)

    document_id = graph_context.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        logger.error(
            "Upload ingestion graph completed without persisting document",
            extra={
                "tenant_id": meta.get("tenant_id"),
                "case_id": meta.get("case_id"),
                "graph_decision": decision,
            },
        )
        return _error_response(
            "Failed to persist uploaded document.",
            "document_persistence_failed",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    document_ids = [document_id]

    try:
        profile_binding = _resolve_ingestion_profile(
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

    try:
        _get_run_ingestion_task().delay(
            meta["tenant_id"],
            meta["case_id"],
            document_ids,
            resolved_profile_id,
            tenant_schema=meta["tenant_schema"],
            run_id=ingestion_run_id,
            trace_id=meta["trace_id"],
            idempotency_key=idempotency_key,
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

    idempotent = bool(idempotency_key)
    response_payload = {
        "status": "accepted",
        "document_id": document_id,
        "trace_id": meta["trace_id"],
        "idempotent": idempotent,
        "external_id": external_id,
        "ingestion_run_id": ingestion_run_id,
        "ingestion_status": "queued",
    }

    return Response(response_payload, status=status.HTTP_202_ACCEPTED)
