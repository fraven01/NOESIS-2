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

import json
import logging
from collections.abc import Iterable, Mapping
from uuid import uuid4
from importlib import import_module

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone
from pydantic import ValidationError
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response

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
    NeedsMappingRequest,
    RagIngestionRunRequest,
    RagQueryRequest,
    RagUploadMetadata,
    ScopeCheckRequest,
    SystemDescriptionRequest,
)


logger = logging.getLogger(__name__)


CHECKPOINTER = FileCheckpointer()


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


GRAPH_REQUEST_MODELS = {
    "info_intake": InfoIntakeRequest,
    "scope_check": ScopeCheckRequest,
    "needs_mapping": NeedsMappingRequest,
    "system_description": SystemDescriptionRequest,
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

    try:
        state = _get_checkpointer().load(context)
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
        new_state, result = graph_runner.run(merged_state, runner_meta)
    except InputError as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)
    except ToolContextError as exc:
        # Invalid execution context (e.g., router/tenant/case issues) should be
        # surfaced to clients as a 400 rather than a transient 503.
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)
    except ValueError as exc:
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)
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
        return _error_response(str(exc), "invalid_request", status.HTTP_400_BAD_REQUEST)

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

    return response


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
    if metadata_obj.get("external_id") is None:
        metadata_obj.pop("external_id")
    if metadata_obj.get("collection_id") is None:
        metadata_obj.pop("collection_id", None)

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

    object_store.write_json(f"{storage_prefix}/{document_id}.meta.json", metadata_obj)

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
    document_ids = [document_id]

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
