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

from ai_core.infra import ledger
from ai_core.infra.observability import (
    emit_event,
    end_trace as lf_end_trace,
    start_trace as lf_start_trace,
    tracing_enabled as lf_tracing_enabled,
    update_observation,
)
from ai_core.case_events import emit_ingestion_case_event
from common.celery import with_scope_apply_async

from .document_upload import handle_document_upload
from .graph_executor import execute_graph
from .graph_support import (
    ASYNC_GRAPH_NAMES,
    CHECKPOINTER,
    GRAPH_REQUEST_MODELS,
    _apply_collection_header_bridge,
    _dump_jsonable,
    _error_response,
    _extract_initial_cost,
    _extract_ledger_identifier,
    _format_validation_error,
    _get_checkpointer,
    _log_graph_response_payload,
    _normalize_meta,
    _should_enqueue_graph,
)
from .ingestion import (
    RUN_INGESTION,
    _enqueue_ingestion_task,
    _ensure_document_collection,
    _get_partition_document_ids,
    _get_run_ingestion_task,
    _persist_collection_scope,
    _resolve_ingestion_profile,
    start_ingestion_run,
)
from .repository import _build_documents_repository, _get_documents_repository
from .upload_support import (
    _build_document_meta,
    _coerce_transition_result,
    _derive_workflow_id,
    _ensure_collection_with_warning,
    _ensure_document_collection_record,
    _infer_media_type,
    _map_upload_graph_skip,
)

__all__ = [
    "ASYNC_GRAPH_NAMES",
    "CHECKPOINTER",
    "GRAPH_REQUEST_MODELS",
    "RUN_INGESTION",
    "_apply_collection_header_bridge",
    "_build_document_meta",
    "_build_documents_repository",
    "_coerce_transition_result",
    "_derive_workflow_id",
    "_dump_jsonable",
    "_enqueue_ingestion_task",
    "_ensure_collection_with_warning",
    "_ensure_document_collection",
    "_ensure_document_collection_record",
    "_error_response",
    "_extract_initial_cost",
    "_extract_ledger_identifier",
    "_format_validation_error",
    "_get_checkpointer",
    "_get_documents_repository",
    "_get_partition_document_ids",
    "_get_run_ingestion_task",
    "_infer_media_type",
    "_log_graph_response_payload",
    "_map_upload_graph_skip",
    "_normalize_meta",
    "_persist_collection_scope",
    "_resolve_ingestion_profile",
    "_should_enqueue_graph",
    "execute_graph",
    "emit_event",
    "emit_ingestion_case_event",
    "lf_end_trace",
    "lf_start_trace",
    "lf_tracing_enabled",
    "handle_document_upload",
    "ledger",
    "start_ingestion_run",
    "update_observation",
    "with_scope_apply_async",
]
