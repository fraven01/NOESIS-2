from __future__ import annotations

import uuid

from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Mapping, Optional

from django.utils import timezone
from pydantic import ValidationError

from ai_core.graphs.transition_contracts import (
    GraphTransition,
    StandardTransitionResult,
)
from ai_core.infra import object_store
from ai_core.infra.observability import record_span, tracing_enabled
from ai_core.infra.serialization import to_jsonable
from ai_core.rag import metrics
from ai_core.tool_contracts.base import tool_context_from_meta
from common.logging import get_logger
from common.task_result import TaskResult

logger = get_logger(__name__)


def _object_store_path_exists(path: str) -> bool:
    try:
        return (object_store.BASE_PATH / path).exists()
    except Exception:
        return False


def _task_context_payload(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(meta, MappingABC):
        return {}
    try:
        context = tool_context_from_meta(meta)
    except Exception:
        return {}
    payload: Dict[str, Any] = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "trace_id": context.scope.trace_id,
    }
    if context.scope.run_id:
        payload["run_id"] = context.scope.run_id
    if context.scope.ingestion_run_id:
        payload["ingestion_run_id"] = context.scope.ingestion_run_id
    return payload


def _task_context_snapshot(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(meta, MappingABC):
        return {}
    try:
        context = tool_context_from_meta(meta)
    except Exception:
        return {}
    return context.model_dump(mode="json", exclude_none=True)


def _task_result(
    *,
    status: str,
    data: Mapping[str, Any] | None = None,
    meta: Optional[Mapping[str, Any]] = None,
    task_name: str,
    error: str | None = None,
) -> Dict[str, Any]:
    payload = TaskResult(
        status=status,
        data=dict(data or {}),
        context_snapshot=_task_context_snapshot(meta),
        task_name=task_name,
        completed_at=timezone.now(),
        error=error,
    )
    return payload.model_dump(mode="json")


def _build_path(meta: Dict[str, str], *parts: str) -> str:
    """Build object store path with tenant and case identifiers.

    BREAKING CHANGE (Option A - Strict Separation):
    case_id is a business identifier, extracted from business_context.
    """
    context = tool_context_from_meta(meta)
    tenant = object_store.sanitize_identifier(context.scope.tenant_id)
    # BREAKING CHANGE: Extract case_id from business_context, not scope_context
    case = object_store.sanitize_identifier(context.business.case_id or "upload")
    return "/".join([tenant, case, *parts])


def _resolve_artifact_filename(meta: Mapping[str, Any], kind: str) -> str:
    """Derive a per-document filename for chunking artifacts."""

    def _normalise(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        candidate = str(value).strip()
        if not candidate:
            return None
        try:
            return object_store.sanitize_identifier(candidate)
        except Exception:
            return None

    hash_seed: Optional[str] = None
    for key in ("content_hash", "hash"):
        hash_seed = _normalise(meta.get(key))
        if hash_seed:
            break

    identifier_seed: Optional[str] = None
    for key in ("external_id", "document_id"):
        identifier_seed = _normalise(meta.get(key))
        if identifier_seed:
            break

    seeds = [component for component in (hash_seed, identifier_seed) if component]
    if not seeds:
        seeds = [_normalise(meta.get("id"))]
    seeds = [component for component in seeds if component]

    if not seeds:
        seeds = [uuid.uuid4().hex]

    seed = "-".join(seeds)
    base_name = f"{kind}-{seed}.json"
    try:
        return object_store.safe_filename(base_name)
    except Exception:
        return object_store.safe_filename(f"{kind}-{uuid.uuid4().hex}.json")


def _jsonify_for_task(value: Any) -> Any:
    """Convert objects returned by ingestion tasks into JSON primitives."""
    return to_jsonable(value)


def log_ingestion_run_start(
    *,
    tenant: str,
    case: str,
    run_id: str,
    doc_count: int,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
    case_status: Optional[str] = None,
    case_phase: Optional[str] = None,
    collection_scope: Optional[str] = None,
    document_collection_key: Optional[str] = None,
) -> None:
    extra = {
        "tenant_id": tenant,
        "case_id": case,
        "run_id": run_id,
        "doc_count": doc_count,
    }
    if trace_id:
        extra["trace_id"] = trace_id
    if idempotency_key:
        extra["idempotency_key"] = idempotency_key
    if embedding_profile:
        extra["embedding_profile"] = embedding_profile
    if vector_space_id:
        extra["vector_space_id"] = vector_space_id
    if case_status:
        extra["case_status"] = case_status
    if case_phase:
        extra["case_phase"] = case_phase
    if collection_scope:
        extra["collection_scope"] = collection_scope
    if document_collection_key:
        extra["document_collection_key"] = document_collection_key
    logger.info("ingestion.start", extra=extra)
    if trace_id:
        record_span(
            "rag.ingestion.run.start",
            attributes={**extra},
        )


def log_ingestion_run_end(
    *,
    tenant: str,
    case: str,
    run_id: str,
    doc_count: int,
    inserted: int,
    replaced: int,
    skipped: int,
    total_chunks: int,
    duration_ms: float,
    trace_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    embedding_profile: Optional[str] = None,
    vector_space_id: Optional[str] = None,
    case_status: Optional[str] = None,
    case_phase: Optional[str] = None,
    collection_scope: Optional[str] = None,
    document_collection_key: Optional[str] = None,
) -> None:
    extra = {
        "tenant_id": tenant,
        "case_id": case,
        "run_id": run_id,
        "doc_count": doc_count,
        "inserted": inserted,
        "replaced": replaced,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "duration_ms": duration_ms,
    }
    if trace_id:
        extra["trace_id"] = trace_id
    if idempotency_key:
        extra["idempotency_key"] = idempotency_key
    if embedding_profile:
        extra["embedding_profile"] = embedding_profile
    if vector_space_id:
        extra["vector_space_id"] = vector_space_id
    if case_status:
        extra["case_status"] = case_status
    if case_phase:
        extra["case_phase"] = case_phase
    if collection_scope:
        extra["collection_scope"] = collection_scope
    if document_collection_key:
        extra["document_collection_key"] = document_collection_key
    logger.info("ingestion.end", extra=extra)
    metrics.INGESTION_RUN_MS.observe(float(duration_ms))
    if trace_id:
        record_span(
            "rag.ingestion.run.end",
            attributes={**extra},
        )


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


def _coerce_str(value: Any | None) -> Optional[str]:
    """Return a stripped string representation when possible."""

    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        return candidate or None
    try:
        text = str(value)
    except Exception:
        return None
    return text.strip() or None


def _resolve_document_id(state: Mapping[str, Any], tool_context: Any) -> Optional[str]:
    """Try to resolve the document identifier from tool_context or state payloads.

    BREAKING CHANGE (Option A - Strict Separation):
    Document ID is a business identifier, so we check business_context first.
    """

    # Check business_context first (BREAKING CHANGE)
    candidate = getattr(tool_context.business, "document_id", None)
    if candidate:
        return _coerce_str(candidate)

    raw_document = state.get("raw_document") if isinstance(state, MappingABC) else None
    if isinstance(raw_document, MappingABC):
        for key in ("document_id", "external_id", "id"):
            candidate = _coerce_str(raw_document.get(key))
            if candidate:
                return candidate
        raw_metadata = raw_document.get("metadata")
        if isinstance(raw_metadata, MappingABC):
            for key in ("document_id", "external_id", "id"):
                candidate = _coerce_str(raw_metadata.get(key))
                if candidate:
                    return candidate

    return None


def _resolve_trace_context(
    state: Mapping[str, Any], meta: Optional[Mapping[str, Any]]
) -> Dict[str, Optional[str]]:
    """Collect identifiers required for tracing metadata.

    BREAKING CHANGE (Option A - Strict Separation):
    Business IDs (case_id, workflow_id) now extracted from business_context,
    not scope_context.
    """

    if not isinstance(meta, MappingABC):
        raise ValueError("meta with tool_context is required for trace context")
    tool_context = tool_context_from_meta(meta)
    scope_context = tool_context.scope
    business_context = tool_context.business

    # Infrastructure IDs from scope_context
    tenant_id = _coerce_str(scope_context.tenant_id)
    trace_id = _coerce_str(scope_context.trace_id)
    user_id = _coerce_str(scope_context.user_id)
    service_id = _coerce_str(scope_context.service_id)

    # Business IDs from business_context (BREAKING CHANGE)
    case_id = _coerce_str(business_context.case_id)
    workflow_id = _coerce_str(business_context.workflow_id)
    document_id = _resolve_document_id(state, tool_context)

    return {
        "tenant_id": tenant_id,
        "case_id": case_id,
        "workflow_id": workflow_id,
        "trace_id": trace_id,
        "document_id": document_id,
        "user_id": user_id,
        "service_id": service_id,
    }


def _build_base_span_metadata(
    trace_context: Mapping[str, Optional[str]],
    graph_run_id: Optional[str],
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    for key in (
        "tenant_id",
        "case_id",
        "trace_id",
        "document_id",
        "workflow_id",
        "run_id",
        "ingestion_run_id",
        "collection_id",
        "document_version_id",
    ):
        value = _coerce_str(trace_context.get(key))
        if value:
            attributes[f"meta.{key}"] = value
    if graph_run_id:
        attributes["meta.graph_run_id"] = graph_run_id
    return attributes


def _coerce_transition_result(
    transition: object,
) -> StandardTransitionResult | None:
    if isinstance(transition, GraphTransition):
        return transition.result
    if isinstance(transition, StandardTransitionResult):
        return transition
    if isinstance(transition, MappingABC):
        try:
            return StandardTransitionResult.model_validate(transition)
        except ValidationError:
            return None
    return None


def _collect_transition_attributes(
    transition: StandardTransitionResult, phase: str
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    pipeline_phase = None
    if transition.pipeline and transition.pipeline.phase:
        pipeline_phase = _coerce_str(transition.pipeline.phase)
    phase_value = pipeline_phase or _coerce_str(transition.phase)
    attributes["meta.phase"] = phase_value or phase

    severity_value = _coerce_str(transition.severity)
    if severity_value:
        attributes["meta.severity"] = severity_value

    context_payload = transition.context
    if isinstance(context_payload, MappingABC):
        document_id = _coerce_str(context_payload.get("document_id"))
        if document_id:
            attributes["meta.document_id"] = document_id
        run_id = _coerce_str(context_payload.get("run_id"))
        if run_id:
            attributes["meta.run_id"] = run_id
        ingestion_run_id = _coerce_str(context_payload.get("ingestion_run_id"))
        if ingestion_run_id:
            attributes["meta.ingestion_run_id"] = ingestion_run_id
        collection_id = _coerce_str(context_payload.get("collection_id"))
        if collection_id:
            attributes["meta.collection_id"] = collection_id
        document_version_id = _coerce_str(context_payload.get("document_version_id"))
        if document_version_id:
            attributes["meta.document_version_id"] = document_version_id

    if phase == "document_pipeline":
        pipeline_payload = transition.pipeline
        if pipeline_payload is not None:
            run_until = getattr(pipeline_payload, "run_until", None)
            run_until_label = None
            if run_until is not None:
                if hasattr(run_until, "value"):
                    run_until_label = _coerce_str(getattr(run_until, "value", None))
                if not run_until_label:
                    run_until_label = _coerce_str(run_until)
            if run_until_label:
                attributes["meta.run_until"] = run_until_label
            if pipeline_payload.phase:
                phase_label = _coerce_str(pipeline_payload.phase)
                if phase_label:
                    attributes["meta.phase"] = phase_label
                    attributes["meta.pipeline_phase"] = phase_label
    elif phase == "update_status":
        lifecycle_payload = transition.lifecycle
        if lifecycle_payload is not None:
            status = _coerce_str(lifecycle_payload.status)
            if status:
                attributes["meta.status"] = status
    elif phase == "ingest":
        embedding_section = transition.embedding
        if embedding_section is not None:
            embedding_result = getattr(embedding_section, "result", None)
            outcome = _coerce_str(getattr(embedding_result, "status", None))
            if outcome:
                attributes["meta.outcome"] = outcome
        delta_section = transition.delta
        if delta_section is not None:
            delta_decision = _coerce_str(delta_section.decision)
            if delta_decision:
                attributes.setdefault("meta.delta", delta_decision)
    elif phase == "guardrails":
        guardrail_payload = transition.guardrail
        if guardrail_payload is not None:
            allowed = getattr(guardrail_payload, "allowed", None)
            if isinstance(allowed, bool):
                attributes["meta.allowed"] = allowed
    elif phase == "ingest_decision":
        delta_section = transition.delta
        if delta_section is not None:
            delta_decision = _coerce_str(delta_section.decision)
            if delta_decision:
                attributes["meta.delta"] = delta_decision

    decision_value = _coerce_str(transition.decision)
    if phase in {"guardrails", "ingest_decision"} and decision_value:
        attributes["meta.decision"] = decision_value
    return attributes


def _ensure_ingestion_phase_spans(
    state: Mapping[str, Any],
    result: Mapping[str, Any],
    trace_context: Mapping[str, Optional[str]],
) -> None:
    """Record fallback spans for ingestion phases when decorators are bypassed."""

    if not tracing_enabled():
        return

    transitions = result.get("transitions")
    if not isinstance(transitions, MappingABC):
        return

    recorded_phases: set[str] = set()
    span_tracker = state.get("_span_phases")
    if isinstance(span_tracker, (set, list, tuple)):
        recorded_phases.update(str(phase) for phase in span_tracker)

    graph_run_id = _coerce_str(result.get("graph_run_id")) or _coerce_str(
        state.get("graph_run_id")
    )
    base_metadata = _build_base_span_metadata(trace_context, graph_run_id)

    phase_mapping = {
        "update_status": "update_status_normalized",
        "guardrails": "enforce_guardrails",
        "document_pipeline": "document_pipeline",
        "ingest_decision": "ingest_decision",
        "ingest": "ingest",
    }

    for phase, transition_key in phase_mapping.items():
        if phase in recorded_phases:
            continue

        transition = _coerce_transition_result(transitions.get(transition_key))
        if transition is None:
            continue

        attributes = dict(base_metadata)
        attributes.update(_collect_transition_attributes(transition, phase))

        if attributes:
            record_span(f"crawler.ingestion.{phase}", attributes=attributes)
