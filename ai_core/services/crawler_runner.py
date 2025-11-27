"""Coordinator for orchestrating crawler ingestion LangGraph runs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass, dataclass
from collections.abc import Mapping
from typing import Any, Callable
from uuid import UUID, uuid4

from django.conf import settings
from rest_framework import status

from common.logging import get_logger
from crawler.http_fetcher import HttpFetcher
from llm_worker.runner import submit_worker_task
from llm_worker.tasks import run_graph as run_graph_task

from ai_core.contracts.crawler_runner import (
    CrawlerRunContext,
    CrawlerStateBundle,
)
from ai_core.graph import registry as graph_registry
from ai_core.infra import object_store
from ai_core.middleware import guardrails as guardrails_middleware
from ai_core.rag.guardrails import GuardrailLimits
from ai_core.schemas import CrawlerRunRequest
from ai_core.rag.vector_client import get_default_client
from customers.tenant_context import TenantContext
from documents.domain_service import DocumentDomainService
from documents.models import DocumentCollection

import ai_core.services as services_module
from . import _get_documents_repository, _make_json_safe
from .crawler_state_builder import build_crawler_state

logger = get_logger(__name__)


@dataclass(slots=True)
class CrawlerRunnerCoordinatorResult:
    """Return value for crawler ingestion coordination."""

    payload: dict[str, Any]
    status_code: int
    idempotency_key: str | None


def run_crawler_runner(
    *,
    meta: dict[str, Any],
    request_model: CrawlerRunRequest,
    lifecycle_store: object | None,
    graph_factory: Callable[[], object] | None = None,
) -> CrawlerRunnerCoordinatorResult:
    """Execute the crawler ingestion LangGraph for the provided request."""

    if request_model.collection_id:
        meta["collection_id"] = request_model.collection_id

    workflow_default = getattr(settings, "CRAWLER_DEFAULT_WORKFLOW_ID", None)
    workflow_resolved = (
        request_model.workflow_id or workflow_default or meta.get("tenant_id")
    )
    if not workflow_resolved:
        raise ValueError("workflow_id could not be resolved for the crawler run")

    try:
        repository = _get_documents_repository()
    except Exception:
        repository = None

    context = CrawlerRunContext(
        meta=meta,
        request=request_model,
        workflow_id=str(workflow_resolved),
        repository=repository,
    )
    guardrail_defaults = GuardrailLimits(
        max_document_bytes=getattr(settings, "CRAWLER_MAX_DOCUMENT_BYTES", None)
    )

    def fetcher_factory(config):
        return HttpFetcher(config)

    state_builds = build_crawler_state(
        context,
        fetcher_factory=fetcher_factory,
        lifecycle_store=lifecycle_store,
        object_store=object_store,
        guardrail_limits=guardrail_defaults,
    )
    if not state_builds:
        raise ValueError("No origins resolved for crawler run.")

    tenant = _resolve_tenant(meta.get("tenant_id"))
    if tenant:
        _register_documents_for_builds(
            tenant=tenant,
            builds=state_builds,
            embedding_profile=request_model.embedding_profile,
            scope=request_model.scope,
        )

    workflow_id = state_builds[0].state.get("workflow_id") or context.workflow_id
    fingerprint_match = _check_idempotency_fingerprint(
        meta, workflow_id, request_model, state_builds
    )
    header_idempotent = bool(meta.get("idempotency_key"))

    task_ids: list[dict[str, object]] = []
    completed_runs: list[dict[str, object]] = []
    pending_async = False
    graph_name = "crawler.ingestion"
    wait_timeout = 0.1
    inline_execution = request_model.mode == "manual"
    inline_graph = None
    if inline_execution and graph_factory is not None:
        try:
            inline_graph = graph_factory()
        except Exception:
            logger.exception(
                "crawler_runner.graph_build_failed",
                extra={"graph_name": graph_name},
            )
            inline_graph = None
    for build in state_builds:
        task_payload = {"state": build.state, "graph_name": graph_name}
        scope = {
            "tenant_id": meta["tenant_id"],
            "case_id": meta["case_id"],
            "trace_id": meta.get("trace_id"),
            "workflow_id": build.state.get("workflow_id"),
        }
        if inline_execution:
            _prime_manual_state(build, graph_name, inline_graph)
            result, completed = _run_graph_inline(
                build=build,
                meta=meta,
                graph_name=graph_name,
                scope=scope,
                graph_runner=inline_graph,
            )
        else:
            result, completed = submit_worker_task(
                task_payload=task_payload,
                scope=scope,
                graph_name=graph_name,
                ledger_identifier=None,
                initial_cost_total=None,
                timeout_s=wait_timeout,
            )
        task_id = result.get("task_id")
        if task_id:
            task_ids.append(
                {
                    "task_id": task_id,
                    "origin": build.origin,
                    "document_id": build.document_id,
                }
            )
        if completed:
            completed_runs.append(
                {
                    "build": build,
                    "state": result.get("state") or {},
                    "result": result.get("result") or {},
                }
            )
        else:
            pending_async = True

    idempotency_key = meta.get("idempotency_key")
    idempotent_flag = bool(fingerprint_match or header_idempotent)

    if completed_runs and not pending_async:
        guardrail_error = _detect_guardrail_error(completed_runs)
        if guardrail_error is not None:
            return CrawlerRunnerCoordinatorResult(
                payload=guardrail_error,
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                idempotency_key=idempotency_key,
            )
        response_payload = _build_synchronous_payload(
            request_model,
            workflow_id,
            completed_runs,
            meta,
            idempotent_flag,
        )
        return CrawlerRunnerCoordinatorResult(
            payload=response_payload,
            status_code=status.HTTP_200_OK,
            idempotency_key=idempotency_key,
        )

    response_payload = {
        "status": "accepted",
        "workflow_id": workflow_id,
        "mode": request_model.mode,
        "collection_id": request_model.collection_id,
        "task_ids": task_ids,
        "idempotent": idempotent_flag,
        "message": f"Crawler-Ingestion für {len(task_ids)} URL(s) gestartet (asynchron)",
    }
    return CrawlerRunnerCoordinatorResult(
        payload=response_payload,
        status_code=status.HTTP_202_ACCEPTED,
        idempotency_key=idempotency_key,
    )


def _check_idempotency_fingerprint(
    meta: Mapping[str, Any],
    workflow_id: str,
    request_model: CrawlerRunRequest,
    builds: list[CrawlerStateBundle],
) -> bool:
    origin_keys = sorted(build.origin for build in builds)
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

    try:
        tenant_key = object_store.sanitize_identifier(str(meta.get("tenant_id")))
        case_key = object_store.sanitize_identifier(str(meta.get("case_id")))
        fingerprint_path = f"{tenant_key}/{case_key}/crawler_runner_idempotency.json"
    except Exception:
        fingerprint_path = None

    if not fingerprint_path:
        return False

    try:
        existing = object_store.read_json(fingerprint_path)
    except FileNotFoundError:
        existing = None
    except Exception:
        existing = None

    if isinstance(existing, dict) and existing.get("fingerprint") == fingerprint:
        return True

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

    return False


def _register_documents_for_builds(
    *,
    tenant,
    builds: list[CrawlerStateBundle],
    embedding_profile: str | None,
    scope: str | None,
) -> None:
    service = DocumentDomainService(vector_store=get_default_client())

    for build in builds:
        normalized = build.state.get("normalized_document_input")
        if not isinstance(normalized, Mapping):
            continue

        metadata = normalized.get("meta") if isinstance(normalized, Mapping) else None
        metadata_dict = dict(metadata or {})

        blob_payload = normalized.get("blob") if isinstance(normalized, Mapping) else None
        checksum = None
        if isinstance(blob_payload, Mapping):
            checksum = blob_payload.get("sha256")
        if checksum is None:
            checksum = normalized.get("checksum")
        if not checksum:
            logger.warning(
                "crawler_runner.document_registration_missing_checksum",
                extra={"origin": build.origin},
            )
            continue

        source = metadata_dict.get("origin_uri") or build.origin
        collection_identifier = build.collection_id
        ref_payload = normalized.get("ref") if isinstance(normalized, Mapping) else None
        if collection_identifier is None and isinstance(ref_payload, Mapping):
            collection_identifier = ref_payload.get("collection_id")

        collection_instance = None
        if collection_identifier is not None:
            collection_instance = _ensure_collection_with_warning(
                service,
                tenant,
                collection_identifier,
                embedding_profile=embedding_profile,
                scope=scope,
            )

        ingest_result = service.ingest_document(
            tenant=tenant,
            source=str(source),
            content_hash=str(checksum),
            metadata=metadata_dict,
            collections=(
                (collection_instance,) if collection_instance is not None else ()
            ),
            embedding_profile=embedding_profile,
            scope=scope,
            dispatcher=lambda *_: None,
        )

        document_id = str(ingest_result.document.id)
        build.document_id = document_id
        build.state["document_id"] = document_id

        if isinstance(normalized, Mapping):
            normalized_mutable = dict(normalized)
            ref = normalized_mutable.get("ref")
            if isinstance(ref, Mapping):
                ref = dict(ref)
            else:
                ref = {}
            ref["document_id"] = document_id
            if collection_instance is not None:
                ref["collection_id"] = str(collection_instance.collection_id)
            normalized_mutable["ref"] = ref
            normalized_mutable["checksum"] = checksum
            build.state["normalized_document_input"] = normalized_mutable


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
                "crawler_runner.collection_missing_created",
                extra={
                    "tenant_id": str(tenant.id),
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


def _resolve_tenant(identifier: object):
    if identifier is None:
        return None
    try:
        resolved = TenantContext.resolve_identifier(identifier, allow_pk=True)
    except Exception:
        logger.exception(
            "crawler_runner.resolve_tenant_failed", extra={"tenant_id": identifier}
        )
        return None
    return resolved


def _run_graph_inline(
    *,
    build: CrawlerStateBundle,
    meta: Mapping[str, Any],
    graph_name: str,
    scope: Mapping[str, object],
    graph_runner: object | None,
) -> tuple[dict[str, Any], bool]:
    meta_payload = {"graph_name": graph_name}
    if graph_runner is not None and hasattr(graph_runner, "run"):
        new_state, result_payload = graph_runner.run(build.state, meta_payload)
        inline_result = {
            "state": new_state,
            "result": result_payload,
            "cost_summary": None,
        }
    else:
        inline_result = run_graph_task.run(
            graph_name=graph_name,
            state=build.state,
            meta=meta_payload,
            ledger_identifier=None,
            initial_cost_total=None,
            tenant_id=scope.get("tenant_id"),
            case_id=scope.get("case_id"),
            trace_id=scope.get("trace_id"),
        )
    response_payload = dict(inline_result)
    response_payload["task_id"] = f"inline-{uuid4().hex}"
    return response_payload, True


def _prime_manual_state(
    build: CrawlerStateBundle,
    graph_name: str,
    inline_graph: object | None,
) -> None:
    runner = inline_graph
    if runner is None:
        try:
            runner = graph_registry.get(graph_name)
        except KeyError:
            return
    if not hasattr(runner, "start_crawl"):
        return
    try:
        prepared_state = runner.start_crawl(build.state)
    except Exception:
        logger.exception(
            "crawler_runner.start_crawl_failed",
            extra={"graph_name": graph_name, "origin": build.origin},
        )
        return
    if isinstance(prepared_state, Mapping):
        build.state = dict(prepared_state)


def _detect_guardrail_error(
    completed_runs: list[dict[str, object]],
) -> dict[str, object] | None:
    for entry in completed_runs:
        state_data = entry.get("state")
        if not isinstance(state_data, Mapping):
            continue
        artifacts = state_data.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        guardrail_decision = _coerce_guardrail_decision(
            artifacts.get("guardrail_decision")
        )
        if guardrail_decision and not guardrail_decision.allowed:
            return _build_guardrail_denied_payload(entry["build"], guardrail_decision)
    return None


def _build_synchronous_payload(
    request_model: CrawlerRunRequest,
    workflow_id: str,
    completed_runs: list[dict[str, object]],
    meta: Mapping[str, Any],
    idempotent_flag: bool,
) -> dict[str, object]:
    origins_payload: list[dict[str, object]] = []
    transitions_payload: list[dict[str, object]] = []
    telemetry_payload: list[dict[str, object]] = []
    errors_payload: list[dict[str, object]] = []

    for entry in completed_runs:
        build = entry["build"]
        state_data = entry.get("state")
        if not isinstance(state_data, Mapping):
            state_data = {}
        result_payload = entry.get("result")
        if not isinstance(result_payload, Mapping):
            result_payload = {}
        telemetry_payload.append(_build_fetch_telemetry_entry(build))
        transitions_payload.append(
            {
                "origin": build.origin,
                "transitions": _make_json_safe(
                    state_data.get("transitions")
                    or result_payload.get("transitions")
                    or {}
                ),
            }
        )
        errors_payload.extend(_extract_origin_errors(build, state_data))
        ingestion_run_id = _maybe_start_ingestion(build, state_data, meta)
        origins_payload.append(
            _summarize_origin_entry(build, state_data, result_payload, ingestion_run_id)
        )

    return {
        "workflow_id": workflow_id,
        "mode": request_model.mode,
        "collection_id": request_model.collection_id,
        "origins": origins_payload,
        "transitions": transitions_payload,
        "telemetry": telemetry_payload,
        "errors": errors_payload,
        "idempotent": idempotent_flag,
    }


def _maybe_start_ingestion(
    build: CrawlerStateBundle,
    state: Mapping[str, object],
    meta: Mapping[str, Any],
) -> str | None:
    action = state.get("ingest_action")
    if not action:
        return None

    document_id = state.get("document_id") or build.document_id
    payload: dict[str, object] = {"document_ids": [document_id]}
    if build.collection_id:
        payload["collection_id"] = build.collection_id

    try:
        response = services_module.start_ingestion_run(
            payload, dict(meta), meta.get("idempotency_key")
        )
    except Exception:
        logger.exception(
            "crawler_runner.start_ingestion_failed",
            extra={"document_id": document_id, "origin": build.origin},
        )
        return None

    data = getattr(response, "data", None)
    if isinstance(data, Mapping):
        ingestion_run_id = data.get("ingestion_run_id")
        if isinstance(ingestion_run_id, str):
            return ingestion_run_id
    return None


def _build_fetch_telemetry_entry(build: CrawlerStateBundle) -> dict[str, object]:
    return {
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


def _extract_origin_errors(
    build: CrawlerStateBundle, state: Mapping[str, object]
) -> list[dict[str, object]]:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return []
    errors = artifacts.get("errors")
    if not isinstance(errors, list):
        return []
    serialised = []
    for error in errors:
        if not isinstance(error, Mapping):
            continue
        serialised.append({"origin": build.origin, **_make_json_safe(error)})
    return serialised


def _summarize_origin_entry(
    build: CrawlerStateBundle,
    state: Mapping[str, object],
    result_payload: Mapping[str, object],
    ingestion_run_id: str | None,
) -> dict[str, object]:
    control = state.get("control")
    if not isinstance(control, Mapping):
        control = {}
    summary_state = {
        "workflow_id": state.get("workflow_id"),
        "document_id": state.get("document_id") or build.document_id,
        "origin_uri": state.get("origin_uri") or build.origin,
        "provider": state.get("provider") or build.provider,
        "content_hash": state.get("content_hash"),
        "tags": control.get("tags"),
        "snapshot_requested": build.snapshot_requested,
        "snapshot_label": build.snapshot_label,
    }
    entry: dict[str, object] = {
        "origin": build.origin,
        "provider": build.provider,
        "document_id": build.document_id,
        "result": _make_json_safe(result_payload),
        "control": _make_json_safe(control),
        "ingest_action": state.get("ingest_action"),
        "gating_score": state.get("gating_score"),
        "graph_run_id": state.get("graph_run_id") or result_payload.get("graph_run_id"),
        "state": _make_json_safe(summary_state),
        "collection_id": build.collection_id,
        "review": build.review,
        "dry_run": build.dry_run,
    }
    if ingestion_run_id:
        entry["ingestion_run_id"] = ingestion_run_id
    return entry


def _serialise_guardrail_component(value: object) -> object:
    candidate = value
    if hasattr(candidate, "model_dump"):
        try:
            candidate = candidate.model_dump()
        except Exception:
            pass
    if is_dataclass(candidate):
        try:
            candidate = asdict(candidate)
        except Exception:
            candidate = dict(getattr(candidate, "__dict__", {}))
    elif hasattr(candidate, "__dict__") and not isinstance(
        candidate, (str, bytes, bytearray)
    ):
        candidate = dict(getattr(candidate, "__dict__", {}))

    if isinstance(candidate, Mapping):
        processed = {
            str(key): _serialise_guardrail_component(value)
            for key, value in candidate.items()
        }
        return _make_json_safe(processed)
    if isinstance(candidate, (list, tuple, set)):
        processed_list = [_serialise_guardrail_component(item) for item in candidate]
        return _make_json_safe(processed_list)
    return _make_json_safe(candidate)


def _serialise_guardrail_attributes(
    attributes: Mapping[str, object] | None,
) -> dict[str, object]:
    if not attributes:
        return {}
    return {
        str(key): _serialise_guardrail_component(value)
        for key, value in dict(attributes).items()
    }


def _coerce_guardrail_decision(
    candidate: object,
) -> guardrails_middleware.GuardrailDecision | None:
    if isinstance(candidate, guardrails_middleware.GuardrailDecision):
        return candidate
    if isinstance(candidate, Mapping):
        decision_value = candidate.get("decision")
        reason_value = candidate.get("reason")
        attributes = candidate.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {
                key: value
                for key, value in candidate.items()
                if key not in {"decision", "reason"}
            }
        try:
            return guardrails_middleware.GuardrailDecision(
                str(decision_value or ""),
                str(reason_value or ""),
                attributes=dict(attributes),
            )
        except Exception:
            return None
    return None


def _build_guardrail_denied_payload(
    build: CrawlerStateBundle,
    decision: guardrails_middleware.GuardrailDecision,
) -> dict[str, object]:
    limits = {}
    guardrail_state = build.state.get("guardrails")
    if isinstance(guardrail_state, Mapping):
        limits = guardrail_state.get("limits") or {}
    return {
        "code": "crawler_guardrail_denied",
        "reason": decision.reason,
        "origin": build.origin,
        "policy_events": list(decision.policy_events),
        "attributes": _serialise_guardrail_attributes(decision.attributes),
        "limits": limits,
    }
