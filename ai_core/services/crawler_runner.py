"""Coordinator for orchestrating crawler ingestion LangGraph runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, dataclass
from collections.abc import Mapping
from typing import Any, Callable
from uuid import uuid4

from django.conf import settings
from rest_framework import status

from common.constants import DEFAULT_WORKFLOW_PLACEHOLDER
from common.logging import get_logger
from crawler.http_fetcher import HttpFetcher


from ai_core.contracts.crawler_runner import (
    CrawlerRunContext,
    CrawlerStateBundle,
)

from ai_core.infra import object_store
from ai_core.middleware import guardrails as guardrails_middleware
from ai_core.rag.guardrails import GuardrailLimits
from ai_core.schemas import CrawlerRunRequest
from ai_core.tool_contracts.base import tool_context_from_meta

# NOTE: build_universal_ingestion_graph imported lazily inside run_crawler_runner()
# to prevent OOM in test environments that mock heavy modules.
from customers.tenant_context import TenantContext
from documents.contract_utils import resolve_workflow_id
from documents.domain_service import (
    DocumentIngestSpec,
    BulkIngestRecord,
)

import ai_core.services as services_module
from . import _get_documents_repository, _dump_jsonable
from .crawler_state_builder import build_crawler_state
from documents.normalization import normalize_url

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

    tool_context = tool_context_from_meta(meta)
    scope_meta = tool_context.scope
    # BREAKING CHANGE (Option A - Strict Separation):
    # Business IDs (case_id, workflow_id) now in business_context
    business_meta = tool_context.business

    # Lazy import to prevent OOM in test environments
    from ai_core.graphs.technical.universal_ingestion_graph import (
        build_universal_ingestion_graph,
        UniversalIngestionInput,
    )

    # BREAKING CHANGE (Option A): collection_id goes to business_context, not scope_context
    if request_model.collection_id and (
        request_model.collection_id != business_meta.collection_id
    ):
        updated_business = business_meta.model_copy(
            update={"collection_id": request_model.collection_id}
        )
        tool_context = tool_context.model_copy(update={"business": updated_business})
        meta["business_context"] = updated_business.model_dump(
            mode="json", exclude_none=True
        )
        meta["tool_context"] = tool_context.model_dump(mode="json", exclude_none=True)
        business_meta = updated_business

    workflow_default = getattr(settings, "CRAWLER_DEFAULT_WORKFLOW_ID", None)
    workflow_resolved = resolve_workflow_id(
        request_model.workflow_id or workflow_default,
        required=False,
        placeholder=DEFAULT_WORKFLOW_PLACEHOLDER,
    )

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

    # Idempotency: compute a lightweight fingerprint so repeat calls can be flagged.
    # NOTE: This must run BEFORE ID validation to allow early return for idempotent requests
    idempotency_key = scope_meta.idempotency_key
    idempotent_flag = False
    fingerprint = None

    # Pre-validate required IDs for fingerprinting (full validation happens later)
    tenant_id_for_fp = scope_meta.tenant_id
    # BREAKING CHANGE (Option A): case_id from business_context
    case_id_for_fp = business_meta.case_id  # Optional - may be None

    if not tenant_id_for_fp:
        raise ValueError("tenant_id is required for idempotency fingerprinting")

    try:
        import json
        import hashlib

        fingerprint_payload = {
            "tenant_id": str(tenant_id_for_fp),  # Required (Pre-MVP ID Contract)
            "case_id": (
                str(case_id_for_fp) if case_id_for_fp else None
            ),  # Optional - include in fingerprint if present
            "workflow_id": str(workflow_resolved),
            "collection_id": request_model.collection_id,
            "mode": request_model.mode,
            "origins": sorted(  # Sort for stable fingerprint
                [
                    origin.model_dump(mode="json")
                    for origin in request_model.origins or []
                ],
                key=lambda o: normalize_url(o.get("uri")) or o.get("uri", ""),
            ),
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
    except Exception:
        fingerprint = None

    # Check idempotency using Redis/Django cache (cross-process)
    from django.core.cache import cache

    CACHE_PREFIX = getattr(
        settings, "CRAWLER_IDEMPOTENCY_CACHE_PREFIX", "crawler_idempotency:"
    )
    CACHE_TTL = getattr(settings, "CRAWLER_IDEMPOTENCY_CACHE_TTL_SECONDS", 3600)

    if idempotency_key:
        cache_key = f"{CACHE_PREFIX}key:{idempotency_key}"
        if cache.get(cache_key):
            idempotent_flag = True
        else:
            cache.set(cache_key, True, timeout=CACHE_TTL)
    elif fingerprint:
        cache_key = f"{CACHE_PREFIX}fp:{fingerprint}"
        if cache.get(cache_key):
            idempotent_flag = True
        else:
            cache.set(cache_key, True, timeout=CACHE_TTL)

    # Early return for idempotent requests
    if idempotent_flag:
        logger.info(
            "crawler_request_idempotent_skipped",
            extra={
                "fingerprint": fingerprint,
                "idempotency_key": idempotency_key,
                "tenant_id": str(tenant_id_for_fp),
                "cache_key": cache_key,
            },
        )
        return CrawlerRunnerCoordinatorResult(
            payload={
                "idempotent": True,
                "skipped": True,
                "origins": [],
                "message": "Request already processed (idempotent)",
            },
            status_code=status.HTTP_200_OK,
            idempotency_key=idempotency_key,
        )

    # Validate mandatory IDs before graph invocation (Pre-MVP ID Contract)
    # NOTE: case_id is optional at HTTP level, becomes mandatory for graph execution
    required_ids = {
        "tenant_id": "tenant_id is mandatory for crawler ingestion",
        "trace_id": "trace_id is mandatory for correlation",
        "invocation_id": "invocation_id is mandatory per ID contract",
    }

    for field, error_msg in required_ids.items():
        if not getattr(scope_meta, field, None):
            raise ValueError(error_msg)

    # Extract validated IDs
    tenant_id = scope_meta.tenant_id
    # BREAKING CHANGE (Option A): case_id from business_context
    case_id = business_meta.case_id  # Optional at HTTP level
    trace_id = scope_meta.trace_id

    # Identity IDs (Pre-MVP ID Contract)
    # HTTP requests have user_id (if authenticated), service_id is None
    # S2S hops (Celery tasks) have service_id, user_id may be present for audit trail
    # Crawler runner accepts both patterns since it can be called from HTTP or Celery
    _service_id = scope_meta.service_id
    _user_id = scope_meta.user_id

    completed_runs: list[dict[str, object]] = []
    # Updated to use UniversalIngestionGraph
    graph_app = build_universal_ingestion_graph()

    for build in state_builds:
        # Construct Input for Universal Graph
        normalized = build.state.get("normalized_document_input")
        input_payload: UniversalIngestionInput = {
            "source": "crawler",
            "mode": "ingest_only",
            "collection_id": request_model.collection_id,
            "upload_blob": None,
            "metadata_obj": None,
            "normalized_document": normalized,
        }

        # Generate canonical ingestion_run_id for this build
        canonical_ingestion_run_id = str(uuid4())

        # Determine strict context
        # Note: Universal Graph expects tenant_id, trace_id, case_id in context
        run_context = {
            "tenant_id": str(tenant_id),
            "case_id": str(case_id),
            "trace_id": str(trace_id),
            "workflow_id": str(workflow_resolved),
            "ingestion_run_id": canonical_ingestion_run_id,
            "dry_run": request_model.dry_run,
            "idempotency_key": idempotency_key,
        }

        try:
            result = graph_app.invoke({"input": input_payload, "context": run_context})
            output = result.get("output", {})

            # Log successful graph invocation with full context
            logger.info(
                "universal_graph_invoked",
                extra={
                    "origin": build.origin,
                    "tenant_id": run_context["tenant_id"],
                    "trace_id": run_context["trace_id"],
                    "case_id": run_context["case_id"],
                    "workflow_id": run_context["workflow_id"],
                    "ingestion_run_id": canonical_ingestion_run_id,
                    "decision": output.get("decision"),
                    "reason": output.get("reason"),
                    "document_id": output.get("document_id"),
                    "transitions_count": len(output.get("transitions", [])),
                },
            )
        except Exception as exc:
            logger.exception(
                "universal_crawler_ingestion_failed",
                extra={
                    "origin": build.origin,
                    "tenant_id": run_context["tenant_id"],
                    "trace_id": run_context["trace_id"],
                    "case_id": run_context["case_id"],
                    "workflow_id": run_context["workflow_id"],
                    "ingestion_run_id": canonical_ingestion_run_id,
                },
            )
            output = {
                "decision": "error",
                "reason": str(exc),
                "ingestion_run_id": None,
                "telemetry": {},
                "transitions": {},
                "artifacts": {"errors": [{"message": str(exc)}]},
            }
            result = {"output": output}  # Ensure result is dict so we can use it below

        # Map to legacy entry format for response builder
        # Synthesize state from output
        # Per contract: artifacts in root state, transitions in output
        synthesized_state = {
            "artifacts": result.get("artifacts", {}),
            "transitions": output.get("transitions", []),
            "control": build.state.get("control", {}),
        }

        # DO NOT set ingest_action - Universal Graph handles ingestion internally
        # No legacy start_ingestion_run should be triggered

        entry = {
            "build": build,
            "result": {
                "decision": output.get("decision"),
                "reason": output.get("reason"),
            },
            "state": synthesized_state,
        }

        # Validate and use canonical ingestion_run_id
        output_run_id = output.get("ingestion_run_id")
        if output_run_id and output_run_id != canonical_ingestion_run_id:
            logger.warning(
                "ingestion_run_id_mismatch",
                extra={
                    "context_id": canonical_ingestion_run_id,
                    "output_id": output_run_id,
                    "origin": build.origin,
                    "tenant_id": run_context["tenant_id"],
                    "trace_id": run_context["trace_id"],
                },
            )

        # Always use canonical ID (coordinator is source of truth)
        entry["ingestion_run_id"] = canonical_ingestion_run_id

        completed_runs.append(entry)

    guardrail_error = _detect_guardrail_error(completed_runs)
    if guardrail_error:
        return CrawlerRunnerCoordinatorResult(
            payload=guardrail_error,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            idempotency_key=idempotency_key,
        )

    payload = _build_synchronous_payload(
        request_model=request_model,
        workflow_id=str(workflow_resolved),
        completed_runs=completed_runs,
        meta=meta,
        idempotent_flag=idempotent_flag,
    )

    return CrawlerRunnerCoordinatorResult(
        payload=payload,
        status_code=status.HTTP_200_OK,
        idempotency_key=idempotency_key,
    )


def _build_ingest_specs(
    *,
    builds: list[CrawlerStateBundle],
    embedding_profile: str | None,
    scope: str | None,
) -> list[tuple[CrawlerStateBundle, DocumentIngestSpec]]:
    pairs: list[tuple[CrawlerStateBundle, DocumentIngestSpec]] = []

    for build in builds:
        normalized = build.state.get("normalized_document_input")
        if not isinstance(normalized, Mapping):
            continue

        metadata = normalized.get("meta")
        metadata_dict = dict(metadata or {})

        blob_payload = normalized.get("blob")
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
        ref_payload = normalized.get("ref")
        if collection_identifier is None and isinstance(ref_payload, Mapping):
            collection_identifier = ref_payload.get("collection_id")

        collections: list[str] = []
        if collection_identifier:
            collections.append(str(collection_identifier))

        spec = DocumentIngestSpec(
            source=str(source),
            content_hash=str(checksum),
            metadata=metadata_dict,
            collections=tuple(collections),
            embedding_profile=embedding_profile,
            scope=scope,
        )
        pairs.append((build, spec))

    return pairs


def _apply_ingest_result_to_build(
    build: CrawlerStateBundle, record: BulkIngestRecord
) -> None:
    document_id = str(record.result.document.id)
    build.document_id = document_id
    build.state["document_id"] = document_id

    normalized = build.state.get("normalized_document_input")
    if not isinstance(normalized, Mapping):
        return

    normalized_mutable = dict(normalized)
    ref_payload = normalized_mutable.get("ref")
    if isinstance(ref_payload, Mapping):
        ref_payload = dict(ref_payload)
    else:
        ref_payload = {}

    ref_payload["document_id"] = document_id
    collection_ids = record.result.collection_ids
    if collection_ids:
        collection_identifier = str(collection_ids[0])
        ref_payload["collection_id"] = collection_identifier
        build.collection_id = collection_identifier

    normalized_mutable["ref"] = ref_payload
    normalized_mutable["checksum"] = record.spec.content_hash
    build.state["normalized_document_input"] = normalized_mutable


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
                # Single source: state.transitions (populated from output.transitions)
                "transitions": _dump_jsonable(state_data.get("transitions", [])),
            }
        )
        errors_payload.extend(_extract_origin_errors(build, state_data))

        # Use existing ingestion_run_id if present (from Universal Graph)
        ingestion_run_id = entry.get("ingestion_run_id")
        if not ingestion_run_id:
            ingestion_run_id = _maybe_start_ingestion(build, state_data, meta)

        origins_payload.append(
            _summarize_origin_entry(build, state_data, result_payload, ingestion_run_id)
        )

    payload = {
        "workflow_id": workflow_id,
        "mode": request_model.mode,
        "collection_id": request_model.collection_id,
        "origins": origins_payload,
        "transitions": transitions_payload,
        "telemetry": telemetry_payload,
        "errors": errors_payload,
        "idempotent": idempotent_flag,
    }
    return payload


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
        context = tool_context_from_meta(meta)
        response = services_module.start_ingestion_run(
            payload, dict(meta), context.scope.idempotency_key
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
        serialised.append({"origin": build.origin, **_dump_jsonable(error)})
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
    chunk_count = 0
    artifacts = state.get("artifacts")
    if isinstance(artifacts, Mapping):
        parse_artifact = artifacts.get("parse_artifact") or artifacts.get(
            "prefetched_parse_result"
        )
        if isinstance(parse_artifact, Mapping):
            text_blocks = parse_artifact.get("text_blocks")
            if isinstance(text_blocks, (list, tuple)):
                chunk_count = len(text_blocks)

    entry: dict[str, object] = {
        "origin": build.origin,
        "provider": build.provider,
        "document_id": build.document_id,
        "result": _dump_jsonable(result_payload),
        "control": _dump_jsonable(control),
        "ingest_action": state.get("ingest_action"),
        "gating_score": state.get("gating_score"),
        "graph_run_id": state.get("graph_run_id") or result_payload.get("graph_run_id"),
        "state": _dump_jsonable(summary_state),
        "collection_id": build.collection_id,
        "review": build.review,
        "dry_run": build.dry_run,
        "chunk_count": chunk_count,
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
        return _dump_jsonable(processed)
    if isinstance(candidate, (list, tuple, set)):
        processed_list = [_serialise_guardrail_component(item) for item in candidate]
        return _dump_jsonable(processed_list)
    return _dump_jsonable(candidate)


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
