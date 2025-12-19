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

# NOTE: build_universal_ingestion_graph imported lazily inside run_crawler_runner()
# to prevent OOM in test environments that mock heavy modules.
from customers.tenant_context import TenantContext
from documents.contract_utils import resolve_workflow_id
from documents.domain_service import (
    DocumentIngestSpec,
    BulkIngestRecord,
)

import ai_core.services as services_module
from . import _get_documents_repository, _make_json_safe
from .crawler_state_builder import build_crawler_state

logger = get_logger(__name__)


def debug_check_json_serializable(obj, path=""):
    import json
    import inspect

    class DebugEncoder(json.JSONEncoder):
        def default(self, o):
            if (
                inspect.ismethod(o)
                or inspect.isfunction(o)
                or inspect.isbuiltin(o)
                or type(o).__name__ == "method"
            ):
                raise Exception(f"DEBUG FOUND METHOD: {o} type={type(o)}")
            try:
                return super().default(o)
            except TypeError:
                raise Exception(f"DEBUG FOUND NON-SERIALIZABLE: {o} type={type(o)}")

    try:
        json.dumps(obj, cls=DebugEncoder)
    except Exception as exc:
        raise Exception(f"DEBUG CHECK FAILED at {path}: {exc}") from exc


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

    # Lazy import to prevent OOM in test environments
    from ai_core.graphs.technical.universal_ingestion_graph import (
        build_universal_ingestion_graph,
        UniversalIngestionInput,
    )

    if request_model.collection_id:
        meta["collection_id"] = request_model.collection_id

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
    idempotency_key = meta.get("idempotency_key")
    try:
        import json
        import hashlib

        fingerprint_payload = {
            "tenant_id": meta.get("tenant_id"),
            "case_id": meta.get("case_id"),
            "workflow_id": str(workflow_resolved),
            "collection_id": request_model.collection_id,
            "mode": request_model.mode,
            "origins": [
                origin.model_dump(mode="json") for origin in request_model.origins or []
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
    except Exception:
        fingerprint = None
    idempotent_flag = False
    if idempotency_key:
        idempotent_flag = True
    elif fingerprint:
        seen_cache = _CRAWLER_IDEMPOTENCY_CACHE
        idempotent_flag = fingerprint in seen_cache
        seen_cache.add(fingerprint)

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

        # Determine strict context
        # Note: Universal Graph expects tenant_id, trace_id, case_id in context
        run_context = {
            "tenant_id": meta.get("tenant_id"),
            "case_id": meta.get("case_id"),
            "trace_id": meta.get("trace_id"),
            "workflow_id": str(workflow_resolved),
            "ingestion_run_id": str(uuid4()),  # Generate one for the graph run
            "dry_run": request_model.dry_run,  # Pass dry_run in context for future support
            "idempotency_key": idempotency_key,  # Pass for graph-side tracking
        }

        try:
            result = graph_app.invoke({"input": input_payload, "context": run_context})
            output = result.get("output", {})
        except Exception as exc:
            logger.exception(
                "universal_crawler_ingestion_failed", extra={"origin": build.origin}
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
        # _summarize_origin_entry expects:
        # - state (with artifacts, transitions, control)
        # - result (with decision)
        # - ingestion_run_id (arg)

        # Synthesize state from output
        # Artifacts are in the root state (result), NOT in output for LangGraph usually
        # But if invoke returns state, result IS state.
        synthesized_state = {
            "artifacts": result.get("artifacts") or output.get("artifacts") or {},
            "transitions": output.get("transitions") or {},
            "control": build.state.get("control", {}),  # Preserve control from build
            # If successful, mark as having ingestion action so summary reflects it?
            # actually _summarize_origin_entry uses state.get("ingest_action")
        }

        # Legacy: ingest_action="upsert" meant "we triggered ingestion".
        # Universal: "ingested" decision means it's done.
        if output.get("decision") == "ingested":
            synthesized_state["ingest_action"] = "upsert"

        entry = {
            "build": build,
            "result": {
                "decision": output.get("decision"),
                "reason": output.get("reason"),
            },
            "state": synthesized_state,
        }

        # If Universal Graph provided an ingestion_run_id, pass it.
        # But wait, entry structure in _run_graph_inline returned inline_payload.
        # _summarize_origin_entry takes `ingestion_run_id` as separate arg.
        # So we attach it to the entry dict to use it later or modify how we call summarize.

        # Actually, let's look at loop in _build_synchronous_payload (lines 394+)
        # It iterates completed_runs.
        # It calls _maybe_start_ingestion.
        # We want to BYPASS _maybe_start_ingestion if Universal already did it.

        # Solution: Store ingestion_run_id in entry['result'] or similar, and update _maybe_start_ingestion
        # OR update _build_synchronous_payload to check for it.

        # But I can't easily change _build_synchronous_payload without replacing it.
        # Wait, _maybe_start_ingestion checks `state.get("ingest_action")`.
        # If I set `ingest_action` to `upsert`, it attempts `start_ingestion_run`.
        # I DO NOT want that.
        # So I should SET `ingest_action` to `None` or something else?
        # But if I set it to None, `_summarize_origin_entry` might not report "ingest_action".

        # Let's look at _summarize_origin_entry (line 537): "ingest_action": state.get("ingest_action").

        # If I want to REPORT "upsert" but NOT TRIGGER `start_ingestion_run`:
        # I can change `_maybe_start_ingestion` to check for existing `ingestion_run_id`?
        # OR I can spoof `_maybe_start_ingestion` behavior.

        # Better: Populate `entry["ingestion_run_id"]` locally,
        # and modify `_build_synchronous_payload` loop to use it.

        entry["ingestion_run_id"] = output.get("ingestion_run_id")

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


# Simple in-memory cache to flag repeat manual crawler requests during a process.
_CRAWLER_IDEMPOTENCY_CACHE: set[str] = set()


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
                "transitions": _make_json_safe(
                    state_data.get("transitions")
                    or result_payload.get("transitions")
                    or {}
                ),
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
    debug_check_json_serializable(payload, "sync_payload_internal")
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
        "result": _make_json_safe(result_payload),
        "control": _make_json_safe(control),
        "ingest_action": state.get("ingest_action"),
        "gating_score": state.get("gating_score"),
        "graph_run_id": state.get("graph_run_id") or result_payload.get("graph_run_id"),
        "state": _make_json_safe(summary_state),
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
