"""Crawler ingestion orchestration expressed as a LangGraph-style state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import secrets
import time
import inspect
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple
from uuid import UUID

from crawler.contracts import Decision, NormalizedSource
from crawler.delta import DeltaDecision, evaluate_delta
from crawler.fetcher import (
    FetchRequest,
    FetchResult,
    FetchStatus,
    evaluate_fetch_response,
)
from crawler.frontier import (
    CrawlSignals,
    FrontierAction,
    FrontierDecision,
    SourceDescriptor,
    decide_frontier_action,
)
from crawler.guardrails import (
    GuardrailDecision,
    GuardrailLimits,
    GuardrailSignals,
    GuardrailStatus,
    enforce_guardrails,
)
from crawler.ingestion import (
    IngestionStatus,
    build_ingestion_decision,
)
from documents.contracts import NormalizedDocument

from crawler.normalizer import (
    build_normalized_document,
    resolve_provider_reference,
)
from crawler.retire import LifecycleState
from crawler.parser import (
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)

from ai_core.rag import vector_client
from ai_core.rag.ingestion_contracts import ChunkMeta
from ai_core.rag.schemas import Chunk
from documents.repository import DocumentsRepository


def _generate_uuid7() -> UUID:
    """Generate a monotonic UUIDv7 compatible identifier."""

    unix_ts_ms = int(time.time_ns() // 1_000_000)
    unix_ts_ms &= (1 << 60) - 1
    rand = secrets.randbits(62)
    value = (unix_ts_ms << 68) | (0x7 << 64) | (0b10 << 62) | rand
    return UUID(int=value)


def _resolve_documents_repository() -> DocumentsRepository:
    services = import_module("ai_core.services")
    repository = services._get_documents_repository()  # type: ignore[attr-defined]
    if not isinstance(repository, DocumentsRepository):
        raise TypeError("documents_repository_invalid")
    return repository


def _resolve_vector_client():
    return vector_client.get_default_client()


def _invoke_handler(
    handler: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler(*args, **kwargs)

    params = signature.parameters.values()
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in params
    )
    if accepts_var_kwargs:
        return handler(*args, **kwargs)

    allowed_kwargs = {
        parameter.name
        for parameter in params
        if parameter.kind
        in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    }
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}

    try:
        return handler(*args, **filtered_kwargs)
    except TypeError:
        if not filtered_kwargs and kwargs:
            return handler(*args)
        raise


def _default_store_handler(
    document: NormalizedDocument,
    *,
    repository: Optional[DocumentsRepository] = None,
) -> Dict[str, object]:
    repo = repository or _resolve_documents_repository()
    stored = repo.upsert(document)
    return {
        "status": "stored",
        "document_id": str(stored.ref.document_id),
        "version": stored.ref.version,
    }


def _default_upsert_handler(
    decision: Decision,
    *,
    chunk: Optional[Chunk],
    vector_client_instance: Optional[object] = None,
) -> Dict[str, object]:
    if chunk is None:
        return {"status": "skipped", "reason": "missing_chunk"}
    client = vector_client_instance or _resolve_vector_client()
    written = client.upsert_chunks([chunk])
    return {
        "status": "upserted",
        "chunks_written": int(written),
        "document_id": chunk.meta.get("document_id"),
    }


def _default_retire_handler(
    decision: Decision,
    *,
    chunk: Optional[Chunk],
    vector_client_instance: Optional[object] = None,
) -> Dict[str, object]:
    lifecycle_attr = decision.attributes.get(
        "lifecycle_state", LifecycleState.ACTIVE
    )
    if isinstance(lifecycle_attr, LifecycleState):
        lifecycle_value = lifecycle_attr.value
    else:
        lifecycle_value = lifecycle_attr
    if chunk is None:
        return {
            "status": "retired",
            "lifecycle_state": lifecycle_value,
            "reason": "missing_chunk",
        }
    client = vector_client_instance or _resolve_vector_client()
    written = client.upsert_chunks([chunk])
    return {
        "status": "retired",
        "lifecycle_state": lifecycle_value,
        "chunks_written": int(written),
        "document_id": chunk.meta.get("document_id"),
    }


def _ingestion_status(decision: Optional[Decision]) -> Optional[IngestionStatus]:
    if decision is None:
        return None
    return IngestionStatus(decision.decision)


@dataclass(frozen=True)
class Transition:
    """Typed transition payload for graph node outcomes."""

    decision: str
    reason: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_decision = str(self.decision or "").strip()
        if not normalized_decision:
            raise ValueError("decision_required")
        normalized_reason = str(self.reason or "").strip()
        if not normalized_reason:
            raise ValueError("reason_required")
        object.__setattr__(self, "decision", normalized_decision)
        object.__setattr__(self, "reason", normalized_reason)
        attrs = dict(self.attributes or {})
        attrs.setdefault("severity", "info")
        object.__setattr__(self, "attributes", attrs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Transition":
        decision = payload.get("decision", "")
        reason = payload.get("reason", "")
        attributes = payload.get("attributes", {})
        return cls(decision=decision, reason=reason, attributes=attributes)


def _datetime_to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone().isoformat()


def _ensure_mapping(
    obj: Mapping[str, Any] | MutableMapping[str, Any] | None,
) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, MutableMapping):
        return dict(obj)
    return dict(obj)


def _get_review_value(control: Mapping[str, Any]) -> Optional[str]:
    review_value = control.get("review")
    if isinstance(review_value, str) and review_value:
        return review_value
    manual_value = control.get("manual_review")
    if isinstance(manual_value, str) and manual_value:
        return manual_value
    return None


def _set_review_value(control: MutableMapping[str, Any], value: Optional[str]) -> None:
    if value is None:
        control["review"] = None
        control["manual_review"] = None
        return
    control["review"] = value
    control["manual_review"] = value


def _transition(
    decision: str,
    reason: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
) -> Transition:
    return Transition(decision=decision, reason=reason, attributes=attributes or {})


RequiredStateKeys = (
    "tenant_id",
    "case_id",
    "workflow_id",
    "external_id",
    "origin_uri",
    "provider",
    "content_hash",
    "gating_score",
    "ingest_action",
)


@dataclass(frozen=True)
class CrawlerIngestionGraph:
    """State machine coordinating crawler ingestion decisions."""

    frontier_decider: Callable[..., FrontierDecision] = decide_frontier_action
    fetch_evaluator: Callable[..., FetchResult] = evaluate_fetch_response
    parse_builder: Callable[..., ParseResult] = build_parse_result
    normalizer: Callable[..., NormalizedDocument] = build_normalized_document
    delta_evaluator: Callable[..., DeltaDecision] = evaluate_delta
    guardrail_enforcer: Callable[..., GuardrailDecision] = enforce_guardrails
    ingestion_builder: Callable[..., Decision] = build_ingestion_decision
    store_handler: Callable[..., Any] = _default_store_handler
    upsert_handler: Callable[..., Any] = _default_upsert_handler
    retire_handler: Callable[..., Any] = _default_retire_handler
    event_emitter: Optional[Callable[[str, Transition, str], None]] = None

    def run(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        working_state = self._prepare_state(state, meta)
        working_state["ingest_action"] = "pending"
        working_state["transitions"] = {}
        graph_run_id = str(_generate_uuid7())
        working_state["graph_run_id"] = graph_run_id
        transitions: Dict[str, Transition] = {}
        artifacts: Dict[str, Any] = dict(working_state.get("artifacts", {}))
        control: Dict[str, Any] = working_state["control"]

        node_sequence = (
            ("crawler.frontier", self._run_frontier),
            ("crawler.fetch", self._run_fetch),
            ("crawler.parse", self._run_parse),
            ("crawler.normalize", self._run_normalize),
            ("crawler.delta", self._run_delta),
            ("crawler.gating", self._run_gating),
            ("crawler.ingest_decision", self._run_ingestion),
            ("crawler.store", self._run_store),
            ("rag.upsert", self._run_upsert),
            ("rag.retire", self._run_retire),
        )

        last_transition: Optional[Transition] = None
        should_continue = True

        for name, handler in node_sequence:
            if not should_continue:
                break
            outcome, should_continue = handler(working_state, artifacts, control)
            transitions[name] = outcome
            last_transition = outcome
            self._emit_event(name, outcome, graph_run_id)

        working_state["transitions"] = {
            node: payload.to_dict() for node, payload in transitions.items()
        }
        working_state["artifacts"] = artifacts

        summary = self._select_summary_transition(
            transitions,
            last_transition,
        )

        result = {
            "decision": working_state.get("ingest_action"),
            "reason": summary.reason if summary else None,
            "attributes": dict(summary.attributes) if summary else {},
            "graph_run_id": graph_run_id,
            "transitions": {
                node: payload.to_dict() for node, payload in transitions.items()
            },
        }
        return working_state, result

    def start_crawl(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        updated["transitions"] = {}
        updated["artifacts"] = {}
        updated.pop("graph_run_id", None)
        control = dict(updated.get("control", {}))
        review_value = _get_review_value(control)
        control["force_retire"] = False
        control["recompute_delta"] = False
        control["mode"] = control.get("mode", "ingest")
        control["dry_run"] = bool(control.get("dry_run", False))
        _set_review_value(control, review_value)
        updated["control"] = control
        updated["ingest_action"] = "pending"
        updated["gating_score"] = 0.0
        updated["content_hash"] = None
        return updated

    def shadow_mode_on(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        control["shadow_mode"] = True
        updated["control"] = control
        return updated

    def shadow_mode_off(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        control["shadow_mode"] = False
        updated["control"] = control
        return updated

    def approve_ingest(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        _set_review_value(control, "approved")
        updated["control"] = control
        return updated

    def reject_ingest(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        _set_review_value(control, "rejected")
        updated["control"] = control
        updated["ingest_action"] = "skip"
        return updated

    def policy_update(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        *,
        limits: Optional[GuardrailLimits] = None,
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        revision = control.get("policy_revision", 0)
        control["policy_revision"] = revision + 1
        updated["control"] = control
        if limits is not None:
            gating_input = dict(updated.get("gating_input", {}))
            gating_input["limits"] = limits
            updated["gating_input"] = gating_input
        return updated

    def retire(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        control["force_retire"] = True
        updated["control"] = control
        updated["ingest_action"] = "retire"
        return updated

    def recompute_delta(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        control["recompute_delta"] = True
        updated["control"] = control
        return updated

    def _prepare_state(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        base = dict(state)
        meta_mapping = _ensure_mapping(meta)
        for key in RequiredStateKeys:
            if key not in base:
                base[key] = meta_mapping.get(key)
        if not isinstance(base.get("transitions"), dict):
            base["transitions"] = {}
        if not isinstance(base.get("artifacts"), dict):
            base["artifacts"] = {}
        control = dict(base.get("control", {}))
        review_value = _get_review_value(control)
        control.setdefault("shadow_mode", False)
        control.setdefault("force_retire", False)
        control.setdefault("policy_revision", 0)
        control.setdefault("recompute_delta", False)
        control.setdefault("mode", "ingest")
        control.setdefault("dry_run", False)
        _set_review_value(control, review_value)
        base["control"] = control
        base["ingest_action"] = "pending"
        if base.get("gating_score") is None:
            base["gating_score"] = 0.0
        missing_required = [
            key
            for key in ("tenant_id", "case_id", "workflow_id")
            if base.get(key) in {None, ""}
        ]
        if missing_required:
            raise KeyError(
                "missing_required_state_keys",
                tuple(sorted(missing_required)),
            )
        return base

    def _copy_state(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        copied = dict(state)
        if "control" in copied and isinstance(copied["control"], Mapping):
            copied["control"] = dict(copied["control"])
        if "transitions" in copied and isinstance(copied["transitions"], Mapping):
            copied["transitions"] = dict(copied["transitions"])
        if "artifacts" in copied and isinstance(copied["artifacts"], Mapping):
            copied["artifacts"] = dict(copied["artifacts"])
        return copied

    def _emit_event(self, node: str, transition: Transition, graph_run_id: str) -> None:
        if self.event_emitter is None:
            return
        try:
            self.event_emitter(node, transition, graph_run_id)
        except Exception:
            # Observability hooks must not break graph execution.
            pass

    def _missing_artifact(
        self,
        artifact: str,
        *,
        reason: Optional[str] = None,
    ) -> Transition:
        description = reason or f"{artifact}_missing"
        return _transition(
            "missing_artifact",
            description,
            attributes={"artifact": artifact, "severity": "error"},
        )

    def _build_chunk(
        self,
        decision: Decision,
        artifacts: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        deleted_at: Optional[datetime] = None,
    ) -> Optional[Chunk]:
        chunk_meta_obj = decision.attributes.get("chunk_meta")
        meta_model: Optional[ChunkMeta]
        if isinstance(chunk_meta_obj, ChunkMeta):
            meta_model = chunk_meta_obj
        elif isinstance(chunk_meta_obj, Mapping):
            try:
                meta_model = ChunkMeta.model_validate(chunk_meta_obj)
            except Exception:
                meta_model = None
        else:
            meta_model = None
        if meta_model is None:
            return None

        meta_payload = meta_model.model_dump(exclude_none=True)

        for key in ("tenant_id", "case_id", "workflow_id", "collection_id"):
            value = meta_payload.get(key) or state.get(key)
            if value is not None:
                meta_payload[key] = str(value)

        document_id = meta_payload.get("document_id")
        if document_id is None:
            fallback_id = state.get("document_id")
            if fallback_id is not None:
                meta_payload["document_id"] = str(fallback_id)
        if not meta_payload.get("document_id"):
            meta_payload.pop("document_id", None)

        adapter_metadata = decision.attributes.get("adapter_metadata")
        if isinstance(adapter_metadata, Mapping):
            meta_payload.setdefault("adapter_metadata", dict(adapter_metadata))

        if deleted_at is not None:
            meta_payload["deleted_at"] = deleted_at.astimezone(timezone.utc).isoformat()
        else:
            meta_payload.pop("deleted_at", None)

        primary_text: str = ""
        parse_result: Optional[ParseResult] = artifacts.get("parse_result")
        if (
            parse_result is not None
            and isinstance(parse_result, ParseResult)
            and parse_result.content is not None
        ):
            primary_text = parse_result.content.primary_text or ""

        if not primary_text:
            normalized: Optional[NormalizedDocument] = artifacts.get("normalized_document")
            if isinstance(normalized, NormalizedDocument):
                blob = normalized.blob
                payload = getattr(blob, "decoded_payload", None)
                if callable(payload):
                    try:
                        decoded = payload()
                    except Exception:
                        decoded = None
                    if isinstance(decoded, (bytes, bytearray)):
                        try:
                            primary_text = decoded.decode("utf-8", errors="ignore")
                        except Exception:
                            primary_text = ""

        return Chunk(content=primary_text, meta=meta_payload, embedding=None, parents=None)

    def _run_frontier(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        params = state.get("frontier_input", {})
        descriptor: SourceDescriptor = params.get("descriptor")
        signals: Optional[CrawlSignals] = params.get("signals")
        robots = params.get("robots")
        host_policy = params.get("host_policy")
        host_state = params.get("host_state")
        now = params.get("now")

        decision = self.frontier_decider(
            descriptor,
            signals,
            robots=robots,
            host_policy=host_policy,
            host_state=host_state,
            now=now,
        )
        artifacts["frontier_decision"] = decision
        attributes = {
            "earliest_visit_at": _datetime_to_iso(decision.earliest_visit_at),
            "policy_events": list(decision.policy_events),
        }
        if decision.action is not FrontierAction.ENQUEUE:
            attributes["severity"] = "warn"
        outcome = _transition(
            decision.action.value, decision.reason, attributes=attributes
        )
        should_continue = decision.action is FrontierAction.ENQUEUE
        if not should_continue:
            state["ingest_action"] = decision.action.value
        return outcome, should_continue

    def _run_fetch(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        params = state.get("fetch_input", {})
        request: FetchRequest = params.get("request")
        status_code = params.get("status_code")
        body = params.get("body")
        headers = params.get("headers")
        elapsed = params.get("elapsed")
        retries = params.get("retries", 0)
        limits = params.get("limits")
        failure = params.get("failure")
        retry_reason = params.get("retry_reason")
        downloaded_bytes = params.get("downloaded_bytes")
        backoff_total_ms = params.get("backoff_total_ms", 0.0)

        result = self.fetch_evaluator(
            request,
            status_code=status_code,
            body=body,
            headers=headers,
            elapsed=elapsed,
            retries=retries,
            limits=limits,
            failure=failure,
            retry_reason=retry_reason,
            downloaded_bytes=downloaded_bytes,
            backoff_total_ms=backoff_total_ms,
        )
        artifacts["fetch_result"] = result
        attributes = {
            "status_code": result.metadata.status_code,
            "bytes_downloaded": result.telemetry.bytes_downloaded,
            "policy_events": list(result.policy_events),
        }
        should_continue = result.status is FetchStatus.FETCHED
        if not should_continue:
            attributes["severity"] = "error"
        outcome = _transition(
            result.status.value,
            result.detail or result.status.value,
            attributes=attributes,
        )
        if not should_continue:
            state["ingest_action"] = result.status.value
        return outcome, should_continue

    def _run_parse(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        fetch_result: Optional[FetchResult] = artifacts.get("fetch_result")
        if fetch_result is None:
            outcome = self._missing_artifact("fetch_result")
            state["ingest_action"] = "error"
            return outcome, False
        params = state.get("parse_input", {})
        status: ParseStatus = params.get("status")
        content: Optional[ParserContent] = params.get("content")
        stats: Optional[ParserStats] = params.get("stats")
        diagnostics = params.get("diagnostics")

        parse_result = self.parse_builder(
            fetch_result,
            status=status,
            content=content,
            stats=stats,
            diagnostics=diagnostics,
        )
        artifacts["parse_result"] = parse_result
        attributes = {
            "status": parse_result.status.value,
            "media_type": (
                parse_result.content.media_type if parse_result.content else None
            ),
        }
        should_continue = parse_result.status is ParseStatus.PARSED
        if not should_continue:
            attributes["severity"] = "error"
            state["ingest_action"] = "skip"
        outcome = _transition(
            parse_result.status.value, parse_result.status.value, attributes=attributes
        )
        return outcome, should_continue

    def _run_normalize(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        parse_result: Optional[ParseResult] = artifacts.get("parse_result")
        if parse_result is None:
            outcome = self._missing_artifact("parse_result")
            state["ingest_action"] = "error"
            return outcome, False
        params = state.get("normalize_input", {})
        source: NormalizedSource = params.get("source")
        document_id = params.get("document_id")
        tags = params.get("tags")
        tenant_id = state.get("tenant_id")
        workflow_id = state.get("workflow_id")

        normalized = self.normalizer(
            parse_result=parse_result,
            source=source,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            document_id=document_id,
            tags=tags,
        )
        artifacts["normalized_document"] = normalized
        state["origin_uri"] = normalized.meta.origin_uri
        provider_reference = resolve_provider_reference(normalized)
        state["provider"] = provider_reference.provider
        state["external_id"] = provider_reference.external_id
        attributes = {
            "document_id": str(normalized.ref.document_id),
            "media_type": getattr(normalized.blob, "media_type", None),
        }
        outcome = _transition("normalized", "normalized", attributes=attributes)
        return outcome, True

    def _run_delta(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        normalized: Optional[NormalizedDocument] = artifacts.get("normalized_document")
        if normalized is None:
            outcome = self._missing_artifact("normalized_document")
            state["ingest_action"] = "error"
            return outcome, False
        params = state.get("delta_input", {})
        previous_hash = params.get("previous_content_hash")
        previous_version = params.get("previous_version")
        binary_payload = params.get("binary_payload")
        algorithm = params.get("hash_algorithm", "sha256")
        parse_result: Optional[ParseResult] = artifacts.get("parse_result")
        primary_text: Optional[str] = None
        if (
            parse_result is not None
            and parse_result.status is ParseStatus.PARSED
            and parse_result.content is not None
        ):
            primary_text = parse_result.content.primary_text

        decision = self.delta_evaluator(
            normalized,
            primary_text=primary_text,
            previous_content_hash=previous_hash,
            previous_version=previous_version,
            binary_payload=binary_payload,
            hash_algorithm=algorithm,
        )
        artifacts["delta_decision"] = decision
        state["content_hash"] = decision.signatures.content_hash
        attributes = {
            "status": decision.status.value,
            "content_hash": decision.signatures.content_hash,
            "version": decision.version,
            "parent_document_id": decision.parent_document_id,
        }
        if control.get("recompute_delta"):
            attributes["recomputed"] = True
            control["recompute_delta_recent"] = True
            control["recompute_delta"] = False
        outcome = _transition(
            decision.status.value, decision.reason, attributes=attributes
        )
        return outcome, True

    def _run_gating(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        initial_review = _get_review_value(control)
        if initial_review == "approved":
            decision = GuardrailDecision.from_legacy(
                GuardrailStatus.ALLOW,
                "manual_approved",
                ("manual_approved",),
            )
            _set_review_value(control, None)
        elif initial_review == "rejected":
            decision = GuardrailDecision.from_legacy(
                GuardrailStatus.DENY,
                "manual_rejected",
                ("manual_rejected",),
            )
        else:
            params = state.get("gating_input", {})
            limits: Optional[GuardrailLimits] = params.get("limits")
            signals: Optional[GuardrailSignals] = params.get("signals")
            decision = self.guardrail_enforcer(limits=limits, signals=signals)
            if decision.status is GuardrailStatus.DENY and initial_review != "rejected":
                _set_review_value(control, "required")

        artifacts["guardrail_decision"] = decision
        allowed = decision.status is GuardrailStatus.ALLOW
        params = state.get("gating_input", {})
        score = params.get("score")
        if score is None:
            score = 1.0 if allowed else 0.0
        state["gating_score"] = float(score)
        resolved_review = _get_review_value(control) or initial_review
        attributes = {
            "status": decision.status.value,
            "policy_events": list(decision.policy_events),
            "review": resolved_review,
            "mode": control.get("mode"),
            "dry_run": bool(control.get("dry_run")),
        }
        if decision.status is GuardrailStatus.DENY:
            attributes["severity"] = (
                "warn" if resolved_review != "rejected" else "error"
            )
        outcome = _transition(
            decision.status.value, decision.reason, attributes=attributes
        )
        should_continue = allowed
        if not allowed and resolved_review == "rejected":
            state["ingest_action"] = "skip"
        return outcome, should_continue

    def _run_ingestion(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        normalized: Optional[NormalizedDocument] = artifacts.get("normalized_document")
        if normalized is None:
            outcome = self._missing_artifact("normalized_document")
            state["ingest_action"] = "error"
            return outcome, False
        delta_decision: Optional[DeltaDecision] = artifacts.get("delta_decision")
        if delta_decision is None:
            outcome = self._missing_artifact("delta_decision")
            state["ingest_action"] = "error"
            return outcome, False
        lifecycle = state.get("lifecycle_decision")
        conflict_note = None
        recompute_recent = control.pop("recompute_delta_recent", False)
        if control.get("force_retire") and (
            control.get("recompute_delta") or recompute_recent
        ):
            conflict_note = "retire_overrides_recompute"
            control["recompute_delta"] = False
        ingestion = self.ingestion_builder(
            normalized,
            delta_decision,
            case_id=state.get("case_id"),
            retire=control.get("force_retire", False),
            lifecycle=lifecycle,
        )
        artifacts["ingestion_decision"] = ingestion
        status = IngestionStatus(ingestion.decision)
        state["ingest_action"] = status.value
        ingestion_attrs = ingestion.attributes
        chunk_meta = ingestion_attrs.get("chunk_meta")
        adapter_metadata: Mapping[str, object] = ingestion_attrs.get(
            "adapter_metadata", {}
        )
        if chunk_meta is not None:
            state["external_id"] = getattr(chunk_meta, "external_id", None)
            state["content_hash"] = getattr(chunk_meta, "content_hash", None)
        if isinstance(adapter_metadata, Mapping):
            provider = adapter_metadata.get("provider")
            if provider:
                state["provider"] = provider
            origin = adapter_metadata.get("origin_uri")
            if origin:
                state["origin_uri"] = origin
        review_state = control.get("review")
        dry_run = bool(control.get("dry_run"))
        mode = control.get("mode", "ingest")
        blocked: list[str] = []
        if review_state == "required":
            blocked.append("review_required")
        if dry_run:
            blocked.append("dry_run")
        lifecycle_state = ingestion_attrs.get("lifecycle_state", LifecycleState.ACTIVE)
        if not isinstance(lifecycle_state, LifecycleState):
            lifecycle_state = LifecycleState(lifecycle_state)
        policy_events = ingestion_attrs.get("policy_events", ())
        if not isinstance(policy_events, Sequence):
            policy_events = (policy_events,) if policy_events else ()
        else:
            policy_events = tuple(policy_events)
        attributes = {
            "status": status.value,
            "lifecycle_state": lifecycle_state.value,
            "policy_events": list(policy_events),
            "mode": mode,
            "review": review_state,
            "dry_run": dry_run,
        }
        if conflict_note:
            attributes["conflict_resolution"] = conflict_note
        if blocked:
            attributes["blocked"] = blocked[0] if len(blocked) == 1 else tuple(blocked)
        store_allowed = (
            status is IngestionStatus.UPSERT
            and not blocked
            and mode in {"store_only", "ingest"}
        )
        upsert_allowed = (
            status is IngestionStatus.UPSERT
            and not blocked
            and mode == "ingest"
            and not control.get("shadow_mode")
        )
        attributes["store_allowed"] = store_allowed
        attributes["upsert_allowed"] = upsert_allowed
        outcome = _transition(
            status.value, ingestion.reason, attributes=attributes
        )
        continue_to_store = status is IngestionStatus.UPSERT
        continue_to_retire = status is IngestionStatus.RETIRE
        return outcome, continue_to_store or continue_to_retire

    def _run_store(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        ingestion: Optional[Decision] = artifacts.get("ingestion_decision")
        status = _ingestion_status(ingestion) if ingestion else None
        if status is not IngestionStatus.UPSERT:
            outcome = _transition(
                "skip",
                "not_applicable",
                attributes={
                    "mode": control.get("mode"),
                    "review": control.get("review"),
                    "dry_run": bool(control.get("dry_run")),
                },
            )
            return outcome, True

        normalized: Optional[NormalizedDocument] = artifacts.get("normalized_document")
        if normalized is None:
            missing = self._missing_artifact("normalized_document")
            attributes = dict(missing.attributes)
            attributes.update(
                {
                    "mode": control.get("mode"),
                    "review": control.get("review"),
                    "dry_run": bool(control.get("dry_run")),
                }
            )
            outcome = _transition(
                missing.decision, missing.reason, attributes=attributes
            )
            return outcome, False

        mode = control.get("mode", "ingest")
        review_state = control.get("review")
        dry_run = bool(control.get("dry_run"))
        attributes = {
            "mode": mode,
            "review": review_state,
            "dry_run": dry_run,
        }

        if review_state == "required":
            attributes["severity"] = "warn"
            artifacts["store_result"] = None
            outcome = _transition("skip", "review_required", attributes=attributes)
            return outcome, True

        if dry_run:
            artifacts["store_result"] = None
            outcome = _transition("skip", "dry_run", attributes=attributes)
            return outcome, True

        if mode not in {"store_only", "ingest"}:
            artifacts["store_result"] = None
            outcome = _transition("skip", "mode_disabled", attributes=attributes)
            return outcome, True

        repository = _resolve_documents_repository()
        result = _invoke_handler(
            self.store_handler,
            normalized,
            repository=repository,
        )
        artifacts["store_result"] = result
        attributes["result"] = result
        outcome = _transition("stored", "document_stored", attributes=attributes)
        return outcome, True

    def _run_upsert(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        ingestion: Optional[Decision] = artifacts.get("ingestion_decision")
        mode = control.get("mode", "ingest")
        dry_run = bool(control.get("dry_run"))
        review_state = control.get("review")
        status = _ingestion_status(ingestion) if ingestion else None
        if status is not IngestionStatus.UPSERT:
            outcome = _transition(
                "skip",
                "not_applicable",
                attributes={
                    "shadow_mode": control.get("shadow_mode", False),
                    "mode": mode,
                    "dry_run": dry_run,
                    "review": review_state,
                },
            )
            return outcome, True

        if review_state == "required":
            attributes = {
                "shadow_mode": control.get("shadow_mode", False),
                "mode": mode,
                "dry_run": dry_run,
                "review": review_state,
                "severity": "warn",
            }
            outcome = _transition("skip", "review_required", attributes=attributes)
            return outcome, True

        if dry_run:
            attributes = {
                "shadow_mode": control.get("shadow_mode", False),
                "mode": mode,
                "dry_run": True,
                "review": review_state,
            }
            outcome = _transition("skip", "dry_run", attributes=attributes)
            return outcome, True

        if mode != "ingest":
            attributes = {
                "shadow_mode": control.get("shadow_mode", False),
                "mode": mode,
                "dry_run": dry_run,
                "review": review_state,
            }
            outcome = _transition("skip", "mode_disabled", attributes=attributes)
            return outcome, True

        if control.get("shadow_mode"):
            artifacts["upsert_result"] = None
            outcome = _transition(
                "shadow_skip",
                "shadow_mode",
                attributes={
                    "shadow_mode": True,
                    "mode": mode,
                    "dry_run": dry_run,
                    "review": review_state,
                },
            )
            return outcome, True

        chunk = self._build_chunk(ingestion, artifacts, state)
        vector_client_instance = _resolve_vector_client()
        result = _invoke_handler(
            self.upsert_handler,
            ingestion,
            chunk=chunk,
            vector_client_instance=vector_client_instance,
        )
        artifacts["upsert_result"] = result
        outcome = _transition(
            "upsert",
            "upsert_dispatched",
            attributes={
                "result": result,
                "mode": mode,
                "dry_run": dry_run,
                "review": review_state,
                "shadow_mode": False,
            },
        )
        return outcome, True

    def _run_retire(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        ingestion: Optional[Decision] = artifacts.get("ingestion_decision")
        status = _ingestion_status(ingestion) if ingestion else None
        if status is not IngestionStatus.RETIRE:
            outcome = _transition("skip", "not_applicable", attributes={})
            return outcome, False

        chunk = self._build_chunk(
            ingestion,
            artifacts,
            state,
            deleted_at=datetime.now(timezone.utc),
        )
        vector_client_instance = _resolve_vector_client()
        result = _invoke_handler(
            self.retire_handler,
            ingestion,
            chunk=chunk,
            vector_client_instance=vector_client_instance,
        )
        artifacts["retire_result"] = result
        outcome = _transition("retire", ingestion.reason, attributes={"result": result})
        return outcome, False

    def _select_summary_transition(
        self,
        transitions: Mapping[str, Transition],
        last_transition: Optional[Transition],
    ) -> Optional[Transition]:
        preferred_order = (
            "crawler.ingest_decision",
            "crawler.gating",
            "crawler.delta",
            "crawler.normalize",
            "crawler.parse",
            "crawler.fetch",
            "crawler.frontier",
        )
        severity_rank = {"error": 3, "warn": 2, "info": 1}
        best_transition: Optional[Transition] = None
        best_severity = -1
        best_order_score = -1
        for name, payload in transitions.items():
            severity = str(payload.attributes.get("severity", "info"))
            severity_score = severity_rank.get(severity, 1)
            order_score = 0
            if name in preferred_order:
                order_score = len(preferred_order) - preferred_order.index(name)
            if severity_score > best_severity or (
                severity_score == best_severity and order_score > best_order_score
            ):
                best_transition = payload
                best_severity = severity_score
                best_order_score = order_score
        if best_transition is not None:
            return best_transition
        for key in preferred_order:
            payload = transitions.get(key)
            if payload:
                return payload
        return last_transition


GRAPH = CrawlerIngestionGraph()


def build_graph() -> CrawlerIngestionGraph:
    """Return the shared crawler ingestion graph instance."""

    return GRAPH


def run(
    state: Mapping[str, Any] | MutableMapping[str, Any],
    meta: Mapping[str, Any] | MutableMapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Module-level convenience delegating to :data:`GRAPH`."""

    return GRAPH.run(state, meta)


__all__ = ["CrawlerIngestionGraph", "GRAPH", "build_graph", "run"]
