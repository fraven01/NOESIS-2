"""Crawler ingestion orchestration expressed as a LangGraph-style state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import secrets
import time
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple
from uuid import UUID

from crawler.contracts import NormalizedSource
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
    IngestionDecision,
    IngestionPayload,
    IngestionStatus,
    build_ingestion_decision,
)
from crawler.normalizer import NormalizedDocument, build_normalized_document
from crawler.parser import (
    ParseResult,
    ParseStatus,
    ParserContent,
    ParserStats,
    build_parse_result,
)


def _generate_uuid7() -> UUID:
    """Generate a monotonic UUIDv7 compatible identifier."""

    unix_ts_ms = int(time.time_ns() // 1_000_000)
    unix_ts_ms &= (1 << 60) - 1
    rand = secrets.randbits(62)
    value = (unix_ts_ms << 68) | (0x7 << 64) | (0b10 << 62) | rand
    return UUID(int=value)


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
    ingestion_builder: Callable[..., IngestionDecision] = build_ingestion_decision
    upsert_handler: Callable[[IngestionDecision], Any] = lambda _: {"status": "queued"}
    retire_handler: Callable[[IngestionDecision], Any] = lambda decision: {
        "status": "retired",
        "lifecycle_state": decision.lifecycle_state.value,
    }
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
        control.update(
            {
                "manual_review": None,
                "force_retire": False,
                "recompute_delta": False,
            }
        )
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
        control["manual_review"] = "approved"
        updated["control"] = control
        return updated

    def reject_ingest(
        self, state: Mapping[str, Any] | MutableMapping[str, Any]
    ) -> Dict[str, Any]:
        updated = self._copy_state(state)
        control = dict(updated.get("control", {}))
        control["manual_review"] = "rejected"
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
        control.setdefault("shadow_mode", False)
        control.setdefault("manual_review", None)
        control.setdefault("force_retire", False)
        control.setdefault("policy_revision", 0)
        control.setdefault("recompute_delta", False)
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
        state["provider"] = normalized.external_ref.provider
        state["external_id"] = normalized.external_ref.external_id
        attributes = {
            "document_id": normalized.document_id,
            "media_type": normalized.meta.media_type,
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
        known_duplicates = params.get("known_near_duplicates")
        threshold = params.get("near_duplicate_threshold")
        binary_payload = params.get("binary_payload")
        check_for_changes = params.get("check_near_duplicates_for_changes", False)
        algorithm = params.get("hash_algorithm", "sha256")

        decision = self.delta_evaluator(
            normalized,
            previous_content_hash=previous_hash,
            previous_version=previous_version,
            known_near_duplicates=known_duplicates,
            near_duplicate_threshold=threshold if threshold is not None else 0.92,
            binary_payload=binary_payload,
            check_near_duplicates_for_changes=check_for_changes,
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
        manual_state = control.get("manual_review")
        if manual_state == "approved":
            decision = GuardrailDecision(
                GuardrailStatus.ALLOW,
                "manual_approved",
                ("manual_approved",),
            )
            control["manual_review"] = None
        elif manual_state == "rejected":
            decision = GuardrailDecision(
                GuardrailStatus.DENY,
                "manual_rejected",
                ("manual_rejected",),
            )
        else:
            params = state.get("gating_input", {})
            limits: Optional[GuardrailLimits] = params.get("limits")
            signals: Optional[GuardrailSignals] = params.get("signals")
            decision = self.guardrail_enforcer(limits=limits, signals=signals)
            if (
                decision.status is GuardrailStatus.DENY
                and control.get("manual_review") is None
            ):
                control["manual_review"] = "required"

        artifacts["guardrail_decision"] = decision
        allowed = decision.status is GuardrailStatus.ALLOW
        params = state.get("gating_input", {})
        score = params.get("score")
        if score is None:
            score = 1.0 if allowed else 0.0
        state["gating_score"] = float(score)
        attributes = {
            "status": decision.status.value,
            "policy_events": list(decision.policy_events),
            "manual_review": control.get("manual_review"),
        }
        if decision.status is GuardrailStatus.DENY:
            attributes["severity"] = (
                "warn" if control.get("manual_review") != "rejected" else "error"
            )
        outcome = _transition(
            decision.status.value, decision.reason, attributes=attributes
        )
        should_continue = allowed
        if not allowed and control.get("manual_review") == "rejected":
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
        state["ingest_action"] = ingestion.status.value
        payload: Optional[IngestionPayload] = ingestion.payload
        if payload is not None:
            state["external_id"] = payload.external_id
            state["provider"] = payload.provider
            state["origin_uri"] = payload.origin_uri
            state["content_hash"] = payload.content_hash
        attributes = {
            "status": ingestion.status.value,
            "lifecycle_state": ingestion.lifecycle_state.value,
            "policy_events": list(ingestion.policy_events),
        }
        if conflict_note:
            attributes["conflict_resolution"] = conflict_note
        outcome = _transition(
            ingestion.status.value, ingestion.reason, attributes=attributes
        )
        continue_to_upsert = ingestion.status is IngestionStatus.UPSERT
        continue_to_retire = ingestion.status is IngestionStatus.RETIRE
        return outcome, continue_to_upsert or continue_to_retire

    def _run_upsert(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        ingestion: Optional[IngestionDecision] = artifacts.get("ingestion_decision")
        if not ingestion or ingestion.status is not IngestionStatus.UPSERT:
            outcome = _transition(
                "skip",
                "not_applicable",
                attributes={"shadow_mode": control.get("shadow_mode", False)},
            )
            return outcome, True

        if control.get("shadow_mode"):
            artifacts["upsert_result"] = None
            outcome = _transition(
                "shadow_skip", "shadow_mode", attributes={"shadow_mode": True}
            )
            return outcome, True

        result = self.upsert_handler(ingestion)
        artifacts["upsert_result"] = result
        outcome = _transition(
            "upsert", "upsert_dispatched", attributes={"result": result}
        )
        return outcome, True

    def _run_retire(
        self,
        state: Dict[str, Any],
        artifacts: Dict[str, Any],
        control: Dict[str, Any],
    ) -> Tuple[Transition, bool]:
        ingestion: Optional[IngestionDecision] = artifacts.get("ingestion_decision")
        if not ingestion or ingestion.status is not IngestionStatus.RETIRE:
            outcome = _transition("skip", "not_applicable", attributes={})
            return outcome, False

        result = self.retire_handler(ingestion)
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
