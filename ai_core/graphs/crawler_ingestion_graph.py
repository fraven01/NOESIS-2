"""LangGraph inspired orchestration for crawler ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
import traceback
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple
from uuid import uuid4

from ai_core import api as ai_core_api
from ai_core.api import EmbeddingResult
from documents.api import LifecycleStatusUpdate, NormalizedDocumentPayload
from .document_service import DocumentLifecycleService, DocumentsApiLifecycleService

StateMapping = Mapping[str, Any] | MutableMapping[str, Any]


@dataclass(frozen=True)
class GraphTransition:
    """Standard transition payload returned by graph nodes."""

    decision: str
    reason: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.decision or "").strip():
            raise ValueError("decision_required")
        if not str(self.reason or "").strip():
            raise ValueError("reason_required")
        attributes = dict(self.attributes or {})
        attributes.setdefault("severity", "info")
        object.__setattr__(self, "attributes", attributes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class GraphNode:
    """Tie a node name to an execution callable."""

    name: str
    runner: Callable[[Dict[str, Any]], Tuple[GraphTransition, bool]]

    def execute(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        return self.runner(state)


def _transition(
    decision: str,
    reason: str,
    *,
    attributes: Optional[Mapping[str, Any]] = None,
) -> GraphTransition:
    return GraphTransition(decision=decision, reason=reason, attributes=attributes or {})


class CrawlerIngestionGraph:
    """Minimal orchestration graph coordinating crawler ingestion."""

    def __init__(
        self,
        *,
        document_service: DocumentLifecycleService = DocumentsApiLifecycleService(),
        guardrail_enforcer: Callable[..., ai_core_api.GuardrailDecision] = ai_core_api.enforce_guardrails,
        delta_decider: Callable[..., ai_core_api.DeltaDecision] = ai_core_api.decide_delta,
        embedding_handler: Callable[..., EmbeddingResult] = ai_core_api.trigger_embedding,
        completion_builder: Callable[..., Mapping[str, Any]] = ai_core_api.build_completion_payload,
        event_emitter: Optional[Callable[[str, GraphTransition, str], None]] = None,
    ) -> None:
        self._document_service = document_service
        self._guardrail_enforcer = guardrail_enforcer
        self._delta_decider = delta_decider
        self._embedding_handler = embedding_handler
        self._completion_builder = completion_builder
        self._event_emitter = event_emitter

    def _emit(self, name: str, transition: GraphTransition, run_id: str) -> None:
        if self._event_emitter is not None:
            try:
                self._event_emitter(name, transition, run_id)
            except Exception:  # pragma: no cover - defensive best effort
                pass

    def run(
        self,
        state: StateMapping,
        meta: StateMapping | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        working_state: Dict[str, Any] = dict(state)
        meta_payload = dict(meta or {})
        working_state["meta"] = meta_payload
        working_state.setdefault("artifacts", {})
        working_state.setdefault("transitions", {})
        run_id = working_state.setdefault("graph_run_id", str(uuid4()))

        nodes = (
            GraphNode("normalize", self._run_normalize),
            GraphNode("update_status_normalized", self._run_update_status),
            GraphNode("enforce_guardrails", self._run_guardrails),
            GraphNode("decide_delta", self._run_delta),
            GraphNode("trigger_embedding", self._run_trigger_embedding),
        )

        transitions: Dict[str, GraphTransition] = {}
        continue_flow = True
        last_transition: Optional[GraphTransition] = None

        for node in nodes:
            if not continue_flow:
                break
            try:
                transition, continue_flow = node.execute(working_state)
            except Exception as exc:  # pragma: no cover - orchestrated error path
                transition = self._handle_node_error(working_state, node, exc)
                continue_flow = False
            transitions[node.name] = transition
            last_transition = transition
            self._emit(node.name, transition, run_id)

        finish_transition, _ = self._run_finish(working_state)
        transitions["finish"] = finish_transition
        self._emit("finish", finish_transition, run_id)

        working_state["transitions"] = {
            name: payload.to_dict() for name, payload in transitions.items()
        }
        summary = finish_transition if finish_transition else last_transition
        result = {
            "decision": summary.decision if summary else None,
            "reason": summary.reason if summary else None,
            "attributes": dict(summary.attributes) if summary else {},
            "graph_run_id": run_id,
            "transitions": {
                name: payload.to_dict() for name, payload in transitions.items()
            },
        }
        working_state["result"] = result
        return working_state, result

    def _require(self, state: Dict[str, Any], key: str) -> Any:
        if key not in state:
            raise KeyError(f"state_missing_{key}")
        return state[key]

    def _artifacts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
            state["artifacts"] = artifacts
        return artifacts

    def _run_normalize(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        raw_reference = self._require(state, "raw_document")
        normalized = self._document_service.normalize_from_raw(
            raw_reference=raw_reference,
            tenant_id=self._require(state, "tenant_id"),
            case_id=state.get("case_id"),
            request_id=state.get("request_id") or state.get("meta", {}).get("request_id"),
        )
        artifacts = self._artifacts(state)
        artifacts["normalized_document"] = normalized
        transition = _transition(
            "normalized",
            "document_normalized",
            attributes={
                "severity": "info",
                "document": normalized.to_dict(),
            },
        )
        return transition, True

    def _run_update_status(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts[
            "normalized_document"
        ]
        status = self._document_service.update_lifecycle_status(
            tenant_id=normalized.tenant_id,
            document_id=normalized.document_id,
            status="normalized",
            previous_status=state.get("previous_status"),
            workflow_id=normalized.document.ref.workflow_id,
            reason="document_normalized",
        )
        artifacts["status_update"] = status
        transition = _transition(
            "status_updated",
            "lifecycle_normalized",
            attributes={"severity": "info", "result": status.to_dict()},
        )
        return transition, True

    def _run_guardrails(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts[
            "normalized_document"
        ]
        decision = self._guardrail_enforcer(
            normalized_document=normalized,
            config=state.get("guardrails"),
        )
        artifacts["guardrail_decision"] = decision
        if not decision.allowed:
            status = self._document_service.update_lifecycle_status(
                tenant_id=normalized.tenant_id,
                document_id=normalized.document_id,
                status="deleted",
                workflow_id=normalized.document.ref.workflow_id,
                reason=decision.reason,
                policy_events=decision.attributes.get("policy_events"),
            )
            artifacts.setdefault("status_updates", []).append(status)
        transition = _transition(
            decision.decision,
            decision.reason,
            attributes=decision.attributes,
        )
        continue_flow = decision.allowed
        return transition, continue_flow

    def _run_delta(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts[
            "normalized_document"
        ]
        decision = self._delta_decider(
            normalized_document=normalized,
            baseline=state.get("baseline"),
        )
        artifacts["delta_decision"] = decision
        status_update = self._document_service.update_lifecycle_status(
            tenant_id=normalized.tenant_id,
            document_id=normalized.document_id,
            status="active",
            workflow_id=normalized.document.ref.workflow_id,
            reason=decision.reason,
        )
        artifacts.setdefault("status_updates", []).append(status_update)
        transition = _transition(
            decision.decision,
            decision.reason,
            attributes=decision.attributes,
        )
        return transition, True

    def _run_trigger_embedding(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        artifacts = self._artifacts(state)
        normalized: NormalizedDocumentPayload = artifacts[
            "normalized_document"
        ]
        delta: Optional[ai_core_api.DeltaDecision] = artifacts.get("delta_decision")
        if delta is None:
            transition = _transition(
                "skipped",
                "delta_missing",
                attributes={"severity": "warn"},
            )
            return transition, True
        if delta.decision not in {"new", "changed"}:
            transition = _transition(
                "skipped",
                "delta_not_applicable",
                attributes={"severity": "info", "delta": delta.decision},
            )
            return transition, True
        embedding_state = state.get("embedding") or {}
        result = self._embedding_handler(
            normalized_document=normalized,
            embedding_profile=embedding_state.get("profile"),
            tenant_id=normalized.tenant_id,
            case_id=state.get("case_id"),
            request_id=state.get("request_id"),
            vector_client=embedding_state.get("client"),
            vector_client_factory=embedding_state.get("client_factory"),
        )
        artifacts["embedding_result"] = result
        transition = _transition(
            "embedding_triggered",
            "embedding_enqueued",
            attributes={"severity": "info", "result": result.to_dict()},
        )
        return transition, True

    def _run_finish(self, state: Dict[str, Any]) -> Tuple[GraphTransition, bool]:
        artifacts = self._artifacts(state)
        normalized: Optional[NormalizedDocumentPayload] = artifacts.get(
            "normalized_document"
        )
        guardrail: Optional[ai_core_api.GuardrailDecision] = artifacts.get(
            "guardrail_decision"
        )
        delta: Optional[ai_core_api.DeltaDecision] = artifacts.get("delta_decision")
        embedding_result: Optional[EmbeddingResult] = artifacts.get("embedding_result")

        if normalized is None:
            transition = _transition(
                "error",
                "normalization_missing",
                attributes={"severity": "error"},
            )
            return transition, False

        if guardrail is None:
            guardrail = ai_core_api.GuardrailDecision(
                decision="allow", reason="default", attributes={"severity": "info"}
            )
        if delta is None:
            delta = ai_core_api.DeltaDecision(
                decision="unknown",
                reason="delta_missing",
                attributes={"severity": "warn"},
            )

        payload = self._completion_builder(
            normalized_document=normalized,
            decision=delta,
            guardrails=guardrail,
            embedding_result=embedding_result.to_dict() if embedding_result else None,
        )
        failure = artifacts.get("failure")
        severity = guardrail.attributes.get("severity", "info")
        decision_value = delta.decision
        reason_value = delta.reason
        if failure:
            severity = "error"
            decision_value = failure.get("decision", "error")
            reason_value = failure.get("reason", guardrail.reason)
            payload["failure"] = dict(failure)
        elif not guardrail.allowed:
            severity = "error"
            decision_value = "denied"
            reason_value = guardrail.reason
        transition = _transition(
            decision_value,
            reason_value,
            attributes={"severity": severity, "result": dict(payload)},
        )
        state["summary"] = payload
        return transition, False

    def _handle_node_error(
        self, working_state: Dict[str, Any], node: GraphNode, exc: Exception
    ) -> GraphTransition:
        artifacts = self._artifacts(working_state)
        artifacts.setdefault("errors", []).append(
            {
                "node": node.name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        artifacts["failure"] = {"decision": "error", "reason": f"{node.name}_failed"}
        normalized = artifacts.get("normalized_document")
        if isinstance(normalized, NormalizedDocumentPayload):
            try:
                status: LifecycleStatusUpdate = self._document_service.update_lifecycle_status(
                    tenant_id=normalized.tenant_id,
                    document_id=normalized.document_id,
                    status="deleted",
                    workflow_id=normalized.document.ref.workflow_id,
                    reason=f"{node.name}_failed",
                )
                artifacts.setdefault("status_updates", []).append(status)
            except Exception:  # pragma: no cover - best effort
                pass
        return _transition(
            "error",
            f"{node.name}_failed",
            attributes={"severity": "error", "error": repr(exc)},
        )


GRAPH = CrawlerIngestionGraph(document_service=DocumentsApiLifecycleService())


def build_graph() -> CrawlerIngestionGraph:
    return GRAPH


def run(state: StateMapping, meta: StateMapping | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return GRAPH.run(state, meta)


__all__ = ["CrawlerIngestionGraph", "GRAPH", "build_graph", "run", "GraphTransition"]
