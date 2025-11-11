"""Business graph orchestrating software documentation collection flows."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_core.infra.observability import observe_span, update_observation
from ai_core.tools.web_search import (
    SearchProviderError,
    SearchResult,
    ToolOutcome,
    WebSearchResponse,
    WebSearchWorker,
)
from llm_worker.schemas import (
    FreshnessMode,
    HybridResult,
    ScoringContext,
    SearchCandidate,
)

LOGGER = logging.getLogger(__name__)


class InvalidGraphInput(ValueError):
    """Raised when the graph input payload cannot be validated."""


class StrategyGenerator(Protocol):
    """Callable generating web search strategies for software documentation."""

    def __call__(self, request: "SearchStrategyRequest") -> "SearchStrategy":
        """Return a deterministic search strategy for the given request."""


class HybridScoreExecutor(Protocol):
    """Adapter interface invoking the hybrid_search_and_score worker graph."""

    def run(
        self,
        *,
        scoring_context: ScoringContext,
        candidates: Sequence[SearchCandidate],
        tenant_context: Mapping[str, Any],
    ) -> HybridResult:
        """Execute the hybrid scorer and return the structured payload."""


class HitlGateway(Protocol):
    """Protocol describing the HITL approval surface."""

    def present(self, payload: Mapping[str, Any]) -> "HitlDecision | None":
        """Persist or publish the review payload and optionally return a decision."""


class IngestionTrigger(Protocol):
    """Protocol for handing approved URLs to the ingestion subsystem."""

    def trigger(
        self,
        *,
        approved_urls: Sequence[str],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Trigger ingestion and return metadata about the request."""


class CoverageVerifier(Protocol):
    """Protocol that verifies coverage delta after ingestion."""

    def verify(
        self,
        *,
        tenant_id: str,
        collection_scope: str,
        candidate_urls: Sequence[str],
        timeout_s: int,
        interval_s: int,
    ) -> Mapping[str, Any]:
        """Verify coverage improvements within the configured polling limits."""


class SearchStrategyRequest(BaseModel):
    """Normalised request payload for search strategy generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    query: str
    quality_mode: str

    @field_validator("tenant_id", "query", "quality_mode", mode="before")
    @classmethod
    def _trimmed(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        candidate = value.strip()
        if not candidate:
            raise ValueError("value must not be empty")
        return candidate


class SearchStrategy(BaseModel):
    """Structured search strategy containing query expansions and policy hints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    queries: list[str] = Field(min_length=1, max_length=7)
    policies_applied: tuple[str, ...] = Field(default_factory=tuple)
    preferred_sources: tuple[str, ...] = Field(default_factory=tuple)
    disallowed_sources: tuple[str, ...] = Field(default_factory=tuple)
    notes: str | None = None

    @field_validator("queries", mode="before")
    @classmethod
    def _normalise_queries(cls, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("queries must be a sequence")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("query expansions must be strings")
            candidate = item.strip()
            if not candidate:
                continue
            cleaned.append(candidate)
        if not cleaned:
            raise ValueError("at least one query expansion must be provided")
        if len(cleaned) > 7:
            raise ValueError("no more than seven query expansions are allowed")
        return cleaned

    @field_validator(
        "policies_applied", "preferred_sources", "disallowed_sources", mode="before"
    )
    @classmethod
    def _normalise_sequences(cls, value: Any) -> tuple[str, ...]:
        if value in (None, "", (), []):
            return ()
        if isinstance(value, str):
            candidate = value.strip()
            return (candidate,) if candidate else ()
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("value must be a sequence of strings")
        cleaned: list[str] = []
        for item in value:
            if item in (None, ""):
                continue
            cleaned_item = str(item).strip()
            if cleaned_item:
                cleaned.append(cleaned_item)
        return tuple(cleaned)

    @field_validator("notes", mode="before")
    @classmethod
    def _normalise_notes(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("notes must be a string")
        candidate = value.strip()
        return candidate or None


class GraphContextPayload(BaseModel):
    """Validated runtime context for the business graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    workflow_id: str
    case_id: str
    trace_id: str | None = None
    run_id: str | None = None
    ingestion_run_id: str | None = None

    @field_validator("tenant_id", "workflow_id", "case_id", mode="before")
    @classmethod
    def _trimmed_required(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        candidate = value.strip()
        if not candidate:
            raise ValueError("value must not be empty")
        return candidate

    @field_validator("trace_id", "run_id", "ingestion_run_id", mode="before")
    @classmethod
    def _normalise_optional(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value).strip() or None


class GraphInput(BaseModel):
    """Validated input payload for the software documentation graph."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1)
    collection_scope: str = Field(min_length=1)
    quality_mode: str = Field(default="standard")
    max_candidates: int = Field(default=20, ge=5, le=40)

    @field_validator("quality_mode", mode="before")
    @classmethod
    def _normalise_quality_mode(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("quality_mode must be a string")
        candidate = value.strip().lower()
        if not candidate:
            return "standard"
        return candidate


class HitlDecision(BaseModel):
    """Structured decision returned from the HITL gateway."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(pattern=r"^(pending|approved|rejected|partial)$")
    approved_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    rejected_candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    added_urls: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str | None = None

    @field_validator(
        "approved_candidate_ids",
        "rejected_candidate_ids",
        "added_urls",
        mode="before",
    )
    @classmethod
    def _normalise_ids(cls, value: Any) -> tuple[str, ...]:
        if value in (None, "", (), []):
            return ()
        if isinstance(value, str):
            candidate = value.strip()
            return (candidate,) if candidate else ()
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("decision values must be sequences of strings")
        cleaned: list[str] = []
        for item in value:
            if item in (None, ""):
                continue
            cleaned_item = str(item).strip()
            if cleaned_item:
                cleaned.append(cleaned_item)
        return tuple(cleaned)

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalise_rationale(cls, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str):
            raise ValueError("rationale must be a string")
        candidate = value.strip()
        return candidate or None


@dataclass(frozen=True)
class Transition:
    """Transition payload emitted by each node."""

    decision: str
    rationale: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "rationale": self.rationale,
            "meta": dict(self.meta),
        }


@dataclass
class _GraphIds:
    tenant_id: str
    workflow_id: str
    case_id: str
    trace_id: str
    run_id: str
    collection_scope: str
    ingestion_run_id: str | None = None

    def to_mapping(self) -> dict[str, str]:
        payload = {
            "tenant_id": self.tenant_id,
            "workflow_id": self.workflow_id,
            "case_id": self.case_id,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "collection_scope": self.collection_scope,
        }
        if self.ingestion_run_id:
            payload["ingestion_run_id"] = self.ingestion_run_id
        return payload


_FRESHNESS_MAP: dict[str, FreshnessMode] = {
    "standard": FreshnessMode.STANDARD,
    "software_docs_strict": FreshnessMode.SOFTWARE_DOCS_STRICT,
    "law_evergreen": FreshnessMode.LAW_EVERGREEN,
}

MIN_DIVERSITY_BUCKETS = 3


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class SoftwareDocumentationCollectionGraph:
    """Orchestrates search, hybrid scoring, HITL, and ingestion for docs."""

    _GRAPH_NAME = "software_documentation_collection"

    def __init__(
        self,
        *,
        strategy_generator: StrategyGenerator,
        search_worker: WebSearchWorker,
        hybrid_executor: HybridScoreExecutor,
        hitl_gateway: HitlGateway,
        ingestion_trigger: IngestionTrigger,
        coverage_verifier: CoverageVerifier,
    ) -> None:
        self._strategy_generator = strategy_generator
        self._search_worker = search_worker
        self._hybrid_executor = hybrid_executor
        self._hitl_gateway = hitl_gateway
        self._ingestion_trigger = ingestion_trigger
        self._coverage_verifier = coverage_verifier

    # ----------------------------------------------------------------- helpers
    def _prepare_ids(
        self,
        *,
        context: GraphContextPayload,
        collection_scope: str,
        meta_state: MutableMapping[str, Any],
    ) -> _GraphIds:
        trace_id = context.trace_id or str(uuid4())
        stored_run_id = str(meta_state.get("run_id") or context.run_id or uuid4())
        ingestion_run_id = (
            meta_state.get("ingestion_run_id") or context.ingestion_run_id
        )
        ids = _GraphIds(
            tenant_id=context.tenant_id,
            workflow_id=context.workflow_id,
            case_id=context.case_id,
            trace_id=trace_id,
            run_id=stored_run_id,
            collection_scope=collection_scope,
            ingestion_run_id=str(ingestion_run_id) if ingestion_run_id else None,
        )
        meta_state["run_id"] = ids.run_id
        if ids.ingestion_run_id:
            meta_state["ingestion_run_id"] = ids.ingestion_run_id
        context_snapshot = dict(meta_state.get("context") or {})
        context_snapshot.update(
            {
                "tenant_id": ids.tenant_id,
                "workflow_id": ids.workflow_id,
                "case_id": ids.case_id,
                "trace_id": ids.trace_id,
                "run_id": ids.run_id,
            }
        )
        if ids.ingestion_run_id:
            context_snapshot["ingestion_run_id"] = ids.ingestion_run_id
        meta_state["context"] = context_snapshot
        return ids

    def _base_meta(self, ids: _GraphIds) -> dict[str, Any]:
        base = ids.to_mapping()
        base["graph_name"] = self._GRAPH_NAME
        return base

    def _record_span_attributes(self, attributes: Mapping[str, Any]) -> None:
        if not attributes:
            return
        update_observation(
            metadata={
                str(key): value
                for key, value in attributes.items()
                if value is not None
            }
        )

    def _store_transition(
        self,
        state: MutableMapping[str, Any],
        name: str,
        transition: Transition,
    ) -> None:
        transitions = state.setdefault("transitions", [])
        transitions.append({"node": name, **transition.to_mapping()})

    def _append_telemetry(
        self,
        telemetry: MutableMapping[str, Any],
        name: str,
        details: Mapping[str, Any],
    ) -> None:
        telemetry.setdefault("nodes", {})[name] = dict(details)

    def _build_result(
        self,
        *,
        outcome: str,
        telemetry: Mapping[str, Any],
        hitl: Mapping[str, Any] | None,
        ingestion: Mapping[str, Any] | None,
        coverage: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload = {
            "outcome": outcome,
            "telemetry": dict(telemetry),
            "hitl": dict(hitl) if hitl else None,
            "ingestion": dict(ingestion) if ingestion else None,
            "coverage": dict(coverage) if coverage else None,
        }
        return payload

    # --------------------------------------------------------------------- nodes
    @observe_span(name="graph.software_docs.k_generate_strategy")
    def _k_generate_strategy(
        self,
        *,
        ids: _GraphIds,
        graph_input: GraphInput,
        run_state: MutableMapping[str, Any],
    ) -> Transition:
        request = SearchStrategyRequest(
            tenant_id=ids.tenant_id,
            query=graph_input.question,
            quality_mode=graph_input.quality_mode,
        )
        strategy = self._strategy_generator(request)
        plan = strategy.model_dump(mode="json")
        run_state.setdefault("strategy", {})["plan"] = plan
        attributes = self._base_meta(ids)
        attributes.update(
            {
                "query_count": len(strategy.queries),
                "policies": list(strategy.policies_applied),
                "quality_mode": graph_input.quality_mode,
            }
        )
        self._record_span_attributes(attributes)
        meta = self._base_meta(ids)
        meta.update(
            {
                "query_count": len(strategy.queries),
                "policies": list(strategy.policies_applied),
            }
        )
        return Transition("planned", "search_strategy_ready", meta)

    @observe_span(name="graph.software_docs.k_web_search")
    def _k_parallel_web_search(
        self,
        *,
        ids: _GraphIds,
        strategy: SearchStrategy,
        run_state: MutableMapping[str, Any],
    ) -> Transition:
        aggregated: list[dict[str, Any]] = []
        search_meta: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, query in enumerate(strategy.queries):
            worker_context = {
                "tenant_id": ids.tenant_id,
                "trace_id": ids.trace_id,
                "workflow_id": ids.workflow_id,
                "case_id": ids.case_id,
                "run_id": ids.run_id,
                "worker_call_id": f"search-{index}-{uuid4()}",
            }
            try:
                response: WebSearchResponse = self._search_worker.run(
                    query=query, context=worker_context
                )
            except SearchProviderError as exc:
                LOGGER.warning("web search provider failed", exc_info=exc)
                errors.append(
                    {
                        "query": query,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            outcome: ToolOutcome = response.outcome
            meta = dict(outcome.meta)
            meta.update(
                {
                    "decision": outcome.decision,
                    "rationale": outcome.rationale,
                    "query": query,
                    "worker_call_id": worker_context["worker_call_id"],
                    "result_count": len(response.results),
                }
            )
            if outcome.decision == "ok":
                for position, result in enumerate(response.results):
                    normalised = result.model_dump(mode="json")
                    normalised.update(
                        {
                            "query": query,
                            "query_index": index,
                            "position": position,
                        }
                    )
                    aggregated.append(normalised)
            else:
                errors.append(
                    {
                        "query": query,
                        "error": meta.get("error"),
                        "message": outcome.rationale,
                    }
                )
            search_meta.append(meta)
        run_state.setdefault("search", {})["responses"] = search_meta
        run_state["search"]["errors"] = errors
        run_state["search"]["results"] = aggregated
        attributes = self._base_meta(ids)
        attributes.update(
            {
                "total_results": len(aggregated),
                "queries": len(strategy.queries),
                "errors": len(errors),
            }
        )
        self._record_span_attributes(attributes)
        meta = self._base_meta(ids)
        meta.update(
            {
                "total_results": len(aggregated),
                "error_count": len(errors),
            }
        )
        decision = "results" if aggregated else "no_results"
        rationale = "web_search_completed" if aggregated else "web_search_empty"
        if errors and not aggregated:
            decision = "error"
            rationale = "web_search_failed"
        return Transition(decision, rationale, meta)

    @observe_span(name="graph.software_docs.k_hybrid_score")
    def _k_execute_hybrid_score(
        self,
        *,
        ids: _GraphIds,
        graph_input: GraphInput,
        strategy: SearchStrategy,
        search_results: Sequence[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
    ) -> tuple[Transition, HybridResult | None]:
        if not search_results:
            meta = self._base_meta(ids)
            return Transition("skipped", "no_candidates_to_score", meta), None
        max_candidates = graph_input.max_candidates
        sliced = list(search_results)[:max_candidates]
        candidates: list[SearchCandidate] = []
        candidate_lookup: dict[str, dict[str, Any]] = {}
        for item in sliced:
            url = item.get("url")
            try:
                result = SearchResult.model_validate(
                    {
                        "url": url,
                        "title": item.get("title") or "",
                        "snippet": item.get("snippet") or "",
                        "source": item.get("source") or "unknown",
                        "score": item.get("score"),
                        "is_pdf": bool(item.get("is_pdf")),
                    }
                )
            except ValidationError:
                continue
            candidate_payload = {
                "id": f"q{item.get('query_index', 0)}-{item.get('position', 0)}",
                "title": result.title,
                "snippet": result.snippet,
                "url": str(result.url),
                "is_pdf": bool(item.get("is_pdf")),
                "detected_date": item.get("detected_date"),
                "version_hint": item.get("version_hint"),
                "domain_type": item.get("domain_type"),
                "trust_hint": item.get("trust_hint"),
            }
            try:
                candidate = SearchCandidate.model_validate(candidate_payload)
            except ValidationError:
                continue
            candidates.append(candidate)
            candidate_lookup[candidate.id] = {
                "url": str(result.url),
                "title": result.title,
                "snippet": result.snippet,
                "source": result.source,
                "query": item.get("query"),
            }
        if not candidates:
            meta = self._base_meta(ids)
            return Transition("skipped", "no_valid_candidates", meta), None
        freshness_mode = _FRESHNESS_MAP.get(
            graph_input.quality_mode, FreshnessMode.STANDARD
        )
        scoring_context = ScoringContext(
            question=graph_input.question,
            purpose="software_documentation_collection",
            jurisdiction="DE",
            output_target="hybrid_rerank",
            preferred_sources=list(strategy.preferred_sources),
            disallowed_sources=list(strategy.disallowed_sources),
            collection_scope=graph_input.collection_scope,
            version_target=None,
            freshness_mode=freshness_mode,
            min_diversity_buckets=MIN_DIVERSITY_BUCKETS,
        )
        tenant_context = {
            "tenant_id": ids.tenant_id,
            "trace_id": ids.trace_id,
            "workflow_id": ids.workflow_id,
            "case_id": ids.case_id,
            "run_id": ids.run_id,
        }
        try:
            hybrid_result = self._hybrid_executor.run(
                scoring_context=scoring_context,
                candidates=candidates,
                tenant_context=tenant_context,
            )
        except Exception as exc:
            LOGGER.exception("hybrid scorer failed", exc_info=exc)
            meta = self._base_meta(ids)
            meta["error"] = {"kind": type(exc).__name__, "message": str(exc)}
            return Transition("error", "hybrid_executor_failed", meta), None
        run_state.setdefault("hybrid", {})["result"] = hybrid_result.model_dump(
            mode="json"
        )
        run_state["hybrid"]["candidates"] = candidate_lookup
        meta = self._base_meta(ids)
        meta.update(
            {
                "ranked_count": len(hybrid_result.ranked),
                "top_k_count": len(hybrid_result.top_k),
                "coverage_delta": hybrid_result.coverage_delta,
            }
        )
        attributes = dict(meta)
        self._record_span_attributes(attributes)
        return Transition("scored", "hybrid_score_completed", meta), hybrid_result

    @observe_span(name="graph.software_docs.k_hitl_gate")
    def _k_hitl_gate(
        self,
        *,
        ids: _GraphIds,
        graph_input: GraphInput,
        hybrid_result: HybridResult,
        run_state: MutableMapping[str, Any],
    ) -> tuple[Transition, dict[str, Any], HitlDecision | None]:
        now = datetime.now(timezone.utc)
        hitl_state = run_state.setdefault("hitl", {})
        existing_payload = hitl_state.get("payload")
        existing_payload = (
            existing_payload if isinstance(existing_payload, Mapping) else None
        )
        previous_deadline = _parse_iso_datetime(
            existing_payload.get("deadline_at") if existing_payload else None
        )
        raw_decision = hitl_state.get("decision")
        decision: HitlDecision | None = None
        auto_approved = bool(hitl_state.get("auto_approved", False))
        if raw_decision:
            try:
                decision = HitlDecision.model_validate(raw_decision)
            except ValidationError:
                decision = None

        if decision and decision.status != "pending":
            deadline_at = (existing_payload or {}).get("deadline_at") or (
                previous_deadline.isoformat() if previous_deadline else now.isoformat()
            )
        else:
            deadline_at = (existing_payload or {}).get("deadline_at")
            if decision is None and previous_deadline and now >= previous_deadline:
                approved_ids = tuple(item.candidate_id for item in hybrid_result.top_k)
                decision = HitlDecision(
                    status="approved",
                    approved_candidate_ids=approved_ids,
                    rejected_candidate_ids=(),
                    added_urls=(),
                    rationale="Auto-approved after HITL deadline",
                )
                hitl_state["decision"] = decision.model_dump(mode="json")
                hitl_state["auto_approved_at"] = now.isoformat()
                auto_approved = True
            if deadline_at is None:
                deadline_at = (now + timedelta(hours=24)).isoformat()
            elif isinstance(deadline_at, datetime):
                deadline_at = deadline_at.isoformat()

        payload = {
            "tenant_id": ids.tenant_id,
            "workflow_id": ids.workflow_id,
            "case_id": ids.case_id,
            "trace_id": ids.trace_id,
            "run_id": ids.run_id,
            "collection_scope": ids.collection_scope,
            "question": graph_input.question,
            "deadline_at": deadline_at,
            "coverage_delta": hybrid_result.coverage_delta,
            "ranked": [item.model_dump(mode="json") for item in hybrid_result.ranked],
            "top_k": [item.model_dump(mode="json") for item in hybrid_result.top_k],
            "recommended_ingest": [
                item.model_dump(mode="json")
                for item in hybrid_result.recommended_ingest
            ],
        }
        hitl_state["payload"] = payload
        hitl_state["auto_approved"] = auto_approved

        if decision is None:
            decision = self._hitl_gateway.present(payload)
            if decision is not None:
                hitl_state["decision"] = decision.model_dump(mode="json")
                if decision.status != "pending":
                    hitl_state.pop("auto_approved_at", None)
                    auto_approved = False
                    hitl_state["auto_approved"] = False

        attributes = self._base_meta(ids)
        attributes.update(
            {
                "deadline_at": payload["deadline_at"],
                "top_k_count": len(payload["top_k"]),
                "auto_approved": auto_approved,
            }
        )
        self._record_span_attributes(attributes)
        meta = self._base_meta(ids)
        meta.update(
            {
                "deadline_at": payload["deadline_at"],
                "decision_status": decision.status if decision else "pending",
                "auto_approved": auto_approved,
            }
        )

        if auto_approved and decision and decision.status != "pending":
            transition = Transition("auto_approved", "hitl_auto_approved", meta)
        elif decision is None or decision.status == "pending":
            transition = Transition("pending", "awaiting_hitl_decision", meta)
        else:
            transition = Transition("decided", "hitl_decision_recorded", meta)
        return transition, payload, decision

    @observe_span(name="graph.software_docs.k_trigger_ingestion")
    def _k_trigger_ingestion(
        self,
        *,
        ids: _GraphIds,
        decision: HitlDecision,
        run_state: MutableMapping[str, Any],
    ) -> tuple[Transition, Mapping[str, Any]]:
        approved_ids = list(decision.approved_candidate_ids)
        hybrid_state = run_state.get("hybrid", {})
        candidate_lookup = hybrid_state.get("candidates") or {}
        approved_urls: list[str] = []
        for candidate_id in approved_ids:
            item = candidate_lookup.get(candidate_id)
            url = item.get("url") if isinstance(item, Mapping) else None
            if isinstance(url, str) and url:
                approved_urls.append(url)
        for url in decision.added_urls:
            if isinstance(url, str) and url:
                approved_urls.append(url)
        context = {
            "tenant_id": ids.tenant_id,
            "workflow_id": ids.workflow_id,
            "case_id": ids.case_id,
            "collection_scope": ids.collection_scope,
            "trace_id": ids.trace_id,
            "run_id": ids.run_id,
        }
        ingestion_meta = self._ingestion_trigger.trigger(
            approved_urls=approved_urls,
            context=context,
        )
        run_state.setdefault("ingestion", {})["meta"] = dict(ingestion_meta)
        meta = self._base_meta(ids)
        meta.update(
            {
                "approved_urls": approved_urls,
                "ingestion_meta": dict(ingestion_meta),
            }
        )
        self._record_span_attributes(meta)
        transition = Transition("ingest_triggered", "ingestion_triggered", meta)
        return transition, ingestion_meta

    @observe_span(name="graph.software_docs.k_verify_coverage")
    def _k_verify_coverage(
        self,
        *,
        ids: _GraphIds,
        approved_urls: Sequence[str],
    ) -> tuple[Transition, Mapping[str, Any]]:
        verification = self._coverage_verifier.verify(
            tenant_id=ids.tenant_id,
            collection_scope=ids.collection_scope,
            candidate_urls=approved_urls,
            timeout_s=600,
            interval_s=30,
        )
        records: Sequence[Mapping[str, Any]] = []
        if isinstance(verification, Mapping):
            for key in ("results", "items", "entries", "records"):
                raw = verification.get(key)
                if isinstance(raw, Sequence):
                    records = [entry for entry in raw if isinstance(entry, Mapping)]
                    break
        total_candidates = len(approved_urls)
        reported = len(records)
        success_status = {"success", "completed", "complete", "ingested"}
        failure_status = {"failed", "error", "rejected"}
        pending_status = {"pending", "processing", "in_progress"}
        ingested = sum(
            1
            for record in records
            if str(record.get("status", "")).lower() in success_status
        )
        failed = sum(
            1
            for record in records
            if str(record.get("status", "")).lower() in failure_status
        )
        pending = sum(
            1
            for record in records
            if str(record.get("status", "")).lower() in pending_status
        )
        summary = {
            "total_candidates": total_candidates,
            "reported": reported,
            "ingested_count": ingested,
            "failed_count": failed,
            "pending_count": pending,
        }
        if total_candidates:
            summary["success_ratio"] = round(ingested / total_candidates, 3)
        meta = self._base_meta(ids)
        meta.update({"status": verification.get("status", "unknown"), **summary})
        self._record_span_attributes(meta)
        payload = dict(verification)
        payload["summary"] = summary
        transition = Transition("verified", "coverage_verified", meta)
        return transition, payload

    # ---------------------------------------------------------------------- run
    def run(
        self,
        state: Mapping[str, Any] | None,
        meta: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        working_state: dict[str, Any] = dict(state or {})
        meta_state: dict[str, Any] = working_state.setdefault("meta", {})
        telemetry: dict[str, Any] = working_state.setdefault("telemetry", {})
        telemetry.setdefault("graph", self._GRAPH_NAME)
        telemetry.setdefault("nodes", {})
        working_state.setdefault("transitions", [])

        context_source: dict[str, Any] = {}
        if meta is not None:
            context_source.update(meta)
        context_source.update(meta_state.get("context") or {})
        try:
            context_payload = GraphContextPayload.model_validate(context_source)
        except ValidationError as exc:
            raise InvalidGraphInput("invalid_context") from exc
        meta_state["context"] = context_payload.model_dump()

        input_source = dict(working_state.get("input") or {})
        input_source.setdefault("question", working_state.get("question"))
        input_source.setdefault(
            "collection_scope", working_state.get("collection_scope")
        )
        input_source.setdefault("quality_mode", working_state.get("quality_mode"))
        if (
            "max_candidates" not in input_source
            and working_state.get("max_candidates") is not None
        ):
            input_source["max_candidates"] = working_state.get("max_candidates")
        try:
            graph_input = GraphInput.model_validate(input_source)
        except ValidationError as exc:
            raise InvalidGraphInput("invalid_input") from exc
        working_state["input"] = graph_input.model_dump()

        ids = self._prepare_ids(
            context=context_payload,
            collection_scope=graph_input.collection_scope,
            meta_state=meta_state,
        )
        telemetry["ids"] = ids.to_mapping()
        meta_state["graph_name"] = self._GRAPH_NAME

        def _record(name: str, transition: Transition) -> None:
            self._store_transition(working_state, name, transition)
            self._append_telemetry(telemetry, name, transition.meta)

        # --------------------------------------------------------- generate plan
        strategy_transition = self._k_generate_strategy(
            ids=ids, graph_input=graph_input, run_state=working_state
        )
        _record("k_generate_strategy", strategy_transition)
        strategy = SearchStrategy.model_validate(working_state["strategy"]["plan"])

        # -------------------------------------------------------------- web search
        search_transition = self._k_parallel_web_search(
            ids=ids, strategy=strategy, run_state=working_state
        )
        _record("k_parallel_web_search", search_transition)
        search_results = working_state.get("search", {}).get("results") or []
        if search_transition.decision == "error":
            result = self._build_result(
                outcome="search_failed",
                telemetry=telemetry,
                hitl=None,
                ingestion=None,
                coverage=None,
            )
            return working_state, result
        if not search_results:
            result = self._build_result(
                outcome="no_candidates",
                telemetry=telemetry,
                hitl=None,
                ingestion=None,
                coverage=None,
            )
            return working_state, result

        # ----------------------------------------------------------- hybrid score
        hybrid_transition, hybrid_result = self._k_execute_hybrid_score(
            ids=ids,
            graph_input=graph_input,
            strategy=strategy,
            search_results=search_results,
            run_state=working_state,
        )
        _record("k_execute_hybrid_score", hybrid_transition)
        if hybrid_transition.decision == "error":
            result = self._build_result(
                outcome="hybrid_failed",
                telemetry=telemetry,
                hitl=None,
                ingestion=None,
                coverage=None,
            )
            return working_state, result
        if hybrid_result is None:
            result = self._build_result(
                outcome="no_scoring_result",
                telemetry=telemetry,
                hitl=None,
                ingestion=None,
                coverage=None,
            )
            return working_state, result

        # --------------------------------------------------------------- HITL gate
        hitl_transition, hitl_payload, decision = self._k_hitl_gate(
            ids=ids,
            graph_input=graph_input,
            hybrid_result=hybrid_result,
            run_state=working_state,
        )
        _record("k_hitl_approval_gate", hitl_transition)
        hitl_state = {
            "payload": hitl_payload,
            "decision": decision.model_dump(mode="json") if decision else None,
            "auto_approved": bool(working_state.get("hitl", {}).get("auto_approved")),
        }
        if decision is None or decision.status == "pending":
            result = self._build_result(
                outcome="awaiting_hitl",
                telemetry=telemetry,
                hitl=hitl_state,
                ingestion=None,
                coverage=None,
            )
            return working_state, result
        approved_ids = decision.approved_candidate_ids or ()
        candidate_lookup = working_state.get("hybrid", {}).get("candidates") or {}
        approved_urls = [
            item.get("url")
            for cid in approved_ids
            for item in [candidate_lookup.get(cid)]
            if isinstance(item, Mapping) and isinstance(item.get("url"), str)
        ]
        approved_urls.extend(url for url in decision.added_urls if isinstance(url, str))
        if not approved_urls:
            result = self._build_result(
                outcome="hitl_rejected",
                telemetry=telemetry,
                hitl=hitl_state,
                ingestion=None,
                coverage=None,
            )
            return working_state, result

        # -------------------------------------------------------- trigger ingestion
        ingestion_transition, ingestion_meta = self._k_trigger_ingestion(
            ids=ids,
            decision=decision,
            run_state=working_state,
        )
        _record("k_trigger_ingestion", ingestion_transition)

        # --------------------------------------------------------- verify coverage
        coverage_transition, coverage_meta = self._k_verify_coverage(
            ids=ids, approved_urls=approved_urls
        )
        _record("k_verify_coverage", coverage_transition)

        result = self._build_result(
            outcome="coverage_verified",
            telemetry=telemetry,
            hitl=hitl_state,
            ingestion=ingestion_meta,
            coverage=coverage_meta,
        )
        return working_state, result


# ================================================================== Production Factory


def _default_strategy_generator(request: SearchStrategyRequest) -> SearchStrategy:
    """Default strategy generator for software documentation searches."""
    base_query = request.query
    queries = [
        f"{base_query} documentation",
        f"{base_query} API reference",
        f"{base_query} guide",
    ]
    return SearchStrategy(
        queries=queries,
        policies_applied=("default",),
        preferred_sources=(),
        disallowed_sources=(),
    )


class _ProductionHitlGateway:
    """Production HITL gateway that persists payloads to dev-hitl storage."""

    def present(self, payload: Mapping[str, Any]) -> HitlDecision | None:
        """Persist the HITL payload and return None (awaiting manual approval)."""
        from theme.dev_hitl_store import store

        run_id = payload.get("run_id")
        if run_id:
            store.store_run(run_id, dict(payload))
        # Return None to indicate pending approval
        return None


class _ProductionIngestionTrigger:
    """Production ingestion trigger using crawler_runner API."""

    def trigger(
        self,
        *,
        approved_urls: Sequence[str],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Trigger ingestion via internal crawler API."""
        import httpx
        from django.urls import reverse

        tenant_id = context.get("tenant_id", "dev")
        collection_scope = context.get("collection_scope", "")
        trace_id = context.get("trace_id", "")
        case_id = context.get("case_id", "local")

        headers = {
            "Content-Type": "application/json",
            "X-Tenant-ID": str(tenant_id),
            "X-Case-ID": str(case_id),
            "X-Trace-ID": str(trace_id),
        }

        payload = {
            "workflow_id": "software-docs-ingestion",
            "mode": "live",
            "origins": [{"url": url} for url in approved_urls],
            "collection_id": collection_scope,
        }

        try:
            crawler_url = "http://localhost:8000" + reverse("ai_core:rag_crawler_run")
            with httpx.Client(timeout=30.0) as client:
                response = client.post(crawler_url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            return {"error": "crawler_failed", "status_code": response.status_code}
        except Exception as exc:
            return {"error": "trigger_failed", "message": str(exc)}


class _ProductionCoverageVerifier:
    """Production coverage verifier (currently a no-op)."""

    def verify(
        self,
        *,
        tenant_id: str,
        collection_scope: str,
        candidate_urls: Sequence[str],
        timeout_s: int,
        interval_s: int,
    ) -> Mapping[str, Any]:
        """Verify coverage improvements (currently returns success)."""
        return {
            "verified": True,
            "candidate_count": len(candidate_urls),
            "message": "Coverage verification not yet implemented",
        }


def build_graph() -> SoftwareDocumentationCollectionGraph:
    """Build a production-ready software documentation collection graph."""
    from ai_core.tools.shared_workers import get_web_search_worker
    from llm_worker.graphs.hybrid_search_and_score import run
    from llm_worker.schemas import HybridResult

    class _HybridExecutorAdapter:
        """Adapter for hybrid search and score worker."""

        def run(
            self,
            *,
            scoring_context,
            candidates,
            tenant_context,
        ) -> HybridResult:
            """Execute hybrid scoring via worker graph."""
            # Build state and meta as expected by hybrid_search_and_score.run
            state = {
                "candidates": [c.model_dump() for c in candidates],
            }
            meta = {
                "scoring_context": scoring_context.model_dump(),
                "tenant_id": tenant_context.get("tenant_id"),
                "trace_id": tenant_context.get("trace_id"),
                "case_id": tenant_context.get("case_id"),
            }
            _, result = run(state, meta)

            # Convert result to HybridResult
            return HybridResult.model_validate(result)

    # Use shared WebSearchWorker instance (singleton, created once)
    search_worker = get_web_search_worker()

    return SoftwareDocumentationCollectionGraph(
        strategy_generator=_default_strategy_generator,
        search_worker=search_worker,
        hybrid_executor=_HybridExecutorAdapter(),
        hitl_gateway=_ProductionHitlGateway(),
        ingestion_trigger=_ProductionIngestionTrigger(),
        coverage_verifier=_ProductionCoverageVerifier(),
    )
