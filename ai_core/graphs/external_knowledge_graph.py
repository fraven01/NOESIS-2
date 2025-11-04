"""External knowledge acquisition orchestrator graph."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    model_validator,
)

from ai_core.infra.observability import observe_span, update_observation
from ai_core.tools.search_adapters.google import GoogleSearchAdapter
from ai_core.tools.web_search import (
    SearchResult,
    ToolOutcome,
    WebSearchResponse,
    WebSearchWorker,
)
from django.conf import settings
from pydantic import SecretStr


class InvalidGraphInput(ValueError):
    """Raised when the graph input payload cannot be validated."""


class CrawlerIngestionAdapter(Protocol):
    """Adapter interface for triggering the crawler ingestion flow."""

    def trigger(
        self,
        *,
        url: str,
        collection_id: str,
        context: Mapping[str, str],
    ) -> "CrawlerIngestionOutcome":
        """Trigger the ingestion pipeline for *url* within *collection_id*."""


@dataclass(frozen=True)
class CrawlerIngestionOutcome:
    """Structured result returned by the crawler ingestion adapter."""

    decision: Literal["ingested", "skipped", "ingestion_error"]
    crawler_decision: str
    document_id: str | None = None


class ReviewEmitError(RuntimeError):
    """Raised when emitting a review payload fails."""


class ReviewEmitter(Protocol):
    """Protocol describing the HITL event emission surface."""

    def emit(self, payload: Mapping[str, Any]) -> None:
        """Persist or publish the pending review payload."""


class GraphContextPayload(BaseModel):
    """Validated runtime context for the external knowledge graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    workflow_id: str
    case_id: str
    trace_id: str | None = None
    run_id: str | None = None
    ingestion_run_id: str | None = None

    @model_validator(mode="after")
    def _normalise(self) -> "GraphContextPayload":
        values = self.model_dump()
        cleaned: dict[str, str | None] = {}
        for _field_name in (
            "tenant_id",
            "workflow_id",
            "case_id",
            "trace_id",
            "run_id",
            "ingestion_run_id",
        ):
            raw = values.get(_field_name)
            text = str(raw).strip() if raw is not None else ""
            cleaned[_field_name] = text or None
        if not cleaned["tenant_id"]:
            raise ValueError("tenant_id_required")
        if not cleaned["workflow_id"]:
            raise ValueError("workflow_id_required")
        if not cleaned["case_id"]:
            raise ValueError("case_id_required")
        object.__setattr__(self, "tenant_id", cleaned["tenant_id"])
        object.__setattr__(self, "workflow_id", cleaned["workflow_id"])
        object.__setattr__(self, "case_id", cleaned["case_id"])
        object.__setattr__(self, "trace_id", cleaned["trace_id"])
        object.__setattr__(self, "run_id", cleaned["run_id"])
        object.__setattr__(self, "ingestion_run_id", cleaned["ingestion_run_id"])
        return self


class GraphInput(BaseModel):
    """Validated graph input payload."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)
    collection_id: str = Field(min_length=1)
    enable_hitl: bool = False
    run_until: Literal["after_search", "after_selection", "review_complete"] | None = (
        None
    )


class _OverrideUrlPayload(BaseModel):
    """Strict override URL validation payload."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    url: HttpUrl


@dataclass(frozen=True)
class Transition:
    """Normalized transition payload emitted by each node."""

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
class ExternalKnowledgeGraphConfig:
    """Tuning parameters for the external knowledge graph."""

    top_n: int = 5
    prefer_pdf: bool = True
    blocked_domains: frozenset[str] = frozenset()
    min_snippet_length: int = 40
    run_until: Literal["after_search", "after_selection", "review_complete"] | None = (
        None
    )

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if self.min_snippet_length < 0:
            raise ValueError("min_snippet_length must not be negative")


@dataclass
class _GraphIds:
    tenant_id: str
    trace_id: str
    workflow_id: str
    case_id: str
    run_id: str
    collection_id: str
    ingestion_run_id: str | None = None

    def to_mapping(self) -> dict[str, str]:
        payload = {
            "tenant_id": self.tenant_id,
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "collection_id": self.collection_id,
        }
        if self.ingestion_run_id:
            payload["ingestion_run_id"] = self.ingestion_run_id
        return payload


class ExternalKnowledgeGraph:
    """Coordinate web search, selection, optional HITL, and ingestion."""

    _GRAPH_NAME = "external_knowledge"

    def __init__(
        self,
        *,
        search_worker: WebSearchWorker,
        ingestion_adapter: CrawlerIngestionAdapter,
        config: ExternalKnowledgeGraphConfig | None = None,
        review_emitter: ReviewEmitter | None = None,
    ) -> None:
        self._search_worker = search_worker
        self._ingestion_adapter = ingestion_adapter
        self._config = config or ExternalKnowledgeGraphConfig()
        self._review_emitter = review_emitter

    # ------------------------------------------------------------------ helpers
    def _prepare_ids(
        self,
        *,
        context: GraphContextPayload,
        collection_id: str,
        meta_state: MutableMapping[str, Any],
    ) -> _GraphIds:
        trace_id = context.trace_id or str(uuid4())
        stored_run_id = str(meta_state.get("run_id") or context.run_id or uuid4())
        ingestion_run_id = (
            meta_state.get("ingestion_run_id") or context.ingestion_run_id
        )
        ids = _GraphIds(
            tenant_id=context.tenant_id,
            trace_id=trace_id,
            workflow_id=context.workflow_id,
            case_id=context.case_id,
            run_id=stored_run_id,
            collection_id=collection_id,
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

    def _base_span_attributes(self, ids: _GraphIds) -> dict[str, Any]:
        return self._base_meta(ids)

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

    def _blocked_domain(self, url: str) -> bool:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname or not self._config.blocked_domains:
            return False
        for domain in self._config.blocked_domains:
            blocked = domain.lower()
            if hostname == blocked or hostname.endswith(f".{blocked}"):
                return True
        return False

    # ------------------------------------------------------------------ nodes
    @observe_span(name="graph.external_knowledge.k_search")
    def _k_search(
        self,
        *,
        query: str,
        ids: _GraphIds,
        run_state: MutableMapping[str, Any],
    ) -> Transition:
        worker_context = {
            "tenant_id": ids.tenant_id,
            "trace_id": ids.trace_id,
            "workflow_id": ids.workflow_id,
            "case_id": ids.case_id,
            "run_id": ids.run_id,
            "worker_call_id": str(uuid4()),
        }
        response: WebSearchResponse = self._search_worker.run(
            query=query, context=worker_context
        )
        outcome: ToolOutcome = response.outcome
        meta = dict(outcome.meta)
        result_count = meta.get("result_count", len(response.results))
        provider = str(
            meta.get("provider") or getattr(self._search_worker, "provider", "unknown")
        )
        attributes = self._base_span_attributes(ids)
        attributes.update(
            {
                "provider": provider,
                "query": query,
                "result.count": result_count,
                "worker_call_id": worker_context["worker_call_id"],
            }
        )
        error_meta = meta.get("error")
        if isinstance(error_meta, Mapping):
            attributes["error.kind"] = error_meta.get("kind")
        self._record_span_attributes(attributes)

        run_state["search"] = {
            "results": [result.model_dump(mode="json") for result in response.results],
            "meta": meta,
        }

        transition_meta = self._base_meta(ids)
        transition_meta.update(
            {
                "result.count": result_count,
                "provider": provider,
                "latency_ms": meta.get("latency_ms", 0),
                "worker_call_id": worker_context["worker_call_id"],
            }
        )

        if outcome.decision == "error":
            transition_meta["error"] = meta.get("error")
            transition = Transition("error", "search_error", transition_meta)
        elif response.results:
            transition = Transition(
                "results", "search_results_available", transition_meta
            )
        else:
            transition = Transition(
                "no_results", "search_returned_empty", transition_meta
            )
        return transition

    @observe_span(name="graph.external_knowledge.k_filter_and_select")
    def _k_filter_and_select(
        self,
        *,
        ids: _GraphIds,
        search_results: list[Mapping[str, Any]],
    ) -> tuple[Transition, Mapping[str, Any] | None, list[Mapping[str, Any]]]:
        validated: list[Mapping[str, Any]] = []
        rejected_count = 0
        for raw in search_results:
            try:
                result = SearchResult.model_validate(raw)
            except ValidationError:
                rejected_count += 1
                continue
            snippet = result.snippet.strip()
            if len(snippet) < self._config.min_snippet_length:
                rejected_count += 1
                continue
            lowered = snippet.lower()
            if "noindex" in lowered and "robot" in lowered:
                rejected_count += 1
                continue
            if self._blocked_domain(str(result.url)):
                rejected_count += 1
                continue
            validated.append(result.model_dump(mode="json"))
        shortlisted = validated[: self._config.top_n]
        selected: Mapping[str, Any] | None = None
        if shortlisted:
            if self._config.prefer_pdf:
                for candidate in shortlisted:
                    if candidate.get("is_pdf"):
                        selected = candidate
                        break
            if selected is None:
                selected = max(
                    shortlisted,
                    key=lambda item: float(item.get("score") or 0.0),
                )
        transition_meta = self._base_meta(ids)
        transition_meta.update(
            {
                "selected_url": selected.get("url") if selected else None,
                "rejected_count": rejected_count + (len(validated) - len(shortlisted)),
            }
        )
        attributes = self._base_span_attributes(ids)
        attributes.update(
            {
                "selected_url": selected.get("url") if selected else None,
                "rejected_count": transition_meta["rejected_count"],
            }
        )
        self._record_span_attributes(attributes)

        if selected is None:
            transition = Transition(
                "nothing_suitable", "no_candidate_selected", transition_meta
            )
        else:
            transition = Transition(
                "candidate_selected", "candidate_selected", transition_meta
            )
        return transition, selected, shortlisted

    def _pause_for_review(
        self,
        *,
        ids: _GraphIds,
        state: MutableMapping[str, Any],
        selection: Mapping[str, Any],
        query: str,
    ) -> dict[str, Any]:
        review_state = state.setdefault("review", {})
        if review_state.get("status") == "pending":
            return review_state
        review_token = str(uuid4())
        payload = {
            "status": "PENDING_REVIEW",
            "review_token": review_token,
            "tenant_id": ids.tenant_id,
            "trace_id": ids.trace_id,
            "workflow_id": ids.workflow_id,
            "case_id": ids.case_id,
            "run_id": ids.run_id,
            "collection_id": ids.collection_id,
            "selected_url": selection.get("url"),
            "query": query,
        }
        review_state.update(
            {
                "status": "pending",
                "review_token": review_token,
                "payload": payload,
            }
        )
        if self._review_emitter is not None:
            try:
                self._review_emitter.emit(payload)
            except Exception as exc:  # pragma: no cover - defensive
                raise ReviewEmitError("review_emit_failed") from exc
        return review_state

    @observe_span(name="graph.external_knowledge.k_hitl_gate")
    def _k_hitl_gate(
        self,
        *,
        ids: _GraphIds,
        state: MutableMapping[str, Any],
        selection: Mapping[str, Any],
        review_response: Mapping[str, Any],
    ) -> tuple[Transition, Mapping[str, Any]]:
        approved = bool(review_response.get("approved"))
        override_url = review_response.get("override_url")
        selected_url = selection.get("url")
        override_issue = False
        if isinstance(override_url, str) and override_url.strip():
            override_text = override_url.strip()
            try:
                validated_override = _OverrideUrlPayload.model_validate(
                    {"url": override_text}
                )
            except ValidationError:
                override_issue = True
            else:
                candidate_url = str(validated_override.url)
                if self._blocked_domain(candidate_url):
                    override_issue = True
                else:
                    selected_url = candidate_url
                    selection = dict(selection)
                    selection["url"] = selected_url
        if override_issue:
            approved = False
        meta = self._base_meta(ids)
        review_state = state.setdefault("review", {})
        review_token = review_state.get("review_token")
        if review_token:
            meta["review_token"] = review_token
        meta["approved"] = approved
        meta["selected_url"] = selected_url
        attributes = self._base_span_attributes(ids)
        attributes.update(
            {
                "review_token": review_token,
                "approved": approved,
            }
        )
        self._record_span_attributes(attributes)
        if override_issue:
            transition = Transition("rejected", "override_url_blocked_or_invalid", meta)
        elif approved:
            transition = Transition("approved", "hitl_approved", meta)
        else:
            transition = Transition("rejected", "hitl_rejected", meta)
        return transition, selection

    @observe_span(name="graph.external_knowledge.k_trigger_ingestion")
    def _k_trigger_ingestion(
        self,
        *,
        ids: _GraphIds,
        selection: Mapping[str, Any],
        run_state: MutableMapping[str, Any],
    ) -> Transition:
        ingestion_state = run_state.setdefault("ingestion", {})
        if not ids.ingestion_run_id:
            ids.ingestion_run_id = str(uuid4())
            run_state.setdefault("meta", {})["ingestion_run_id"] = ids.ingestion_run_id
        telemetry_state = run_state.setdefault("telemetry", {})
        telemetry_ids = telemetry_state.setdefault("ids", {})
        telemetry_ids.update(ids.to_mapping())
        context = ids.to_mapping()
        context["graph_name"] = self._GRAPH_NAME
        outcome = self._ingestion_adapter.trigger(
            url=str(selection.get("url")),
            collection_id=ids.collection_id,
            context=context,
        )
        ingestion_state.update(
            {
                "decision": outcome.decision,
                "crawler_decision": outcome.crawler_decision,
                "document_id": outcome.document_id,
            }
        )
        attributes = self._base_span_attributes(ids)
        attributes.update(
            {
                "document_id": outcome.document_id,
                "crawler_decision": outcome.crawler_decision,
            }
        )
        self._record_span_attributes(attributes)
        meta = self._base_meta(ids)
        meta.update(
            {
                "document_id": outcome.document_id,
                "crawler_decision": outcome.crawler_decision,
            }
        )
        transition = Transition(outcome.decision, "ingestion_completed", meta)
        return transition

    def _build_result(
        self,
        *,
        outcome: str,
        telemetry: Mapping[str, Any],
        selection: Mapping[str, Any] | None,
        document_id: str | None,
    ) -> dict[str, Any]:
        payload = {
            "outcome": outcome,
            "document_id": document_id,
            "selected_url": selection.get("url") if selection else None,
            "telemetry": dict(telemetry),
        }
        return payload

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

        input_source = {
            "query": working_state.get("query")
            or (working_state.get("input") or {}).get("query"),
            "collection_id": working_state.get("collection_id")
            or (working_state.get("input") or {}).get("collection_id"),
            "enable_hitl": working_state.get("enable_hitl", False),
            "run_until": working_state.get("run_until", self._config.run_until),
        }
        try:
            graph_input = GraphInput.model_validate(input_source)
        except ValidationError as exc:
            raise InvalidGraphInput("invalid_input") from exc

        ids = self._prepare_ids(
            context=context_payload,
            collection_id=graph_input.collection_id,
            meta_state=meta_state,
        )
        telemetry["ids"] = ids.to_mapping()
        meta_state["graph_name"] = self._GRAPH_NAME

        def _record(name: str, transition: Transition) -> None:
            self._store_transition(working_state, name, transition)
            self._append_telemetry(telemetry, name, transition.meta)

        # ---------------------------------------------------------------- search
        search_state: dict[str, Any] = working_state.setdefault("search", {})
        search_transition_mapping = search_state.get("transition")
        if not search_transition_mapping:
            try:
                search_transition = self._k_search(
                    query=graph_input.query,
                    ids=ids,
                    run_state=working_state,
                )
            except Exception as exc:  # pragma: no cover - defensive
                error_meta = self._base_meta(ids)
                error_meta["error"] = repr(exc)
                search_transition = Transition("error", "search_exception", error_meta)
                working_state.setdefault("search", {})
            search_state["transition"] = search_transition.to_mapping()
            search_state["completed"] = True
            _record("k_search", search_transition)
        else:
            search_transition = Transition(
                decision=search_transition_mapping.get("decision", ""),
                rationale=search_transition_mapping.get("rationale", ""),
                meta=search_transition_mapping.get("meta", {}),
            )

        if search_transition.decision == "error":
            result = self._build_result(
                outcome="error",
                telemetry=telemetry,
                selection=None,
                document_id=None,
            )
            return working_state, result
        if search_transition.decision == "no_results":
            result = self._build_result(
                outcome="no_results",
                telemetry=telemetry,
                selection=None,
                document_id=None,
            )
            return working_state, result
        if graph_input.run_until == "after_search":
            telemetry["stop_reason"] = "after_search"
            result = self._build_result(
                outcome="stopped_after_search",
                telemetry=telemetry,
                selection=None,
                document_id=None,
            )
            return working_state, result

        # ------------------------------------------------------------ filter/select
        selection_state: dict[str, Any] = working_state.setdefault("selection", {})
        selection_data = selection_state.get("selected")
        shortlisted = selection_state.get("shortlisted")
        if not selection_state.get("transition"):
            search_results = working_state.get("search", {}).get("results") or []
            transition, selected, shortlisted = self._k_filter_and_select(
                ids=ids,
                search_results=list(search_results),
            )
            selection_state["transition"] = transition.to_mapping()
            selection_state["selected"] = selected
            selection_state["shortlisted"] = shortlisted
            selection_state["completed"] = True
            _record("k_filter_and_select", transition)
            selection_data = selected
        else:
            transition_data = selection_state["transition"]
            selection_data = selection_state.get("selected")
            transition = Transition(
                decision=transition_data.get("decision", ""),
                rationale=transition_data.get("rationale", ""),
                meta=transition_data.get("meta", {}),
            )
        if selection_data is None:
            result = self._build_result(
                outcome="nothing_suitable",
                telemetry=telemetry,
                selection=None,
                document_id=None,
            )
            return working_state, result
        if graph_input.run_until == "after_selection":
            telemetry["stop_reason"] = "after_selection"
            result = self._build_result(
                outcome="stopped_after_selection",
                telemetry=telemetry,
                selection=selection_data,
                document_id=None,
            )
            return working_state, result

        # ------------------------------------------------------------------ HITL
        review_state: dict[str, Any] = working_state.setdefault("review", {})
        review_response = working_state.get("review_response")
        awaiting_review = review_state.get("status") == "pending"
        if graph_input.enable_hitl:
            if review_response is None and not awaiting_review:
                try:
                    review_state = self._pause_for_review(
                        ids=ids,
                        state=working_state,
                        selection=selection_data,
                        query=graph_input.query,
                    )
                except ReviewEmitError as exc:
                    meta = self._base_meta(ids)
                    error_meta = {
                        "kind": "EmitterError",
                        "message": str(exc.__cause__ or exc),
                    }
                    meta["error"] = error_meta
                    transition = Transition("error", "review_emit_failed", meta)
                    review_state = working_state.setdefault("review", {})
                    review_state["transition_emit"] = transition.to_mapping()
                    review_state["status"] = "error"
                    _record("k_hitl_gate_emit", transition)
                    telemetry.setdefault("review", {})
                    telemetry["review"].update(
                        {
                            "status": "error",
                            "review_token": review_state.get("review_token"),
                        }
                    )
                    result = self._build_result(
                        outcome="error",
                        telemetry=telemetry,
                        selection=selection_data,
                        document_id=None,
                    )
                    return working_state, result
                telemetry.setdefault("review", {})
                telemetry["review"].update(
                    {
                        "status": "pending",
                        "review_token": review_state.get("review_token"),
                    }
                )
                result = self._build_result(
                    outcome="error",
                    telemetry=telemetry,
                    selection=selection_data,
                    document_id=None,
                )
                return working_state, result
            if review_response is None and awaiting_review:
                telemetry.setdefault("review", {})
                telemetry["review"].update(
                    {
                        "status": "pending",
                        "review_token": review_state.get("review_token"),
                    }
                )
                result = self._build_result(
                    outcome="error",
                    telemetry=telemetry,
                    selection=selection_data,
                    document_id=None,
                )
                return working_state, result
            if review_response is not None:
                try:
                    hitl_transition, selection_data = self._k_hitl_gate(
                        ids=ids,
                        state=working_state,
                        selection=selection_data,
                        review_response=review_response,
                    )
                except InvalidGraphInput:
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    meta = self._base_meta(ids)
                    meta["error"] = repr(exc)
                    hitl_transition = Transition("error", "hitl_exception", meta)
                review_state["transition"] = hitl_transition.to_mapping()
                review_state["status"] = hitl_transition.decision
                _record("k_hitl_gate", hitl_transition)
                telemetry.setdefault("review", {})
                telemetry["review"].update(
                    {
                        "status": hitl_transition.decision,
                        "review_token": review_state.get("review_token"),
                    }
                )
                if hitl_transition.decision == "rejected":
                    result = self._build_result(
                        outcome="rejected",
                        telemetry=telemetry,
                        selection=selection_data,
                        document_id=None,
                    )
                    return working_state, result
                if hitl_transition.decision == "error":
                    result = self._build_result(
                        outcome="error",
                        telemetry=telemetry,
                        selection=selection_data,
                        document_id=None,
                    )
                    return working_state, result

        if graph_input.run_until == "review_complete":
            telemetry["stop_reason"] = "review_complete"
            result = self._build_result(
                outcome="stopped_after_review",
                telemetry=telemetry,
                selection=selection_data,
                document_id=None,
            )
            return working_state, result

        # ------------------------------------------------------------- ingestion
        try:
            ingestion_transition = self._k_trigger_ingestion(
                ids=ids,
                selection=selection_data,
                run_state=working_state,
            )
        except Exception as exc:
            meta = self._base_meta(ids)
            meta["error"] = repr(exc)
            ingestion_transition = Transition(
                "ingestion_error", "ingestion_failed", meta
            )
        _record("k_trigger_ingestion", ingestion_transition)
        outcome = ingestion_transition.decision
        document_id = ingestion_transition.meta.get("document_id")
        telemetry["ids"] = ids.to_mapping()
        result = self._build_result(
            outcome=outcome,
            telemetry=telemetry,
            selection=selection_data,
            document_id=document_id if isinstance(document_id, str) else None,
        )
        return working_state, result


def build_graph(
    *,
    ingestion_adapter: CrawlerIngestionAdapter,
    config: ExternalKnowledgeGraphConfig | None = None,
    review_emitter: ReviewEmitter | None = None,
) -> ExternalKnowledgeGraph:
    """Construct a configured :class:`ExternalKnowledgeGraph`."""

    search_adapter = GoogleSearchAdapter(
        api_key=SecretStr(settings.GOOGLE_CUSTOM_SEARCH_API_KEY),
        search_engine_id=settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
    )
    search_worker = WebSearchWorker(search_adapter)

    return ExternalKnowledgeGraph(
        search_worker=search_worker,
        ingestion_adapter=ingestion_adapter,
        config=config,
        review_emitter=review_emitter,
    )


__all__ = [
    "CrawlerIngestionAdapter",
    "CrawlerIngestionOutcome",
    "ExternalKnowledgeGraph",
    "ExternalKnowledgeGraphConfig",
    "InvalidGraphInput",
    "ReviewEmitError",
    "ReviewEmitter",
    "Transition",
    "build_graph",
]
