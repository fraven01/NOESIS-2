"""Business graph orchestrating collection search and ingestion flows."""

from __future__ import annotations

import json
import logging
import math
import time
from hashlib import sha256
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ai_core.infra.observability import emit_event, observe_span, update_observation
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.llm.pricing import calculate_chat_completion_cost
from ai_core.rag.embeddings import EmbeddingClient
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
    """Callable generating web search strategies for collection search."""

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
    purpose: str

    @field_validator("tenant_id", "query", "quality_mode", "purpose", mode="before")
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
    """Validated input payload for the collection search graph."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1)
    collection_scope: str = Field(min_length=1)
    quality_mode: str = Field(default="standard")
    max_candidates: int = Field(default=20, ge=5, le=40)
    purpose: str = Field(min_length=1)
    auto_ingest: bool = Field(default=False)
    auto_ingest_top_k: int = Field(default=10, ge=1, le=20)
    auto_ingest_min_score: float = Field(default=60.0, ge=0.0, le=100.0)

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


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _calculate_generic_heuristics(
    result: Mapping[str, Any],
    query: str,
) -> float:
    """Calculate generic quality heuristics for a search result (0-100 score)."""
    score = 0.0

    title = str(result.get("title") or "").lower()
    snippet = str(result.get("snippet") or "").lower()
    url = str(result.get("url") or "").lower()
    query_lower = query.lower()

    # 1. Title relevance (0-30 points)
    if query_lower in title:
        score += 30.0
    elif any(word in title for word in query_lower.split() if len(word) > 3):
        score += 15.0

    # 2. Snippet quality (0-25 points)
    snippet_words = len(snippet.split())
    score += min(snippet_words / 20.0, 25.0)  # More context = better

    # 3. Query coverage in snippet (0-20 points)
    query_mentions = snippet.count(query_lower)
    score += min(query_mentions * 10.0, 20.0)

    # 4. URL quality penalties (0 to -20 points)
    if any(
        x in url
        for x in ["login", "signup", "register", "cookie-policy", "privacy-policy"]
    ):
        score -= 20.0

    # 5. Source position boost (small bonus for early results)
    position = result.get("position", 0)
    if position < 3:
        score += 5.0

    return max(0.0, min(score, 100.0))


class CollectionSearchGraph:
    """Orchestrates search, hybrid scoring, HITL, and ingestion for collections."""

    _GRAPH_NAME = "collection_search"

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

    def _search_snapshot(self, state: Mapping[str, Any]) -> dict[str, Any]:
        search_state = state.get("search") if isinstance(state, Mapping) else None
        if not isinstance(search_state, Mapping):
            search_state = {}
        strategy_state = state.get("strategy") if isinstance(state, Mapping) else None
        if not isinstance(strategy_state, Mapping):
            strategy_state = {}
        snapshot: dict[str, Any] = {
            "results": list(search_state.get("results") or []),
            "errors": list(search_state.get("errors") or []),
            "responses": list(search_state.get("responses") or []),
        }
        plan = strategy_state.get("plan")
        if isinstance(plan, Mapping):
            snapshot["strategy"] = dict(plan)
        return snapshot

    def _build_result(
        self,
        *,
        outcome: str,
        telemetry: Mapping[str, Any],
        hitl: Mapping[str, Any] | None,
        ingestion: Mapping[str, Any] | None,
        coverage: Mapping[str, Any] | None,
        search: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "outcome": outcome,
            "telemetry": dict(telemetry),
            "hitl": dict(hitl) if hitl else None,
            "ingestion": dict(ingestion) if ingestion else None,
            "coverage": dict(coverage) if coverage else None,
            "search": dict(search) if search else None,
        }
        return payload

    # --------------------------------------------------------------------- nodes
    @observe_span(name="graph.collection_search.k_generate_strategy")
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
            purpose=graph_input.purpose,
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

    @observe_span(name="graph.collection_search.k_web_search")
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
            query_start = time.time()
            try:
                response: WebSearchResponse = self._search_worker.run(
                    query=query, context=worker_context
                )
                query_latency = time.time() - query_start

                # Emit success event
                emit_event(
                    {
                        "event.name": "query.executed",
                        "query.index": index,
                        "query.text": query,
                        "query.result_count": len(response.results),
                        "query.latency_ms": int(query_latency * 1000),
                        "query.status": "success",
                        "query.decision": response.outcome.decision,
                    }
                )
            except SearchProviderError as exc:
                query_latency = time.time() - query_start
                LOGGER.warning("web search provider failed", exc_info=exc)

                # Emit failure event
                emit_event(
                    {
                        "event.name": "query.failed",
                        "query.index": index,
                        "query.text": query,
                        "query.latency_ms": int(query_latency * 1000),
                        "query.status": "error",
                        "query.error_type": type(exc).__name__,
                        "query.error_message": str(exc),
                    }
                )

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

    @observe_span(name="graph.collection_search.k_embedding_rank")
    def _k_embedding_rank(
        self,
        *,
        ids: _GraphIds,
        query: str,
        purpose: str,
        search_results: Sequence[Mapping[str, Any]],
        run_state: MutableMapping[str, Any],
        top_k: int = 20,
    ) -> Transition:
        """Rank search results using embeddings + generic heuristics."""
        if not search_results:
            meta = self._base_meta(ids)
            meta["ranked_count"] = 0
            return Transition("skipped", "no_results_to_rank", meta)

        rank_start = time.time()

        # Build query embedding text
        query_text = f"{query} {purpose}".strip()

        # Prepare texts for embedding (query + all snippets)
        texts_to_embed = [query_text]
        result_texts = []
        for result in search_results:
            title = str(result.get("title") or "")
            snippet = str(result.get("snippet") or "")
            combined = f"{title} {snippet}".strip()
            result_texts.append(combined)
            texts_to_embed.append(combined)

        # Get embeddings
        try:
            embedding_client = EmbeddingClient.from_settings()
            embedding_result = embedding_client.embed(texts_to_embed)
            embeddings = embedding_result.vectors
        except Exception as exc:
            LOGGER.warning("embedding_rank.embedding_failed", exc_info=exc)
            # Fallback: just use heuristics
            embeddings = []

        # Calculate scores
        scored_results = []
        query_embedding = embeddings[0] if len(embeddings) > 0 else []

        for idx, result in enumerate(search_results):
            result_embedding = embeddings[idx + 1] if len(embeddings) > idx + 1 else []

            # Embedding similarity score (0-100)
            embedding_score = 0.0
            if query_embedding and result_embedding:
                similarity = _cosine_similarity(query_embedding, result_embedding)
                embedding_score = similarity * 100.0  # Convert to 0-100

            # Heuristic score (0-100)
            heuristic_score = _calculate_generic_heuristics(result, query)

            # Hybrid score: 60% embedding + 40% heuristics
            hybrid_score = (0.6 * embedding_score) + (0.4 * heuristic_score)

            # Attach score to result
            scored_result = dict(result)
            scored_result["embedding_rank_score"] = hybrid_score
            scored_result["embedding_similarity"] = embedding_score
            scored_result["heuristic_score"] = heuristic_score
            scored_results.append(scored_result)

        # Sort by hybrid score (highest first)
        scored_results.sort(
            key=lambda r: r.get("embedding_rank_score", 0.0), reverse=True
        )

        # Take top-k
        top_results = scored_results[:top_k]

        # Update state
        run_state["search"]["results"] = top_results
        run_state.setdefault("embedding_rank", {})["scored_count"] = len(scored_results)
        run_state["embedding_rank"]["top_k"] = len(top_results)

        rank_latency = time.time() - rank_start

        # Telemetry
        attributes = self._base_meta(ids)
        attributes.update(
            {
                "input_count": len(search_results),
                "ranked_count": len(scored_results),
                "top_k_count": len(top_results),
                "latency_ms": int(rank_latency * 1000),
                "avg_embedding_score": (
                    sum(r.get("embedding_similarity", 0.0) for r in top_results)
                    / len(top_results)
                    if top_results
                    else 0.0
                ),
                "avg_heuristic_score": (
                    sum(r.get("heuristic_score", 0.0) for r in top_results)
                    / len(top_results)
                    if top_results
                    else 0.0
                ),
            }
        )
        self._record_span_attributes(attributes)

        meta = self._base_meta(ids)
        meta.update(
            {
                "ranked_count": len(scored_results),
                "top_k_count": len(top_results),
                "latency_ms": int(rank_latency * 1000),
            }
        )
        return Transition("ranked", "embedding_rank_completed", meta)

    @observe_span(name="graph.collection_search.k_hybrid_score")
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
            purpose="collection_search",
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

    @observe_span(name="graph.collection_search.k_hitl_gate")
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

    @observe_span(name="graph.collection_search.k_trigger_ingestion")
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

    @observe_span(name="graph.collection_search.k_verify_coverage")
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

        # Filter out runtime-only fields (e.g., ledger_logger from worker layer)
        # before validating against GraphContextPayload schema
        allowed_context_fields = {
            "tenant_id",
            "workflow_id",
            "case_id",
            "trace_id",
            "run_id",
            "ingestion_run_id",
        }
        context_source: dict[str, Any] = {}
        if meta is not None:
            context_source.update(
                {k: v for k, v in meta.items() if k in allowed_context_fields}
            )
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
        input_source.setdefault(
            "purpose",
            working_state.get("purpose") or "collection_search",
        )
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
        search_snapshot = self._search_snapshot(working_state)
        search_results = search_snapshot.get("results") or []
        if search_transition.decision == "error":
            result = self._build_result(
                outcome="search_failed",
                telemetry=telemetry,
                hitl=None,
                ingestion=None,
                coverage=None,
                search=search_snapshot,
            )
            return working_state, result
        if not search_results:
            result = self._build_result(
                outcome="no_candidates",
                telemetry=telemetry,
                hitl=None,
                ingestion=None,
                coverage=None,
                search=search_snapshot,
            )
            return working_state, result

        # --------------------------------------------------------- embedding rank
        embedding_rank_transition = self._k_embedding_rank(
            ids=ids,
            query=graph_input.question,
            purpose=graph_input.purpose,
            search_results=search_results,
            run_state=working_state,
            top_k=20,
        )
        _record("k_embedding_rank", embedding_rank_transition)
        # Refresh search snapshot after ranking
        search_snapshot = self._search_snapshot(working_state)

        # ---------------------------------------------------- auto-ingestion (optional)
        ingestion_payload: Mapping[str, Any] | None = None
        outcome = "search_completed"

        if graph_input.auto_ingest:
            # Extract ranked results with scores
            ranked_results = search_snapshot.get("results") or []

            # Score-based filtering with fallback logic
            min_score = graph_input.auto_ingest_min_score
            filtered_results = [
                r
                for r in ranked_results
                if r.get("embedding_rank_score", 0.0) >= min_score
            ]

            # Fallback: If less than 3 results with primary threshold, try 50
            if len(filtered_results) < 3 and min_score > 50.0:
                fallback_min = 50.0
                filtered_results = [
                    r
                    for r in ranked_results
                    if r.get("embedding_rank_score", 0.0) >= fallback_min
                ]
                telemetry["auto_ingest_fallback_threshold"] = {
                    "original_threshold": min_score,
                    "fallback_threshold": fallback_min,
                    "result_count": len(filtered_results),
                }

            # Error if no results meet minimum quality threshold
            if not filtered_results:
                telemetry["auto_ingest_insufficient_quality"] = {
                    "min_score": min_score,
                    "fallback_min": 50.0,
                }
                result = self._build_result(
                    outcome="auto_ingest_failed_quality_threshold",
                    telemetry=telemetry,
                    hitl=None,
                    ingestion=None,
                    coverage=None,
                    search=search_snapshot,
                )
                return working_state, result

            # Limit to top_k
            top_k_limit = min(graph_input.auto_ingest_top_k, len(filtered_results))
            selected_results = filtered_results[:top_k_limit]

            # Extract URLs
            approved_urls = [
                r["url"]
                for r in selected_results
                if isinstance(r.get("url"), str) and r["url"]
            ]

            if approved_urls:
                # Trigger ingestion
                context = {
                    "tenant_id": ids.tenant_id,
                    "workflow_id": ids.workflow_id,
                    "case_id": ids.case_id,
                    "collection_scope": ids.collection_scope,
                    "trace_id": ids.trace_id,
                    "run_id": ids.run_id,
                }

                try:
                    ingestion_meta = self._ingestion_trigger.trigger(
                        approved_urls=approved_urls,
                        context=context,
                    )
                    working_state.setdefault("ingestion", {})["meta"] = dict(
                        ingestion_meta
                    )
                    ingestion_payload = dict(ingestion_meta)
                    outcome = "auto_ingest_triggered"

                    avg_score = sum(
                        r.get("embedding_rank_score", 0.0) for r in selected_results
                    ) / len(selected_results)
                    telemetry["auto_ingest_triggered"] = {
                        "url_count": len(approved_urls),
                        "min_score": min_score,
                        "selected_count": len(selected_results),
                        "avg_score": avg_score,
                    }

                    # Record transition for telemetry
                    auto_ingest_meta = self._base_meta(ids)
                    auto_ingest_meta.update(
                        {
                            "url_count": len(approved_urls),
                            "min_score": min_score,
                            "avg_score": avg_score,
                        }
                    )
                    auto_ingest_transition = Transition(
                        "triggered", "auto_ingestion_triggered", auto_ingest_meta
                    )
                    _record("k_auto_ingest", auto_ingest_transition)

                except Exception as exc:
                    telemetry["auto_ingest_trigger_failed"] = {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                    outcome = "auto_ingest_trigger_failed"

        result = self._build_result(
            outcome=outcome,
            telemetry=telemetry,
            hitl=None,
            ingestion=ingestion_payload,
            coverage=None,
            search=search_snapshot,
        )
        return working_state, result


# ================================================================== Production Factory


def _fallback_strategy(request: SearchStrategyRequest) -> SearchStrategy:
    """Return a deterministic baseline strategy when LLM generation fails."""

    base_query = request.query
    purpose_hint = request.purpose.replace("_", " ")
    candidates = [
        base_query,
        f"{base_query} {purpose_hint}",
        f"{base_query} overview",
        f"{base_query} information",
        f"{base_query} guide",
    ]
    seen: set[str] = set()
    queries: list[str] = []
    for item in candidates:
        normalised = item.strip()
        if not normalised:
            continue
        if normalised.lower() in seen:
            continue
        seen.add(normalised.lower())
        queries.append(normalised)
        if len(queries) == 3:
            break
    return SearchStrategy(
        queries=queries,
        policies_applied=("default",),
        preferred_sources=(),
        disallowed_sources=(),
    )


def _fallback_with_reason(
    request: SearchStrategyRequest, message: str, error: Exception | None = None
) -> SearchStrategy:
    if error is not None:
        LOGGER.warning(message, exc_info=error)
    else:
        LOGGER.warning(message)
    return _fallback_strategy(request)


def _coerce_query_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence):
        raise ValueError("queries must be a sequence")
    queries: list[str] = []
    seen: set[str] = set()
    for item in value:
        if item in (None, ""):
            continue
        candidate = str(item).strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(candidate)
        if len(queries) == 5:
            break
    if len(queries) < 3:
        raise ValueError("at least three queries required")
    return queries


def _extract_strategy_payload(text: str) -> Mapping[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty strategy payload")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1])
            cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("payload does not contain a JSON object")
    fragment = cleaned[start : end + 1]
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError("invalid JSON payload") from exc
    if not isinstance(data, Mapping):
        raise ValueError("strategy payload must be an object")
    return data


def _llm_strategy_generator(request: SearchStrategyRequest) -> SearchStrategy:
    """Generate a search strategy via the production LLM client."""

    prompt = (
        "You are an expert research strategist tasked with "
        "designing focused web search strategies.\n"
        "Analyse the user's intent and produce between 3 and 5 focused web "
        "search queries that maximise authoritative and relevant sources.\n"
        "Consider document types, versioning, source quality, and "
        "content relevance to the task.\n"
        "Respond with a JSON object containing the keys 'queries', "
        "'policies_applied', 'preferred_sources', 'disallowed_sources', and "
        "an optional 'notes'.\n"
        "- 'queries' must be an array of 3-5 strings.\n"
        "- Optional arrays may be empty if not applicable.\n"
        "Do not include any additional text outside the JSON object.\n"
        "\n"
        "Context:\n"
        f"- Tenant: {request.tenant_id}\n"
        f"- Purpose: {request.purpose}\n"
        f"- Quality mode: {request.quality_mode}\n"
        f"- Original query: {request.query}"
    )
    query_hash = sha256(request.query.encode("utf-8")).hexdigest()[:12]
    metadata = {
        "tenant_id": request.tenant_id,
        "case_id": f"collection-search:{request.purpose}:{query_hash}",
        "trace_id": None,
        "prompt_version": "collection_search_strategy_v1",
    }
    try:
        llm_start = time.time()
        response = llm_client.call("analyze", prompt, metadata)
        llm_latency = time.time() - llm_start

        # Track LLM metrics for observability
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        model_id = response.get("model", "gemini-2.5-flash")

        # Calculate cost
        cost_usd = calculate_chat_completion_cost(
            model_id, prompt_tokens, completion_tokens
        )

        # Attach metrics to current span
        update_observation(
            metadata={
                "llm.model": model_id,
                "llm.prompt_tokens": prompt_tokens,
                "llm.completion_tokens": completion_tokens,
                "llm.total_tokens": total_tokens,
                "llm.cost_usd": f"{cost_usd:.6f}",
                "llm.latency_ms": int(llm_latency * 1000),
                "llm.label": "analyze",
            }
        )
    except (LlmClientError, RateLimitError) as exc:  # pragma: no cover
        return _fallback_with_reason(
            request,
            "llm strategy generation failed; using fallback strategy",
            exc,
        )
    except Exception as exc:  # pragma: no cover
        return _fallback_with_reason(
            request,
            "llm strategy generation failed; using fallback strategy",
            exc,
        )

    text = (response.get("text") or "").strip()
    try:
        payload = _extract_strategy_payload(text)
    except ValueError as exc:
        return _fallback_with_reason(
            request,
            "unable to parse LLM strategy payload; using fallback strategy",
            exc,
        )

    try:
        queries = _coerce_query_list(payload.get("queries"))
    except Exception as exc:
        return _fallback_with_reason(
            request,
            "invalid queries in LLM strategy payload; using fallback strategy",
            exc,
        )

    policies = payload.get("policies_applied") or ()
    preferred_sources = payload.get("preferred_sources") or ()
    disallowed_sources = payload.get("disallowed_sources") or ()
    notes = payload.get("notes") if isinstance(payload.get("notes"), str) else None
    if isinstance(notes, str):
        notes = notes.strip() or None

    try:
        return SearchStrategy(
            queries=queries,
            policies_applied=policies,
            preferred_sources=preferred_sources,
            disallowed_sources=disallowed_sources,
            notes=notes,
        )
    except ValidationError as exc:
        return _fallback_with_reason(
            request,
            "structured strategy validation failed; using fallback strategy",
            exc,
        )


class _ProductionHitlGateway:
    """Production HITL gateway (currently no-op - HITL persistence removed)."""

    def present(self, payload: Mapping[str, Any]) -> HitlDecision | None:
        """Return None to indicate pending approval (HITL persistence removed)."""
        # HITL persistence has been removed
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
            with httpx.Client(timeout=10.0) as client:
                response = client.post(crawler_url, json=payload, headers=headers)
            if response.status_code in (200, 202):
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


def build_graph() -> CollectionSearchGraph:
    """Build a production-ready collection search graph."""
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

    return CollectionSearchGraph(
        strategy_generator=_llm_strategy_generator,
        search_worker=search_worker,
        hybrid_executor=_HybridExecutorAdapter(),
        hitl_gateway=_ProductionHitlGateway(),
        ingestion_trigger=_ProductionIngestionTrigger(),
        coverage_verifier=_ProductionCoverageVerifier(),
    )
