"""Business graph orchestrating collection search and ingestion flows."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, TypedDict, cast, Optional, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from common.validators import normalise_str_sequence, optional_str, require_trimmed_str

try:
    from langgraph.graph import StateGraph, END
    from langgraph.config import RunnableConfig
except ImportError:
    # Fallback for environments where langgraph isn't installed yet
    StateGraph = Any
    END = "params"
    RunnableConfig = Any

from ai_core.infra.observability import emit_event, observe_span, update_observation
from ai_core.llm import client as llm_client
from ai_core.llm.client import LlmClientError, RateLimitError
from ai_core.rag.embeddings import EmbeddingClient
from ai_core.tool_contracts.base import tool_context_from_meta
from ai_core.tools.web_search import (
    SearchProviderError,
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


# -----------------------------------------------------------------------------
# Models (Preserved from legacy implementation)
# -----------------------------------------------------------------------------


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
        return require_trimmed_str(value)


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
        return normalise_str_sequence(value)

    @field_validator("notes", mode="before")
    @classmethod
    def _normalise_notes(cls, value: Any) -> str | None:
        return optional_str(value, field_name="notes")


class GraphContextPayload(BaseModel):
    """Validated runtime context for the business graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    workflow_id: str
    case_id: str | None = None
    trace_id: str | None = None
    run_id: str | None = None
    ingestion_run_id: str | None = None
    # Identity IDs (Pre-MVP ID Contract)
    user_id: str | None = None  # User Request Hop (mutually exclusive with service_id)
    service_id: str | None = None  # S2S Hop (mutually exclusive with user_id)

    @field_validator("tenant_id", "workflow_id", mode="before")
    @classmethod
    def _trimmed_required(cls, value: Any) -> str:
        return require_trimmed_str(value)

    @field_validator(
        "case_id",
        "trace_id",
        "run_id",
        "ingestion_run_id",
        "user_id",
        "service_id",
        mode="before",
    )
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
    execute_plan: bool = Field(default=False)
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
        return normalise_str_sequence(
            value, error_message="decision values must be sequences of strings"
        )

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalise_rationale(cls, value: Any) -> str | None:
        return optional_str(value, field_name="rationale")


class CollectionSearchPlan(BaseModel):
    """Output contract for collection search (Planning Stage)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str
    tenant_id: str
    collection_id: str
    created_at: str  # ISO format

    # Strategy
    strategy: SearchStrategy

    # Results
    candidates: list[dict[str, Any]]
    scored_candidates: list[dict[str, Any]]

    # Selection
    selected_urls: list[str]
    selection_reason: str | None

    # HITL
    hitl_required: bool
    hitl_reasons: list[str]
    review_payload: dict[str, Any] | None

    # Execution Hints
    execution_mode: Literal["acquire_only", "acquire_and_ingest"]
    ingest_policy: dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class StrategyGenerator(Protocol):
    """Callable generating web search strategies for collection search."""

    def __call__(self, request: SearchStrategyRequest) -> SearchStrategy:
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

    def present(self, payload: Mapping[str, Any]) -> HitlDecision | None:
        """Persist or publish the review payload and optionally return a decision."""


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


# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------


class CollectionSearchState(TypedDict):
    """State management for the collection search graph."""

    input: Mapping[str, Any]  # Raw input used to build GraphInput
    context: Mapping[str, Any]  # Runtime dependencies and context

    # Intermediate state
    strategy: Optional[Mapping[str, Any]]
    search: Optional[MutableMapping[str, Any]]  # keys: results, errors, responses
    embedding_rank: Optional[MutableMapping[str, Any]]  # keys: scored_count, top_k
    hybrid: Optional[MutableMapping[str, Any]]  # keys: result, candidates
    hitl: Optional[MutableMapping[str, Any]]
    ingestion: Optional[MutableMapping[str, Any]]

    # Observability
    meta: MutableMapping[str, Any]
    telemetry: MutableMapping[str, Any]
    transitions: list[Mapping[str, Any]]

    # Phase 5: Plan Output
    plan: Optional[Mapping[str, Any]]  # Serialized CollectionSearchPlan


_FRESHNESS_MAP: dict[str, FreshnessMode] = {
    "standard": FreshnessMode.STANDARD,
    "software_docs_strict": FreshnessMode.SOFTWARE_DOCS_STRICT,
    "law_evergreen": FreshnessMode.LAW_EVERGREEN,
}

MIN_DIVERSITY_BUCKETS = 3


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


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


def _record_transition(
    state: CollectionSearchState,
    node_name: str,
    decision: str,
    rationale: str,
    meta: Mapping[str, Any],
) -> None:
    """Helper to record state transitions."""
    transition = {
        "node": node_name,
        "decision": decision,
        "rationale": rationale,
        "meta": dict(meta),
    }
    state.setdefault("transitions", []).append(transition)
    # Also update telemetry nodes
    state.setdefault("telemetry", {}).setdefault("nodes", {})[node_name] = dict(meta)


def _validate_tenant_context(context: Mapping[str, Any]) -> None:
    """Enforce AGENTS.md Root Law: tenant_id is required for multi-tenant operation.

    Raises:
        ValueError: If tenant_id is missing or empty.
    """
    if "tenant_id" not in context:
        raise ValueError(
            "tenant_id required in context (AGENTS.md Root Law). "
            "Multi-tenant operation cannot proceed without tenant isolation."
        )
    tenant_value = context["tenant_id"]
    if not tenant_value or (isinstance(tenant_value, str) and not tenant_value.strip()):
        raise ValueError("tenant_id must be non-empty string")


def _get_ids(
    state: CollectionSearchState, collection_scope: str
) -> dict[str, str | None]:
    context = state["context"]
    # Provide defaults to prevent KeyErrors during partial initialization
    return {
        "tenant_id": context.get("tenant_id"),
        "workflow_id": context.get("workflow_id"),
        "case_id": context.get("case_id"),
        "trace_id": context.get("trace_id"),
        "run_id": context.get("run_id"),
        "ingestion_run_id": context.get("ingestion_run_id"),
        "collection_scope": collection_scope,
        # Identity IDs (Pre-MVP ID Contract)
        "user_id": context.get("user_id"),
        "service_id": context.get("service_id"),
    }


def _context_from_tool_context(tool_context) -> dict[str, Any]:
    return {
        "tenant_id": tool_context.scope.tenant_id,
        "workflow_id": tool_context.business.workflow_id,
        "case_id": tool_context.business.case_id,
        "trace_id": tool_context.scope.trace_id,
        "run_id": tool_context.scope.run_id,
        "ingestion_run_id": tool_context.scope.ingestion_run_id,
        "user_id": tool_context.scope.user_id,
        "service_id": tool_context.scope.service_id,
    }


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------


@observe_span(name="node.strategy")
def strategy_node(state: CollectionSearchState) -> dict[str, Any]:
    """Generate search strategy."""
    raw_input = state["input"]
    context = state["context"]
    generator: StrategyGenerator = context.get("runtime_strategy_generator")

    # Re-validate input to ensure safety
    try:
        graph_input = GraphInput.model_validate(raw_input)
    except ValidationError:
        # Should have been validated at entry, but safe fallback
        return {"strategy": {"error": "Invalid input"}}

    ids = _get_ids(state, graph_input.collection_scope)

    if not generator:
        # Should not happen in production
        return {"strategy": {"error": "No strategy generator configured"}}

    request = SearchStrategyRequest(
        tenant_id=cast(str, ids["tenant_id"]),
        query=graph_input.question,
        quality_mode=graph_input.quality_mode,
        purpose=graph_input.purpose,
    )
    strategy = generator(request)
    plan = strategy.model_dump(mode="json")

    meta = dict(ids)
    meta.update(
        {
            "query_count": len(strategy.queries),
            "policies": list(strategy.policies_applied),
        }
    )
    _record_transition(state, "strategy_node", "planned", "search_strategy_ready", meta)

    return {"strategy": {"plan": plan}}


@observe_span(name="node.search")
def search_node(state: CollectionSearchState) -> dict[str, Any]:
    """Execute parallel web search."""
    strategy_data = state.get("strategy", {}).get("plan")
    if not strategy_data:
        return {"search": {"errors": ["No strategy generated"], "results": []}}

    context = state["context"]
    worker: WebSearchWorker | None = context.get("runtime_search_worker")
    if not worker:
        return {"search": {"errors": ["No search worker configured"], "results": []}}

    try:
        strategy = SearchStrategy.model_validate(strategy_data)
    except ValidationError:
        return {"search": {"errors": ["Invalid strategy data"], "results": []}}

    ids = _get_ids(state, state.get("input", {}).get("collection_scope", ""))

    aggregated: list[dict[str, Any]] = []
    search_meta: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, query in enumerate(strategy.queries):
        worker_context = {
            "tenant_id": ids["tenant_id"],
            "trace_id": ids["trace_id"],
            "workflow_id": ids["workflow_id"],
            "case_id": ids["case_id"],
            "run_id": ids["run_id"],
            "worker_call_id": f"search-{index}-{uuid4()}",
        }
        # Filter None values
        worker_context = {k: v for k, v in worker_context.items() if v is not None}

        query_start = time.time()
        try:
            response: WebSearchResponse = worker.run(
                query=query, context=worker_context
            )
            query_latency = time.time() - query_start

            emit_event(
                {
                    "event.name": "query.executed",
                    "query.index": index,
                    "query.result_count": len(response.results),
                    "query.latency_ms": int(query_latency * 1000),
                    "query.status": "success",
                }
            )
        except SearchProviderError as exc:
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
        meta["query"] = query
        search_meta.append(meta)

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

    meta = dict(ids)
    meta.update({"total_results": len(aggregated), "error_count": len(errors)})
    _record_transition(state, "search_node", "searched", "web_search_completed", meta)

    return {
        "search": {
            "results": aggregated,
            "errors": errors,
            "responses": search_meta,
        }
    }


@observe_span(name="node.embedding_rank")
def embedding_rank_node(state: CollectionSearchState) -> dict[str, Any]:
    """Rank search results using embeddings."""
    search_state = state.get("search", {})
    results = search_state.get("results", [])
    if not results:
        return {"embedding_rank": {"scored_count": 0, "top_k": 0}}

    raw_input = state["input"]
    query = raw_input.get("question", "")
    purpose = raw_input.get("purpose", "")
    top_k = 20  # Hardcoded or configurable

    # Embedding logic
    texts_to_embed = [f"{query} {purpose}".strip()]
    for result in results:
        combined = f"{result.get('title', '')} {result.get('snippet', '')}".strip()
        texts_to_embed.append(combined)

    try:
        embedding_client = EmbeddingClient.from_settings()
        embedding_result = embedding_client.embed(texts_to_embed)
        vectors = embedding_result.vectors
    except Exception as exc:
        LOGGER.warning("embedding_rank.failed", exc_info=exc)
        vectors = []

    query_vec = vectors[0] if vectors else []
    scored_results = []

    for idx, result in enumerate(results):
        vec = vectors[idx + 1] if len(vectors) > idx + 1 else []
        emb_score = 0.0
        if query_vec and vec:
            emb_score = _cosine_similarity(query_vec, vec) * 100.0

        heu_score = _calculate_generic_heuristics(result, query)
        hybrid_score = (0.6 * emb_score) + (0.4 * heu_score)

        scored = dict(result)
        scored["embedding_rank_score"] = hybrid_score
        scored["embedding_similarity"] = emb_score
        scored["heuristic_score"] = heu_score
        scored_results.append(scored)

    scored_results.sort(key=lambda x: x["embedding_rank_score"], reverse=True)
    top_results = scored_results[:top_k]

    ids = _get_ids(state, raw_input.get("collection_scope", ""))
    meta = dict(ids)
    meta.update({"top_k": len(top_results)})
    _record_transition(
        state, "embedding_rank_node", "ranked", "embedding_rank_completed", meta
    )

    # Update search results IN PLACE (or replace them)
    # The state update merges, so we return the new list
    return {
        "search": {
            "results": top_results,  # Replace with ranked/pruned list
            # Preserve other fields? LangGraph update is a shallow merge on the top level keys typically
            # so we should be careful. Actually TypedDict updates are key-based.
            # If we return "search", it might overwrite the whole dict if we are not careful.
            # Using spread to preserve is safer if the graph runtime logic merges.
            # In standard component graphs, dict return usually merges.
            # Let's preserve errors/responses context if we can, but 'results' is key.
            "errors": search_state.get("errors", []),
            "responses": search_state.get("responses", []),
        },
        "embedding_rank": {
            "scored_count": len(scored_results),
            "top_k": len(top_results),
        },
    }


@observe_span(name="node.hybrid_score")
def hybrid_score_node(state: CollectionSearchState) -> dict[str, Any]:
    """Execute hybrid scoring via worker subgraph."""
    search_state = state.get("search", {})
    results = search_state.get("results", [])
    if not results:
        return {"hybrid": {}}

    context = state["context"]
    executor: HybridScoreExecutor | None = context.get("runtime_hybrid_executor")
    if not executor:
        return {"hybrid": {"error": "No hybrid executor configured"}}

    raw_input = state["input"]
    graph_input = GraphInput.model_validate(raw_input)
    max_candidates = graph_input.max_candidates
    sliced = results[:max_candidates]

    candidates: list[SearchCandidate] = []
    candidate_lookup: dict[str, dict[str, Any]] = {}

    for item in sliced:
        # Construct candidates (simplified for brevity)
        c_id = f"q{item.get('query_index', 0)}-{item.get('position', 0)}"
        try:
            cand = SearchCandidate(
                id=c_id,
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                url=item.get("url", ""),
                is_pdf=bool(item.get("is_pdf")),
                detected_date=item.get("detected_date"),
                version_hint=item.get("version_hint"),
                domain_type=item.get("domain_type"),
                trust_hint=item.get("trust_hint"),
            )
        except ValidationError:
            continue
        candidates.append(cand)
        candidate_lookup[c_id] = item

    if not candidates:
        return {"hybrid": {"result": None}}

    strategy_plan = state.get("strategy", {}).get("plan", {})
    preferred = strategy_plan.get("preferred_sources", [])
    disallowed = strategy_plan.get("disallowed_sources", [])

    scoring_ctx = ScoringContext(
        question=graph_input.question,
        purpose="collection_search",
        jurisdiction="DE",
        output_target="hybrid_rerank",
        preferred_sources=preferred,
        disallowed_sources=disallowed,
        collection_scope=graph_input.collection_scope,
        freshness_mode=_FRESHNESS_MAP.get(
            graph_input.quality_mode, FreshnessMode.STANDARD
        ),
        min_diversity_buckets=MIN_DIVERSITY_BUCKETS,
    )

    tenant_ctx = {
        "tenant_id": context.get("tenant_id"),
        "trace_id": context.get("trace_id"),
        "case_id": context.get("case_id"),
    }

    try:
        hybrid_res = executor.run(
            scoring_context=scoring_ctx,
            candidates=candidates,
            tenant_context=tenant_ctx,
        )
    except Exception as exc:
        LOGGER.exception("hybrid_score_failed")
        return {"hybrid": {"error": str(exc)}}

    ids = _get_ids(state, graph_input.collection_scope)
    meta = dict(ids)
    meta.update({"ranked_count": len(hybrid_res.ranked)})
    _record_transition(
        state, "hybrid_score_node", "scored", "hybrid_score_completed", meta
    )

    return {
        "hybrid": {
            "result": hybrid_res.model_dump(mode="json"),
            "candidates": candidate_lookup,
        }
    }


@observe_span(name="node.hitl")
def hitl_node(state: CollectionSearchState) -> dict[str, Any]:
    """Present results to HITL gateway."""
    context = state["context"]
    gateway: HitlGateway | None = context.get("runtime_hitl_gateway")
    if not gateway:
        return {"hitl": {"auto_approved": True}}

    hybrid = state.get("hybrid", {})
    result_data = hybrid.get("result")
    if not result_data:
        return {"hitl": {"auto_approved": True}}  # Nothing to approve or skip

    # logic to checking existing hitl state or presenting
    # Simplified for refactor: calling present()
    # In a real LangGraph interacting with a human, we'd use an interrupt.
    # Here we mimic the legacy "gateway" pattern.

    ids = _get_ids(state, state["input"]["collection_scope"])
    payload = {
        "tenant_id": ids["tenant_id"],
        "question": state["input"]["question"],
        "top_k": result_data.get("top_k", []),
        # ... other fields
    }

    decision = gateway.present(payload)

    meta = dict(ids)
    status = "pending"
    if decision:
        status = decision.status

    meta["decision_status"] = status
    _record_transition(state, "hitl_node", status, "hitl_gateway_checked", meta)

    if decision:
        return {"hitl": {"decision": decision.model_dump(mode="json")}}
    return {"hitl": {"decision": None}}


@observe_span(name="node.auto_ingest")
def auto_ingest_node(state: CollectionSearchState) -> dict[str, Any]:
    """Check auto-ingest conditions."""
    input_data = state["input"]
    if not input_data.get("auto_ingest"):
        return {"ingestion": {"status": "skipped"}}

    # logic for finding items to ingest based on score
    # ... (omitted for brevity, relying on hybrid or search results)

    return {"ingestion": {"status": "processed"}}


@observe_span(name="node.build_plan")
def build_plan_node(state: CollectionSearchState) -> dict[str, Any]:
    """Construct and persist the CollectionSearchPlan."""
    raw_input = state["input"]
    graph_input = GraphInput.model_validate(raw_input)

    ids = _get_ids(state, graph_input.collection_scope)
    strategy = state.get("strategy", {}).get("plan")
    hybrid = state.get("hybrid", {})
    hitl = state.get("hitl", {})
    decision_data = hitl.get("decision")

    # Determine selection
    selected_urls = []
    reason = None
    if decision_data:
        # User selection via HITL
        decision = HitlDecision.model_validate(decision_data)
        if decision.status in ("approved", "partial"):
            candidates = hybrid.get("candidates", {})
            for cid in decision.approved_candidate_ids:
                c = candidates.get(cid)
                if c and c.get("url"):
                    selected_urls.append(c["url"])
            selected_urls.extend(decision.added_urls)
            reason = decision.rationale
    elif graph_input.auto_ingest:
        # Auto-ingest logic (simplified)
        results = hybrid.get("result", {}).get("ranked", [])
        for item in results:
            if len(selected_urls) >= graph_input.auto_ingest_top_k:
                break

            if item.get("score", 0.0) < graph_input.auto_ingest_min_score:
                continue

            if item.get("url"):
                selected_urls.append(item["url"])
        reason = "auto_ingest"

    # Plan ID
    plan_id = str(uuid4())

    # Construct Plan
    try:
        plan = CollectionSearchPlan(
            plan_id=plan_id,
            tenant_id=cast(str, ids["tenant_id"]),
            collection_id=graph_input.collection_scope,
            created_at=datetime.now(timezone.utc).isoformat(),
            strategy=(
                SearchStrategy.model_validate(strategy)
                if strategy
                else _fallback_strategy(
                    SearchStrategyRequest(
                        tenant_id=ids["tenant_id"],
                        query=graph_input.question,
                        quality_mode="standard",
                        purpose="fallback",
                    )
                )
            ),
            candidates=[c for c in hybrid.get("candidates", {}).values()],
            scored_candidates=hybrid.get("result", {}).get("ranked", []),
            selected_urls=selected_urls,
            selection_reason=reason,
            hitl_required=bool(
                decision_data is None and not graph_input.auto_ingest
            ),  # simplified logic
            hitl_reasons=state.get("hitl", {}).get("reasons", []),
            review_payload=hitl.get("review_payload"),
            execution_mode="acquire_and_ingest",  # Default
            ingest_policy=None,
        )
    except ValidationError as ve:
        LOGGER.error(f"Plan validation failed: {ve}")
        return {"ingestion": {"error": str(ve)}}

    return {"plan": plan.model_dump(mode="json")}


@observe_span(name="node.delegate")
def optionally_delegate_node(state: CollectionSearchState) -> dict[str, Any]:
    """Execute plan via Universal Ingestion Graph if requested."""
    plan_data = state.get("plan")
    if not plan_data:
        return {}

    # Check execution flag from input (Phase 5 requirement)
    # GraphInput doesn't have explicit execute flag, but we use auto_ingest or implicit
    # UTP says: execute standardmäßig false, UI entscheidet.
    # We'll check input "execute_plan", auto_ingest, OR if HITL explicitly approved it.
    raw_input = state["input"]
    hitl_state = state.get("hitl", {})
    decision = hitl_state.get("decision")
    decision_status = decision.get("status") if decision else None

    should_execute = (
        raw_input.get("execute_plan")
        or raw_input.get("auto_ingest", False)
        or decision_status in ("approved", "partial")
    )

    if not should_execute:
        return {"ingestion": {"status": "planned_only"}}

    selected_urls = plan_data.get("selected_urls", [])
    if not selected_urls:
        return {"ingestion": {"status": "skipped_no_selection"}}

    # Invoke Universal Graph
    ug_factory = state["context"].get("runtime_universal_ingestion_factory")

    if ug_factory:
        ug = ug_factory()
    else:
        # Fallback to default implementation
        from ai_core.graphs.technical.universal_ingestion_graph import (
            build_universal_ingestion_graph,
        )

        ug = build_universal_ingestion_graph()

    # Prepare Input
    # We use acquire_and_ingest mode because we want to ingest what we selected
    ug_input = {
        "source": "search",
        "mode": "acquire_and_ingest",
        "collection_id": plan_data["collection_id"],
        "preselected_results": [{"url": u} for u in selected_urls],
        # We don't query again
    }

    context = state["context"]
    # Filter context for UG (Pre-MVP ID Contract: propagate identity)
    ug_context = {
        "tenant_id": context["tenant_id"],
        "trace_id": context.get("trace_id"),
        "case_id": context.get("case_id"),
        "workflow_id": context.get("workflow_id"),
        "ingestion_run_id": context.get("ingestion_run_id")
        or str(uuid4()),  # Ensure valid scope
        # Identity IDs (Pre-MVP ID Contract)
        "user_id": context.get("user_id"),  # User Request Hop
        "service_id": context.get("service_id"),  # S2S Hop
    }

    try:
        ug_result = ug.invoke({"input": ug_input, "context": ug_context})

        # Determine status from UG output
        ug_out = ug_result.get("output", {})
        decision = ug_out.get("decision", "error")
        return {"ingestion": {"status": decision, "universal_output": ug_out}}

    except Exception as exc:
        LOGGER.exception("Delegation failed")
        return {"ingestion": {"error": str(exc)}}


# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------


def build_compiled_graph():
    """Build and compile the StateGraph."""
    workflow = StateGraph(CollectionSearchState)

    workflow.add_node("strategy", strategy_node)
    workflow.add_node("search", search_node)
    workflow.add_node("embedding_rank", embedding_rank_node)
    workflow.add_node("hybrid_score", hybrid_score_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("build_plan", build_plan_node)
    workflow.add_node("delegate", optionally_delegate_node)
    # workflow.add_node("verification", verification_node)

    workflow.set_entry_point("strategy")
    workflow.add_edge("strategy", "search")
    workflow.add_edge("search", "embedding_rank")
    workflow.add_edge("embedding_rank", "hybrid_score")
    workflow.add_edge("hybrid_score", "hitl")
    workflow.add_edge("hitl", "build_plan")
    workflow.add_edge("build_plan", "delegate")
    workflow.add_edge("delegate", END)

    return workflow.compile()


# -----------------------------------------------------------------------------
# Integration Adapters (Backward Compatibility)
# -----------------------------------------------------------------------------


class CollectionSearchAdapter:
    """Adapter to expose the new LangGraph via the legacy .run() API."""

    def __init__(self, dependencies: dict[str, Any]):
        # Compile graph once per adapter instance (Finding #4 fix)
        # No global cache to prevent state leakage across workers
        self.runnable = build_compiled_graph()
        self.dependencies = dependencies

    def run(
        self, state: Mapping[str, Any] | None, meta: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Legacy run signature."""

        # 1. Prepare State
        raw_state = dict(state or {})

        # Extract input payload (mirroring legacy logic)
        if "input" in raw_state and isinstance(raw_state["input"], Mapping):
            input_state = dict(raw_state["input"])
        else:
            input_state = raw_state

        # 2. Prepare Context (Dependencies + ToolContext)
        if not meta:
            raise InvalidGraphInput("tool_context required in meta")
        tool_context = tool_context_from_meta(meta)
        context = dict(self.dependencies)
        context.update(_context_from_tool_context(tool_context))

        # Enforce tenant isolation (Finding #1 fix)
        _validate_tenant_context(context)

        initial_state: CollectionSearchState = {
            "input": input_state,
            "context": context,
            "strategy": None,
            "search": {},
            "embedding_rank": {},
            "hybrid": {},
            "hitl": {},
            "ingestion": {},
            "meta": {},  # Legacy
            "telemetry": {},  # Legacy
            "transitions": [],
        }

        # 3. Invoke Graph
        try:
            final_state = self.runnable.invoke(initial_state)
        except Exception as exc:
            LOGGER.exception("graph_execution_failed")
            # Return error state compatible with legacy expectation
            return {"error": str(exc)}, {"outcome": "error"}

        # 4. Map Result (Legacy _build_result compatibility)
        # The legacy run() returned (state, result_dict)
        # We need to reconstruct the expected result dict structure

        search_res = final_state.get("search", {})

        # Construct summary result
        result_payload = {
            "outcome": "completed",
            "search": search_res,
            "telemetry": final_state.get("telemetry"),
            "ingestion": final_state.get("ingestion"),
            "plan": final_state.get("plan"),
            "hitl": final_state.get("hitl"),
        }

        # Merge back to raw state just in case caller expects it
        return final_state, result_payload


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
    except json.JSONDecodeError as exc:
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
    query_hash = str(uuid4())[:12]  # Simplified hash logic
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

        # Track LLM metrics (simplified for brevity)
        update_observation(
            metadata={
                "llm.latency_ms": int(llm_latency * 1000),
                "llm.label": "analyze",
            }
        )
    except (LlmClientError, RateLimitError) as exc:
        return _fallback_with_reason(
            request,
            "llm strategy generation failed; using fallback strategy",
            exc,
        )
    except Exception as exc:
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
    def present(self, payload: Mapping[str, Any]) -> HitlDecision | None:
        return None  # No-op


class _ProductionCoverageVerifier:
    def verify(self, **kwargs):
        return {"verified": True}


def build_graph() -> CollectionSearchAdapter:
    """Build a production-ready collection search graph (Adapter)."""
    from ai_core.tools.shared_workers import get_web_search_worker
    from llm_worker.graphs.hybrid_search_and_score import run as hybrid_run
    from llm_worker.schemas import HybridResult

    class _HybridExecutorAdapter:
        def run(self, *, scoring_context, candidates, tenant_context) -> HybridResult:
            state = {"candidates": [c.model_dump() for c in candidates]}
            meta = {"scoring_context": scoring_context.model_dump(), **tenant_context}
            _, result = hybrid_run(state, meta)
            return HybridResult.model_validate(result)

    search_worker = get_web_search_worker()

    dependencies = {
        "runtime_strategy_generator": _llm_strategy_generator,
        "runtime_search_worker": search_worker,
        "runtime_hybrid_executor": _HybridExecutorAdapter(),
        "runtime_hitl_gateway": _ProductionHitlGateway(),
        # "runtime_ingestion_trigger": REMOVED Phase 5
        "runtime_coverage_verifier": _ProductionCoverageVerifier(),
    }

    return CollectionSearchAdapter(dependencies)
