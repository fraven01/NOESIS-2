"""Business graph orchestrating collection search and ingestion flows."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Protocol, TypedDict, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from common.validators import normalise_str_sequence, optional_str
from ai_core.contracts.plans import (
    Evidence,
    Gate,
    ImplementationPlan,
    PlanScope,
    Slot,
    Task,
    derive_plan_key,
)
from ai_core.services.collection_search.auto_ingest import select_auto_ingest_urls
from ai_core.services.collection_search.hitl import build_hitl_payload
from ai_core.services.collection_search.scoring import (
    calculate_generic_heuristics,
    cosine_similarity,
)
from ai_core.services.collection_search.strategy import (
    SearchStrategy,
    SearchStrategyRequest,
    fallback_strategy,
    llm_strategy_generator,
)

from langgraph.graph import StateGraph, END

from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.infra.observability import emit_event, observe_span
from ai_core.rag.embeddings import EmbeddingClient
from ai_core.tool_contracts import ToolContext
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
from ai_core.schemas import CrawlerRunRequest, CrawlerOriginConfig

LOGGER = logging.getLogger(__name__)


class InvalidGraphInput(ValueError):
    """Raised when the graph input payload cannot be validated."""


# -----------------------------------------------------------------------------
# Models (Preserved from legacy implementation)
# -----------------------------------------------------------------------------


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

    # Embedding ranking configuration
    embedding_top_k: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of top results to keep after embedding ranking",
    )
    embedding_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for embedding similarity score (heuristic weight = 1 - this)",
    )

    @field_validator("quality_mode", mode="before")
    @classmethod
    def _normalise_quality_mode(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("quality_mode must be a string")
        candidate = value.strip().lower()
        if not candidate:
            return "standard"
        return candidate


class StrategyState(BaseModel):
    """State container for search strategy generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: SearchStrategy | None = None
    error: str | None = None


class SearchResultPayload(BaseModel):
    """Typed search result with query metadata and scoring fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str | None = None
    title: str
    snippet: str
    source: str
    score: float | None = None
    is_pdf: bool = False
    query: str
    query_index: int
    position: int
    embedding_rank_score: float | None = None
    embedding_similarity: float | None = None
    heuristic_score: float | None = None
    detected_date: datetime | None = None
    version_hint: str | None = None
    domain_type: str | None = None
    trust_hint: str | None = None


class SearchError(BaseModel):
    """Structured search error payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str | None = None
    error: str | None = None
    message: str | None = None


class SearchResponseMeta(BaseModel):
    """Response metadata per query (provider details + query)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    query: str | None = None


class SearchResultsPayload(BaseModel):
    """Boundary output for search execution results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: list[SearchResultPayload] = Field(default_factory=list)
    errors: list[SearchError] = Field(default_factory=list)
    responses: list[SearchResponseMeta] = Field(default_factory=list)
    embedding_failed: bool | None = None


class HybridState(BaseModel):
    """State container for hybrid scoring results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result: HybridResult | None = None
    candidates: list[SearchCandidate] = Field(default_factory=list)
    embedding_degraded: bool | None = None
    error: str | None = None


COLLECTION_SEARCH_SCHEMA_ID = "noesis.graphs.collection_search"
COLLECTION_SEARCH_IO_VERSION = GraphIOVersion(major=1, minor=0, patch=0)
COLLECTION_SEARCH_IO_VERSION_STRING = COLLECTION_SEARCH_IO_VERSION.as_string()


class CollectionSearchGraphRequest(BaseModel):
    """Boundary input model for the collection search graph.

    BREAKING CHANGE: schema_id and schema_version are required fields (no defaults).
    All callers must explicitly provide these values and include tool_context.
    """

    schema_id: Literal[COLLECTION_SEARCH_SCHEMA_ID]
    schema_version: str
    input: GraphInput
    tool_context: ToolContext | None = None
    runtime: dict[str, Any] | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("schema_version must be a string")
        parts = value.strip().split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("schema_version must be in MAJOR.MINOR.PATCH form")
        major = int(parts[0])
        if major != COLLECTION_SEARCH_IO_VERSION.major:
            raise ValueError("schema_version major must match")
        return value.strip()


class CollectionSearchGraphOutput(BaseModel):
    """Boundary output model for the collection search graph."""

    schema_id: Literal[COLLECTION_SEARCH_SCHEMA_ID] = COLLECTION_SEARCH_SCHEMA_ID
    schema_version: Literal[COLLECTION_SEARCH_IO_VERSION_STRING] = (
        COLLECTION_SEARCH_IO_VERSION_STRING
    )
    outcome: str
    search: SearchResultsPayload | None
    telemetry: Mapping[str, Any] | None
    ingestion: Mapping[str, Any] | None
    plan: Mapping[str, Any] | None
    hitl: Mapping[str, Any] | None
    error: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


COLLECTION_SEARCH_IO = GraphIOSpec(
    schema_id=COLLECTION_SEARCH_SCHEMA_ID,
    version=COLLECTION_SEARCH_IO_VERSION,
    input_model=CollectionSearchGraphRequest,
    output_model=CollectionSearchGraphOutput,
)


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


class CrawlerManagerProtocol(Protocol):
    """Protocol for the crawler manager service."""

    def dispatch_crawl_request(
        self, request: CrawlerRunRequest, meta: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Dispatch a crawler run."""


# -----------------------------------------------------------------------------
# State Definition & Reducers
# -----------------------------------------------------------------------------


def _merge_dict(
    left: MutableMapping[str, Any], right: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Shallow-merge reducer for dict state fields.

    Merges right into left, preserving existing keys unless overwritten.
    This prevents accidental data loss when updating only some dict keys.
    """
    if left is None:
        return dict(right) if right else {}
    if right is None:
        return left
    merged = dict(left)
    merged.update(right)
    return merged


def _append_list(left: list[Any], right: list[Any]) -> list[Any]:
    """Append reducer for list state fields."""
    if left is None:
        return list(right) if right else []
    if right is None:
        return left
    return left + list(right)


class CollectionSearchState(TypedDict):
    """State management for the collection search graph.

    Uses Annotated types with reducer functions for safe partial updates.
    This prevents accidental overwrites when nodes only update specific keys.
    """

    input: GraphInput
    tool_context: ToolContext
    runtime: Mapping[str, Any]  # Runtime dependencies

    # Intermediate state with merge reducers
    strategy: StrategyState | None
    search: Annotated[
        MutableMapping[str, Any], _merge_dict
    ]  # keys: results, errors, responses, embedding_failed
    embedding_rank: Annotated[
        MutableMapping[str, Any], _merge_dict
    ]  # keys: scored_count, top_k, failed
    hybrid: HybridState | None
    hitl: Annotated[MutableMapping[str, Any], _merge_dict]
    ingestion: Annotated[MutableMapping[str, Any], _merge_dict]

    # Observability with merge/append reducers
    meta: Annotated[MutableMapping[str, Any], _merge_dict]
    telemetry: Annotated[MutableMapping[str, Any], _merge_dict]
    transitions: Annotated[list[Mapping[str, Any]], _append_list]

    # Phase 5: Plan Output
    plan: ImplementationPlan | None


_FRESHNESS_MAP: dict[str, FreshnessMode] = {
    "standard": FreshnessMode.STANDARD,
    "software_docs_strict": FreshnessMode.SOFTWARE_DOCS_STRICT,
    "law_evergreen": FreshnessMode.LAW_EVERGREEN,
}
_EMBEDDING_WEIGHT_PROFILES: dict[str, float] = {
    "software_docs_strict": 0.7,
    "law_evergreen": 0.5,
}

MIN_DIVERSITY_BUCKETS = 3
_DEFAULT_SEARCH_TIMEOUT_S = 30.0


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


def _resolve_search_timeout_s() -> float:
    default = _DEFAULT_SEARCH_TIMEOUT_S
    try:
        from django.conf import settings
    except Exception:
        return default
    value = getattr(settings, "SEARCH_WORKER_TIMEOUT_SECONDS", default)
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return default
    if timeout <= 0:
        return default
    return timeout


def _resolve_graph_timeout_s(runtime: Mapping[str, Any]) -> float | None:
    runtime_value = runtime.get("graph_timeout_s")
    if runtime_value is not None:
        try:
            timeout = float(runtime_value)
        except (TypeError, ValueError):
            timeout = None
    else:
        timeout = None
    if timeout is None:
        try:
            from django.conf import settings
        except Exception:
            return None
        timeout = getattr(settings, "GRAPH_COLLECTION_SEARCH_TIMEOUT_S", None)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            return None
    if timeout <= 0:
        return None
    return timeout


def _timeout_payload(timeout_s: float | None) -> CollectionSearchGraphOutput:
    telemetry: dict[str, Any] | None = None
    if timeout_s is not None:
        telemetry = {"graph_timeout_s": timeout_s}
    return CollectionSearchGraphOutput(
        outcome="error",
        search=None,
        telemetry=telemetry,
        ingestion=None,
        plan=None,
        hitl=None,
        error="graph_timeout",
    )


async def _ainvoke_with_timeout(
    runnable: Any,
    state: CollectionSearchState,
    timeout_s: float,
) -> CollectionSearchState:
    return await asyncio.wait_for(runnable.ainvoke(state), timeout=timeout_s)


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


def _resolve_embedding_weight(graph_input: GraphInput) -> tuple[float, str]:
    quality_mode = graph_input.quality_mode
    profile_weight = _EMBEDDING_WEIGHT_PROFILES.get(quality_mode)
    if profile_weight is None:
        return graph_input.embedding_weight, "input"
    return profile_weight, f"profile:{quality_mode}"


def _build_embedding_texts(
    query: str, purpose: str, results: Sequence[SearchResultPayload]
) -> list[str]:
    """Build embedding inputs without diluting the query with long purpose text."""
    texts = [query.strip()]
    if purpose.strip():
        texts.append(purpose.strip())
    for result in results:
        combined = f"{result.title} {result.snippet}".strip()
        texts.append(combined)
    return texts


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


def _validate_tenant_context(tool_context: ToolContext) -> None:
    """Enforce AGENTS.md Root Law: tenant_id is required for multi-tenant operation."""
    tenant_value = tool_context.scope.tenant_id
    if not tenant_value or (isinstance(tenant_value, str) and not tenant_value.strip()):
        raise ValueError("tenant_id must be non-empty string")


def _get_ids(tool_context: ToolContext, collection_scope: str) -> dict[str, str | None]:
    scope = tool_context.scope
    business = tool_context.business
    return {
        "tenant_id": scope.tenant_id,
        "workflow_id": business.workflow_id,
        "case_id": business.case_id,
        "trace_id": scope.trace_id,
        "run_id": scope.run_id,
        "ingestion_run_id": scope.ingestion_run_id,
        "collection_scope": collection_scope,
        # Identity IDs (Pre-MVP ID Contract)
        "user_id": scope.user_id,
        "service_id": scope.service_id,
    }


def _resolve_plan_profile(tool_context: ToolContext) -> tuple[str | None, str | None]:
    metadata = tool_context.metadata
    profile_id = metadata.get("framework_profile_id")
    profile_version = metadata.get("framework_profile_version")
    if profile_id and profile_version:
        profile_version = None
    if profile_id is None and profile_version is None:
        profile_version = "v0"
    return (
        str(profile_id) if profile_id else None,
        str(profile_version) if profile_version else None,
    )


def _resolve_gremium_identifier(
    tool_context: ToolContext, graph_input: GraphInput
) -> str:
    value = tool_context.metadata.get("gremium_identifier")
    if value:
        return str(value)
    return graph_input.collection_scope


def _extract_slot_value(plan: ImplementationPlan, key: str) -> Any:
    for slot in plan.slots:
        if slot.key == key:
            return slot.value
    return None


def _update_ingestion_task(
    plan: ImplementationPlan, task_info: Mapping[str, Any]
) -> ImplementationPlan:
    run_id = task_info.get("run_id") or task_info.get("task_id")
    new_evidence = []
    if run_id:
        new_evidence = [
            Evidence(
                ref_type="object_store",
                ref_id=str(run_id),
                summary="crawler_run",
                metadata={"source": "crawler"},
            )
        ]

    updated_tasks = []
    for task in plan.tasks:
        if task.key == "execute_ingestion":
            outputs = list(task.outputs)
            outputs.extend(new_evidence)
            updated_tasks.append(
                task.model_copy(update={"status": "completed", "outputs": outputs})
            )
        else:
            updated_tasks.append(task)

    metadata = dict(plan.metadata)
    metadata["ingestion"] = {"task_info": dict(task_info)}
    return plan.model_copy(
        update={
            "tasks": updated_tasks,
            "evidence": list(plan.evidence) + new_evidence,
            "metadata": metadata,
        }
    )


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------


@observe_span(name="node.strategy")
def strategy_node(state: CollectionSearchState) -> dict[str, Any]:
    """Generate search strategy."""
    raw_input = state["input"]
    tool_context = state["tool_context"]
    runtime = state["runtime"]
    generator: StrategyGenerator = runtime.get("runtime_strategy_generator")

    # Re-validate input to ensure safety
    graph_input = raw_input
    if not isinstance(graph_input, GraphInput):
        try:
            graph_input = GraphInput.model_validate(raw_input)
        except ValidationError:
            # Should have been validated at entry, but safe fallback
            return {"strategy": StrategyState(error="Invalid input")}

    ids = _get_ids(tool_context, graph_input.collection_scope)

    if not generator:
        # Should not happen in production
        return {"strategy": StrategyState(error="No strategy generator configured")}

    request = SearchStrategyRequest(
        tenant_id=cast(str, ids["tenant_id"]),
        query=graph_input.question,
        quality_mode=graph_input.quality_mode,
        purpose=graph_input.purpose,
    )
    strategy = generator(request)

    meta = dict(ids)
    meta.update(
        {
            "query_count": len(strategy.queries),
            "policies": list(strategy.policies_applied),
        }
    )
    _record_transition(state, "strategy_node", "planned", "search_strategy_ready", meta)

    return {"strategy": StrategyState(plan=strategy)}


def _execute_single_search(
    worker: WebSearchWorker,
    query: str,
    index: int,
    tool_context: ToolContext,
) -> tuple[list[SearchResultPayload], SearchResponseMeta | None, SearchError | None]:
    """Execute a single search query and return (results, meta, error).

    Args:
        worker: WebSearchWorker instance
        query: Search query string
        index: Query index for tracking
        tool_context: ToolContext instance (not dict!) for worker

    Returns:
        Tuple of (results_list, search_meta, error_dict).
        Only one of search_meta or error_dict will be non-None.
    """
    query_start = time.time()
    try:
        response: WebSearchResponse = worker.run(query=query, context=tool_context)
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

        outcome: ToolOutcome = response.outcome
        meta_payload = dict(outcome.meta)
        meta_payload["query"] = query
        meta = SearchResponseMeta.model_validate(meta_payload)

        if outcome.decision == "ok":
            results: list[SearchResultPayload] = []
            for position, result in enumerate(response.results):
                results.append(
                    SearchResultPayload(
                        url=str(result.url) if result.url else None,
                        title=result.title,
                        snippet=result.snippet,
                        source=result.source,
                        score=result.score,
                        is_pdf=result.is_pdf,
                        query=query,
                        query_index=index,
                        position=position,
                    )
                )
            return results, meta, None
        else:
            meta_error = None
            if meta.model_extra:
                meta_error = meta.model_extra.get("error")
            error = SearchError(
                query=query,
                error=meta_error,
                message=outcome.rationale,
            )
            return [], meta, error

    except SearchProviderError as exc:
        error = SearchError(
            query=query,
            error=type(exc).__name__,
            message=str(exc),
        )
        return [], None, error


async def _execute_parallel_searches(
    worker: WebSearchWorker,
    queries: list[str],
    tool_context: ToolContext,
    total_timeout_s: float | None = None,
) -> tuple[
    list[SearchResultPayload],
    list[SearchResponseMeta],
    list[SearchError],
]:
    """Execute multiple search queries in parallel using asyncio.

    Args:
        worker: WebSearchWorker instance
        queries: List of search queries
        tool_context: Base ToolContext to derive per-query contexts from

    Returns:
        Tuple of (aggregated_results, search_meta, errors).
    """
    loop = asyncio.get_running_loop()

    async def run_search(index: int, query: str):
        # Create per-query ToolContext with unique worker_call_id
        worker_call_id = f"search-{index}-{uuid4()}"
        updated_metadata = {**tool_context.metadata, "worker_call_id": worker_call_id}
        query_context = tool_context.model_copy(update={"metadata": updated_metadata})

        # Run synchronous worker in thread pool to avoid blocking
        return await loop.run_in_executor(
            None,
            _execute_single_search,
            worker,
            query,
            index,
            query_context,
        )

    # Execute all searches in parallel with optional total timeout
    tasks: list[asyncio.Task[Any]] = []
    task_map: dict[asyncio.Task[Any], tuple[int, str]] = {}
    for index, query in enumerate(queries):
        task = asyncio.create_task(run_search(index, query))
        tasks.append(task)
        task_map[task] = (index, query)

    done: set[asyncio.Task[Any]]
    pending: set[asyncio.Task[Any]]
    if total_timeout_s is not None and total_timeout_s > 0:
        done, pending = await asyncio.wait(tasks, timeout=total_timeout_s)
    else:
        done, pending = await asyncio.wait(tasks)

    aggregated: list[SearchResultPayload] = []
    search_meta: list[SearchResponseMeta] = []
    errors: list[SearchError] = []

    for task in done:
        index, query = task_map[task]
        try:
            res_list, meta, error = task.result()
        except Exception as exc:
            errors.append(
                SearchError(
                    query=query,
                    error=type(exc).__name__,
                    message=str(exc),
                )
            )
        else:
            aggregated.extend(res_list)
            if meta:
                search_meta.append(meta)
            if error:
                errors.append(error)

    if pending:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in pending:
            _, query = task_map[task]
            errors.append(
                SearchError(
                    query=query,
                    error="TimeoutError",
                    message=(
                        f"Search timed out after {total_timeout_s}s"
                        if total_timeout_s is not None
                        else "Search timed out"
                    ),
                )
            )
        LOGGER.warning(
            "parallel_search_timeout",
            extra={
                "timeout_s": total_timeout_s,
                "pending": len(pending),
                "completed": len(done),
            },
        )

    return aggregated, search_meta, errors


@observe_span(name="node.search")
async def search_node(state: CollectionSearchState) -> dict[str, Any]:
    """Execute parallel web search using asyncio.gather for I/O-bound operations."""
    strategy_state = state.get("strategy")
    strategy = strategy_state.plan if strategy_state else None
    if not strategy:
        return {
            "search": {
                "errors": [SearchError(message="No strategy generated")],
                "results": [],
            }
        }

    tool_context = state["tool_context"]
    runtime = state["runtime"]
    worker: WebSearchWorker | None = runtime.get("runtime_search_worker")
    if not worker:
        return {
            "search": {
                "errors": [SearchError(message="No search worker configured")],
                "results": [],
            }
        }

    graph_input = state["input"]
    ids = _get_ids(tool_context, graph_input.collection_scope)

    # Execute searches in parallel
    try:
        aggregated, search_meta, errors = await _execute_parallel_searches(
            worker,
            strategy.queries,
            tool_context,
            total_timeout_s=_resolve_search_timeout_s(),
        )
    except Exception as exc:
        LOGGER.exception("parallel_search_failed")
        return {
            "search": {
                "errors": [SearchError(error=type(exc).__name__, message=str(exc))],
                "results": [],
            }
        }

    meta = dict(ids)
    timeout_count = sum(1 for error in errors if error.error == "TimeoutError")
    meta.update(
        {
            "total_results": len(aggregated),
            "error_count": len(errors),
            "timeout_count": timeout_count,
        }
    )
    _record_transition(state, "search_node", "searched", "web_search_completed", meta)

    return {
        "search": {
            "results": aggregated,
            "errors": errors,
            "responses": search_meta,
        }
    }


_EMBEDDING_MAX_RETRIES = 3
_EMBEDDING_RETRY_DELAY_S = 0.5


async def _embed_with_retry(
    embedding_client: EmbeddingClient,
    texts: list[str],
    max_retries: int = _EMBEDDING_MAX_RETRIES,
) -> tuple[list[list[float]], bool]:
    """Embed texts with retry mechanism.

    Returns:
        Tuple of (vectors, success). On failure after all retries, returns ([], False).
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            embedding_result = await asyncio.to_thread(embedding_client.embed, texts)
            return embedding_result.vectors, True
        except Exception as exc:
            last_exc = exc
            LOGGER.warning(
                "embedding_rank.retry",
                extra={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error": str(exc),
                },
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(_EMBEDDING_RETRY_DELAY_S * (attempt + 1))

    LOGGER.error(
        "embedding_rank.failed_all_retries",
        exc_info=last_exc,
        extra={"max_retries": max_retries},
    )
    return [], False


@observe_span(name="node.embedding_rank")
async def embedding_rank_node(state: CollectionSearchState) -> dict[str, Any]:
    """Rank search results using embeddings.

    Uses configurable weights and top_k from GraphInput.
    Implements retry mechanism for embedding service calls.
    Sets explicit failure status if embedding service is unavailable.
    """
    search_state = state.get("search", {})
    results: list[SearchResultPayload] = search_state.get("results", [])
    if not results:
        return {"embedding_rank": {"scored_count": 0, "top_k": 0, "failed": False}}

    graph_input = state["input"]
    runtime = state["runtime"]
    query = graph_input.question
    purpose = graph_input.purpose

    # Use configurable parameters from GraphInput
    top_k = graph_input.embedding_top_k
    embedding_weight, weight_source = _resolve_embedding_weight(graph_input)
    heuristic_weight = 1.0 - embedding_weight

    # Embedding logic: keep query separate to avoid purpose dilution.
    texts_to_embed = _build_embedding_texts(query, purpose, results)

    # Call embedding service with retry
    embedding_client = runtime.get("runtime_embedding_client")
    if embedding_client is None:
        embedding_client = EmbeddingClient.from_settings()
        runtime["runtime_embedding_client"] = embedding_client

    vectors, embedding_success = await _embed_with_retry(
        embedding_client, texts_to_embed
    )

    query_vec = vectors[0] if vectors else []
    scored_results = []

    offset = 2 if purpose.strip() else 1
    for idx, result in enumerate(results):
        vec = vectors[idx + offset] if len(vectors) > idx + offset else []
        emb_score = 0.0
        if query_vec and vec:
            emb_score = cosine_similarity(query_vec, vec) * 100.0

        heu_score = calculate_generic_heuristics(result, query)

        # Apply configurable weights
        hybrid_score = (embedding_weight * emb_score) + (heuristic_weight * heu_score)

        scored_results.append(
            result.model_copy(
                update={
                    "embedding_rank_score": hybrid_score,
                    "embedding_similarity": emb_score,
                    "heuristic_score": heu_score,
                }
            )
        )

    scored_results.sort(key=lambda x: x.embedding_rank_score or 0.0, reverse=True)
    top_results = scored_results[:top_k]

    ids = _get_ids(state["tool_context"], graph_input.collection_scope)
    meta = dict(ids)
    meta.update(
        {
            "top_k": len(top_results),
            "embedding_success": embedding_success,
            "embedding_weight": embedding_weight,
            "embedding_weight_source": weight_source,
            "embedding_query_only": True,
        }
    )
    _record_transition(
        state, "embedding_rank_node", "ranked", "embedding_rank_completed", meta
    )

    # With Annotated reducer, we only need to return the keys we want to update
    # The _merge_dict reducer will preserve existing keys (errors, responses)
    return {
        "search": {
            "results": top_results,
            "embedding_failed": not embedding_success,
        },
        "embedding_rank": {
            "scored_count": len(scored_results),
            "top_k": len(top_results),
            "failed": not embedding_success,
            "embedding_weight": embedding_weight,
            "embedding_weight_source": weight_source,
            "embedding_query_only": True,
        },
    }


@observe_span(name="node.hybrid_score")
def hybrid_score_node(state: CollectionSearchState) -> dict[str, Any]:
    """Execute hybrid scoring via worker subgraph.

    Checks for upstream embedding failures and logs warnings if data quality
    may be compromised.
    """
    search_state = state.get("search", {})
    results: list[SearchResultPayload] = search_state.get("results", [])
    if not results:
        return {"hybrid": HybridState()}

    # Check for upstream embedding failures
    embedding_failed = search_state.get("embedding_failed", False)
    embedding_rank_state = state.get("embedding_rank", {})
    if embedding_failed or embedding_rank_state.get("failed"):
        LOGGER.warning(
            "hybrid_score.degraded_input",
            extra={
                "reason": "embedding_service_failed",
                "message": "Input scores are based on heuristics only, quality may be reduced",
            },
        )

    tool_context = state["tool_context"]
    runtime = state["runtime"]
    executor: HybridScoreExecutor | None = runtime.get("runtime_hybrid_executor")
    if not executor:
        return {"hybrid": HybridState(error="No hybrid executor configured")}

    graph_input = state["input"]
    max_candidates = graph_input.max_candidates
    sliced = results[:max_candidates]

    candidates: list[SearchCandidate] = []

    for item in sliced:
        # Construct candidates (simplified for brevity)
        c_id = f"q{item.query_index}-{item.position}"
        try:
            cand = SearchCandidate(
                id=c_id,
                title=item.title,
                snippet=item.snippet,
                url=item.url,
                is_pdf=bool(item.is_pdf),
                detected_date=item.detected_date,
                version_hint=item.version_hint,
                domain_type=item.domain_type,
                trust_hint=item.trust_hint,
            )
        except ValidationError:
            continue
        candidates.append(cand)

    if not candidates:
        return {"hybrid": HybridState(result=None)}

    strategy_state = state.get("strategy")
    strategy = strategy_state.plan if strategy_state else None
    preferred = list(strategy.preferred_sources) if strategy else []
    disallowed = list(strategy.disallowed_sources) if strategy else []

    scoring_ctx_kwargs: dict[str, Any] = {
        "question": graph_input.question,
        "purpose": graph_input.purpose,
        "output_target": "hybrid_rerank",
        "preferred_sources": preferred,
        "disallowed_sources": disallowed,
        "collection_scope": graph_input.collection_scope,
        "freshness_mode": _FRESHNESS_MAP.get(
            graph_input.quality_mode, FreshnessMode.STANDARD
        ),
        "min_diversity_buckets": MIN_DIVERSITY_BUCKETS,
    }
    jurisdiction_value = tool_context.metadata.get("jurisdiction")
    if isinstance(jurisdiction_value, str) and jurisdiction_value.strip():
        scoring_ctx_kwargs["jurisdiction"] = jurisdiction_value

    scoring_ctx = ScoringContext(**scoring_ctx_kwargs)

    # Include full tool_context for sub-graph context reconstruction
    # tool_context_from_meta() expects either tool_context or scope_context in meta
    scope_payload = tool_context.scope.model_dump(mode="json", exclude_none=True)
    business_payload = tool_context.business.model_dump(mode="json", exclude_none=True)
    tenant_ctx = {
        "tenant_id": tool_context.scope.tenant_id,
        "trace_id": tool_context.scope.trace_id,
        "case_id": tool_context.business.case_id,
        "query": graph_input.question,
        # Required for tool_context_from_meta() in hybrid_search_and_score
        "tool_context": tool_context.model_dump(mode="json"),
        "scope_context": scope_payload,
        "business_context": business_payload,
    }

    try:
        hybrid_res = executor.run(
            scoring_context=scoring_ctx,
            candidates=candidates,
            tenant_context=tenant_ctx,
        )
    except Exception as exc:
        LOGGER.exception("hybrid_score_failed")
        return {"hybrid": HybridState(error=str(exc))}

    ids = _get_ids(tool_context, graph_input.collection_scope)
    meta = dict(ids)
    meta.update(
        {
            "ranked_count": len(hybrid_res.ranked),
            "embedding_degraded": embedding_failed
            or embedding_rank_state.get("failed", False),
        }
    )
    _record_transition(
        state, "hybrid_score_node", "scored", "hybrid_score_completed", meta
    )

    return {
        "hybrid": HybridState(
            result=hybrid_res,
            candidates=candidates,
            embedding_degraded=embedding_failed
            or embedding_rank_state.get("failed", False),
        )
    }


@observe_span(name="node.hitl")
def hitl_node(state: CollectionSearchState) -> dict[str, Any]:
    """Present results to HITL gateway."""
    tool_context = state["tool_context"]
    runtime = state["runtime"]
    gateway: HitlGateway | None = runtime.get("runtime_hitl_gateway")
    if not gateway:
        return {"hitl": {"auto_approved": True}}

    hybrid_state = state.get("hybrid")
    result_data = hybrid_state.result if hybrid_state else None
    if not result_data:
        return {"hitl": {"auto_approved": True}}  # Nothing to approve or skip

    # logic to checking existing hitl state or presenting
    # Simplified for refactor: calling present()
    # In a real LangGraph interacting with a human, we'd use an interrupt.
    # Here we mimic the legacy "gateway" pattern.

    graph_input = state["input"]
    ids = _get_ids(tool_context, graph_input.collection_scope)
    payload = build_hitl_payload(
        ids=ids,
        input_data=graph_input.model_dump(mode="json"),
        result_data=result_data.model_dump(mode="json"),
    )

    decision = gateway.present(payload)

    meta = dict(ids)
    status = "pending"
    if decision:
        status = decision.status

    meta["decision_status"] = status
    _record_transition(state, "hitl_node", status, "hitl_gateway_checked", meta)

    if decision:
        return {"hitl": {"decision": decision}}
    return {"hitl": {"decision": None}}


@observe_span(name="node.trigger_ingestion")
def trigger_ingestion_node(state: CollectionSearchState) -> dict[str, Any]:
    """Trigger ingestion for selected URLs via CrawlerManager."""
    plan_data = state.get("plan")
    if not plan_data:
        return {"ingestion": {"status": "skipped_no_plan"}}

    if isinstance(plan_data, ImplementationPlan):
        plan = plan_data
    else:
        try:
            plan = ImplementationPlan.model_validate(plan_data)
        except ValidationError as exc:
            LOGGER.exception("plan_validation_failed")
            return {"ingestion": {"error": f"Invalid plan: {exc}"}}

    # Check execution conditions
    graph_input = state["input"]
    hitl_state = state.get("hitl", {})
    decision = hitl_state.get("decision")
    if isinstance(decision, HitlDecision):
        decision_status = decision.status
    elif isinstance(decision, Mapping):
        decision_status = decision.get("status")
    else:
        decision_status = None

    should_execute = (
        graph_input.execute_plan
        or graph_input.auto_ingest
        or decision_status in ("approved", "partial")
    )

    if not should_execute:
        return {"ingestion": {"status": "planned_only"}}

    selected_urls = _extract_slot_value(plan, "selected_urls") or []
    if not isinstance(selected_urls, list) or not selected_urls:
        return {"ingestion": {"status": "skipped_no_selection"}}

    # Resolve Crawler Manager
    runtime = state["runtime"]
    manager: CrawlerManagerProtocol | None = runtime.get("runtime_crawler_manager")
    if not manager:
        return {"ingestion": {"error": "No crawler manager configured"}}

    tool_context = state["tool_context"]
    collection_id = _extract_slot_value(plan, "collection_id")
    if not collection_id:
        return {"ingestion": {"error": "Missing collection_id in plan"}}

    # Build Crawler Request (Bulk)
    # CrawlerRunRequest automatically creates Origins from origins list
    origins = [CrawlerOriginConfig(url=u) for u in selected_urls]

    crawl_req = CrawlerRunRequest(
        origins=origins,
        collection_id=collection_id,
        mode="live",  # Default to live crawl
        provider="web",
        fetch=True,
        workflow_id=tool_context.business.workflow_id,
    )

    # Prepare Meta for Context
    meta = {
        "tenant_id": tool_context.scope.tenant_id,
        "trace_id": tool_context.scope.trace_id,
        "case_id": tool_context.business.case_id,
        "user_id": tool_context.scope.user_id,
        "ingestion_run_id": tool_context.scope.ingestion_run_id or str(uuid4()),
    }

    try:
        result_info = manager.dispatch_crawl_request(crawl_req, meta)
        updated_plan = _update_ingestion_task(plan, result_info)
        return {
            "ingestion": {"status": "triggered", "task_info": result_info},
            "plan": updated_plan,
        }
    except Exception as exc:
        LOGGER.exception("Crawler dispatch failed")
        return {"ingestion": {"error": str(exc)}}


@observe_span(name="node.build_plan")
def build_plan_node(state: CollectionSearchState) -> dict[str, Any]:
    """Construct and persist the ImplementationPlan."""
    graph_input = state["input"]
    tool_context = state["tool_context"]
    ids = _get_ids(tool_context, graph_input.collection_scope)
    strategy_state = state.get("strategy")
    hybrid_state = state.get("hybrid")
    hitl = state.get("hitl", {})
    decision_data = hitl.get("decision")

    # Determine selection
    selected_urls = []
    reason = None
    candidate_by_id = (
        {candidate.id: candidate for candidate in hybrid_state.candidates}
        if hybrid_state
        else {}
    )
    if decision_data:
        # User selection via HITL
        if isinstance(decision_data, HitlDecision):
            decision = decision_data
        else:
            decision = HitlDecision.model_validate(decision_data)
        if decision.status in ("approved", "partial"):
            for cid in decision.approved_candidate_ids:
                candidate = candidate_by_id.get(cid)
                if candidate and candidate.url:
                    selected_urls.append(candidate.url)
            selected_urls.extend(decision.added_urls)
            reason = decision.rationale
    elif graph_input.auto_ingest:
        # Auto-ingest logic (simplified)
        results = (
            hybrid_state.result.ranked if hybrid_state and hybrid_state.result else []
        )
        selected_urls = select_auto_ingest_urls(
            results,
            top_k=graph_input.auto_ingest_top_k,
            min_score=graph_input.auto_ingest_min_score,
        )
        reason = "auto_ingest"

    hitl_reasons = state.get("hitl", {}).get("reasons", [])
    auto_approved = bool(hitl.get("auto_approved"))

    # Construct Plan
    try:
        profile_id, profile_version = _resolve_plan_profile(tool_context)
        scope = PlanScope(
            tenant_id=cast(str, ids["tenant_id"]),
            gremium_identifier=_resolve_gremium_identifier(tool_context, graph_input),
            framework_profile_id=profile_id,
            framework_profile_version=profile_version,
            case_id=ids["case_id"],
            workflow_id=cast(str, ids["workflow_id"]),
            run_id=cast(str, ids["run_id"] or ids["ingestion_run_id"]),
        )
        plan_key = derive_plan_key(scope)
        plan_strategy = (
            strategy_state.plan
            if strategy_state and strategy_state.plan
            else fallback_strategy(
                SearchStrategyRequest(
                    tenant_id=ids["tenant_id"],
                    query=graph_input.question,
                    quality_mode="standard",
                    purpose="fallback",
                )
            )
        )
        search_results: list[SearchResultPayload] = state.get("search", {}).get(
            "results", []
        )
        candidates = [result.model_dump(mode="json") for result in search_results]
        scored_candidates = (
            [item.model_dump(mode="json") for item in hybrid_state.result.ranked]
            if hybrid_state and hybrid_state.result
            else []
        )
        selected_evidence = [
            Evidence(ref_type="url", ref_id=url, summary="selected_url")
            for url in selected_urls
        ]

        hitl_required = decision_data is None and not graph_input.auto_ingest
        gate_required = decision_data is not None or hitl_required
        if decision_data:
            if isinstance(decision_data, HitlDecision):
                decision = decision_data
            else:
                decision = HitlDecision.model_validate(decision_data)
            hitl_status = decision.status
            hitl_required = False
            hitl_rationale = decision.rationale
        elif auto_approved or graph_input.auto_ingest:
            hitl_status = "approved"
            hitl_required = False
            hitl_rationale = None
        else:
            hitl_status = "pending"
            hitl_rationale = None

        plan = ImplementationPlan(
            plan_key=plan_key,
            scope=scope,
            slots=[
                Slot(
                    key="collection_id",
                    status="filled",
                    value=graph_input.collection_scope,
                    slot_type="collection_id",
                ),
                Slot(
                    key="selected_urls",
                    status="filled" if selected_urls else "pending",
                    value=selected_urls,
                    slot_type="url_list",
                    provenance=selected_evidence,
                ),
            ],
            tasks=[
                Task(
                    key="select_sources",
                    status="completed" if selected_urls else "pending",
                    outputs=selected_evidence,
                ),
                Task(
                    key="execute_ingestion",
                    status="pending" if selected_urls else "blocked",
                ),
            ],
            gates=[
                Gate(
                    key="hitl_review",
                    status=hitl_status,
                    required=gate_required,
                    rationale=hitl_rationale,
                    evidence=(
                        selected_evidence
                        if hitl_status in ("approved", "partial")
                        else []
                    ),
                    metadata={
                        "reasons": hitl_reasons,
                        "auto_approved": auto_approved,
                    },
                )
            ],
            deviations=[],
            evidence=selected_evidence,
            metadata={
                "strategy": plan_strategy.model_dump(mode="json"),
                "candidates": candidates,
                "scored_candidates": scored_candidates,
                "selection": {
                    "selected_urls": selected_urls,
                    "selection_reason": reason,
                },
                "hitl": {
                    "required": hitl_required,
                    "reasons": hitl_reasons,
                    "review_payload": hitl.get("review_payload"),
                    "decision_status": hitl_status,
                },
                "execution_mode": "acquire_and_ingest",
                "ingest_policy": None,
            },
        )
    except ValidationError as ve:
        LOGGER.error(f"Plan validation failed: {ve}")
        return {"ingestion": {"error": str(ve)}}
    except ValueError as exc:
        LOGGER.error(f"Plan scope failed: {exc}")
        return {"ingestion": {"error": str(exc)}}

    return {"plan": plan}


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
    workflow.add_node("trigger_ingestion", trigger_ingestion_node)
    # workflow.add_node("verification", verification_node)

    workflow.set_entry_point("strategy")
    workflow.add_edge("strategy", "search")
    workflow.add_edge("search", "embedding_rank")
    workflow.add_edge("embedding_rank", "hybrid_score")
    workflow.add_edge("hybrid_score", "hitl")
    workflow.add_edge("hitl", "build_plan")
    workflow.add_edge("build_plan", "trigger_ingestion")
    workflow.add_edge("trigger_ingestion", END)

    return workflow.compile()


# -----------------------------------------------------------------------------
# Integration Adapters (Backward Compatibility)
# -----------------------------------------------------------------------------


class CollectionSearchAdapter:
    """Adapter to expose the new LangGraph via the legacy .run() API."""

    io_spec: GraphIOSpec = COLLECTION_SEARCH_IO

    def __init__(self, dependencies: dict[str, Any]):
        # Compile graph once per adapter instance (Finding #4 fix)
        # No global cache to prevent state leakage across workers
        self.runnable = build_compiled_graph()
        self.dependencies = dependencies

    def run(
        self, state: Mapping[str, Any] | None, meta: Mapping[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the collection search graph with strict GraphIOSpec enforcement.

        BREAKING CHANGE: schema_id and schema_version are now mandatory.
        All callers must provide a valid CollectionSearchGraphRequest.
        """

        # 1. Validate Boundary (Hard Enforcement)
        raw_state = dict(state or {})

        try:
            boundary = CollectionSearchGraphRequest.model_validate(raw_state)
        except ValidationError as exc:
            raise InvalidGraphInput(
                f"Invalid CollectionSearchGraphRequest: {exc}. "
                f"schema_id and schema_version are mandatory."
            ) from exc

        # 2. Extract validated components
        input_state = boundary.input
        tool_context = boundary.tool_context
        runtime: dict[str, Any] = dict(self.dependencies)

        if boundary.runtime:
            runtime.update(boundary.runtime)

        # 3. Resolve ToolContext (boundary only, fail fast)
        if tool_context is None:
            raise InvalidGraphInput("tool_context is required in the boundary state")

        _validate_tenant_context(tool_context)

        initial_state: CollectionSearchState = {
            "input": input_state,
            "tool_context": tool_context,
            "runtime": runtime,
            "strategy": None,
            "search": {},
            "embedding_rank": {},
            "hybrid": None,
            "hitl": {},
            "ingestion": {},
            "meta": {},  # Legacy
            "telemetry": {},  # Legacy
            "transitions": [],
            "plan": None,
        }

        # 3. Invoke Graph
        try:
            timeout_s = _resolve_graph_timeout_s(runtime)
            if timeout_s is None:
                final_state = _run_async(self.runnable.ainvoke(initial_state))
            else:
                final_state = _run_async(
                    _ainvoke_with_timeout(self.runnable, initial_state, timeout_s)
                )
        except (asyncio.TimeoutError, TimeoutError):
            timeout_s = _resolve_graph_timeout_s(runtime)
            error_payload = _timeout_payload(timeout_s).model_dump(mode="json")
            return {"error": "graph_timeout"}, error_payload
        except Exception as exc:
            LOGGER.exception("graph_execution_failed")
            error_payload = CollectionSearchGraphOutput(
                outcome="error",
                search=None,
                telemetry=None,
                ingestion=None,
                plan=None,
                hitl=None,
                error=str(exc),
            ).model_dump(mode="json")
            return {"error": str(exc)}, error_payload

        # 4. Map Result (Legacy _build_result compatibility)
        # The legacy run() returned (state, result_dict)
        # We need to reconstruct the expected result dict structure

        search_res = final_state.get("search", {}) or {}
        search_payload = (
            SearchResultsPayload.model_validate(search_res) if search_res else None
        )
        hybrid_state = final_state.get("hybrid")
        strategy_state = final_state.get("strategy")

        fatal_messages: list[str] = []
        search_errors = search_payload.errors if search_payload else []
        search_results = search_payload.results if search_payload else []
        if search_errors and not search_results:
            first_error = search_errors[0]
            error_detail = first_error.message or first_error.error
            detail = f": {error_detail}" if error_detail else ""
            fatal_messages.append(f"search_failed{detail}")

        strategy_error = strategy_state.error if strategy_state else None
        if strategy_error:
            fatal_messages.append(f"strategy_failed: {strategy_error}")

        hybrid_error = hybrid_state.error if hybrid_state else None
        if hybrid_error:
            fatal_messages.append(f"hybrid_score_failed: {hybrid_error}")

        outcome = "error" if fatal_messages else "completed"
        error_message = "; ".join(fatal_messages) if fatal_messages else None

        # Construct summary result
        hitl_payload = final_state.get("hitl")
        if isinstance(hitl_payload, Mapping):
            decision = hitl_payload.get("decision")
            if isinstance(decision, HitlDecision):
                hitl_payload = {
                    **hitl_payload,
                    "decision": decision.model_dump(mode="json"),
                }

        plan_payload = final_state.get("plan")
        if isinstance(plan_payload, ImplementationPlan):
            plan_payload = plan_payload.model_dump(mode="json")

        result_payload = CollectionSearchGraphOutput(
            outcome=outcome,
            search=search_payload,
            telemetry=final_state.get("telemetry"),
            ingestion=final_state.get("ingestion"),
            plan=plan_payload,
            hitl=hitl_payload,
            error=error_message,
        ).model_dump(mode="json")

        # Merge back to raw state just in case caller expects it
        return final_state, result_payload


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
    from crawler.manager import CrawlerManager

    class _HybridExecutorAdapter:
        def run(self, *, scoring_context, candidates, tenant_context) -> HybridResult:
            state = {
                "query": scoring_context.question,
                "candidates": list(candidates),
            }
            meta = {"scoring_context": scoring_context, **tenant_context}
            if "query" not in meta:
                meta["query"] = scoring_context.question
            _, result = hybrid_run(state, meta)
            payload = result.get("result") if isinstance(result, Mapping) else None
            if isinstance(payload, HybridResult):
                return payload
            raise InvalidGraphInput(
                "Hybrid executor must return HybridResult in result['result']"
            )

    search_worker = get_web_search_worker()

    dependencies = {
        "runtime_strategy_generator": llm_strategy_generator,
        "runtime_search_worker": search_worker,
        "runtime_hybrid_executor": _HybridExecutorAdapter(),
        "runtime_hitl_gateway": _ProductionHitlGateway(),
        "runtime_crawler_manager": CrawlerManager(),
        "runtime_coverage_verifier": _ProductionCoverageVerifier(),
    }

    return CollectionSearchAdapter(dependencies)
