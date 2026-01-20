"""Production retrieval augmented generation graph."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Tuple, TypedDict, cast

from ai_core.graph.core import GraphContext, ThreadAwareCheckpointer
from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.infra.observability import emit_event, observe_span, update_observation
from ai_core.nodes import compose, retrieve
from ai_core.rag import answer_guardrails
from ai_core.rag import rerank as rag_rerank
from ai_core.rag import semantic_cache
from ai_core.rag import standalone_question as rag_standalone
from ai_core.rag import strategy as rag_strategy
from ai_core.rag.feedback import enqueue_used_source_feedback
from ai_core.rag.schemas import RagReasoning, SourceRef
from ai_core.tool_contracts import ContextError, NotFoundError, ToolContext
from ai_core.tool_contracts.base import tool_context_from_meta
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError


logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 1
DEFAULT_SCORE_THRESHOLD = 0.65
DEFAULT_SCORE_DELTA = 0.05
DEFAULT_QUERY_VARIANTS = 3
RETRY_QUERY_VARIANTS = 5
DEFAULT_HISTORY_LIMIT = 6


def _resolve_history_limit() -> int:
    try:
        limit = int(
            os.getenv("RAG_CHAT_HISTORY_MAX_MESSAGES", str(DEFAULT_HISTORY_LIMIT))
        )
        if limit < 1:
            return DEFAULT_HISTORY_LIMIT
        return limit
    except (TypeError, ValueError):
        return DEFAULT_HISTORY_LIMIT


def _load_history(state: object) -> list[dict[str, str]]:
    if not isinstance(state, dict):
        return []
    history = state.get("chat_history")
    if not isinstance(history, list):
        return []
    cleaned: list[dict[str, str]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _append_history(
    history: list[dict[str, str]],
    *,
    role: str,
    content: str | None,
) -> None:
    if not content:
        return
    history.append({"role": role, "content": content})


def _trim_history(history: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return history
    if len(history) <= limit:
        return history
    return history[-limit:]


RAG_SCHEMA_ID = "noesis.graphs.retrieval_augmented_generation"
RAG_IO_VERSION = GraphIOVersion(major=1, minor=1, patch=0)
RAG_IO_VERSION_STRING = RAG_IO_VERSION.as_string()


class RetrievalAugmentedGenerationInput(retrieve.RetrieveInput):
    """Boundary input model for the retrieval augmented generation graph."""

    schema_id: Literal[RAG_SCHEMA_ID] = RAG_SCHEMA_ID
    schema_version: Literal[RAG_IO_VERSION_STRING] = RAG_IO_VERSION_STRING

    question: str | None = None

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        frozen=True,
    )


class RetrievalAugmentedGenerationOutput(BaseModel):
    """Boundary output model for the retrieval augmented generation graph."""

    schema_id: Literal[RAG_SCHEMA_ID] = RAG_SCHEMA_ID
    schema_version: Literal[RAG_IO_VERSION_STRING] = RAG_IO_VERSION_STRING
    answer: str | None
    prompt_version: str | None
    retrieval: Mapping[str, Any]
    snippets: list[dict[str, Any]]
    reasoning: RagReasoning | None = None
    used_sources: list[SourceRef] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    debug_meta: Mapping[str, Any] | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


RAG_GRAPH_IO = GraphIOSpec(
    schema_id=RAG_SCHEMA_ID,
    version=RAG_IO_VERSION,
    input_model=RetrievalAugmentedGenerationInput,
    output_model=RetrievalAugmentedGenerationOutput,
)


class RetrieveNode(Protocol):
    """Protocol describing the retrieve node callable."""

    def __call__(
        self,
        context: ToolContext,
        params: retrieve.RetrieveInput,
    ) -> retrieve.RetrieveOutput:
        """Execute retrieval and return the structured output payload."""


class ComposeNode(Protocol):
    """Protocol describing the compose node callable."""

    def __call__(
        self, context: ToolContext, params: compose.ComposeInput
    ) -> compose.ComposeOutput:
        """Execute composition and return the structured output payload."""


class RetrievalAugmentedGenerationState(TypedDict, total=False):
    """Runtime state for the retrieval augmented generation LangGraph."""

    state: MutableMapping[str, Any]
    meta: MutableMapping[str, Any]
    context: ToolContext
    graph_input: RetrievalAugmentedGenerationInput
    chat_history: list[Mapping[str, Any]]
    query_variants: list[str]
    retry_count: int
    hybrid_override: Mapping[str, Any] | None
    cache_hit: bool
    cache_response: Mapping[str, Any] | None
    cache_embedding: list[float] | None
    retrieval_output: retrieve.RetrieveOutput
    rerank_result: rag_rerank.RerankResult
    standalone_question: str | None
    retry: bool
    result: Mapping[str, Any]


def _ensure_mutable_state(
    state: Mapping[str, Any] | MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    if isinstance(state, MutableMapping):
        return state
    return dict(state)


def _ensure_mutable_meta(
    meta: Mapping[str, Any] | MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    if isinstance(meta, MutableMapping):
        return meta
    return dict(meta)


def _normalise_snippets(
    snippets: object, fallback: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return a list of snippet mappings with required fields present."""

    normalised: list[dict[str, Any]] = []

    if isinstance(snippets, list):
        candidates = [item for item in snippets if isinstance(item, Mapping)]
    else:
        candidates = []

    if not candidates:
        candidates = [item for item in fallback if isinstance(item, Mapping)]

    fallback_by_id: dict[Any, Mapping[str, Any]] = {}
    for item in fallback:
        if isinstance(item, Mapping):
            identifier = item.get("id")
            if identifier is not None:
                fallback_by_id[identifier] = item

    for index, candidate in enumerate(candidates):
        base = dict(candidate)
        fallback_item: Mapping[str, Any] | None = None

        identifier = base.get("id")
        if identifier is not None and identifier in fallback_by_id:
            fallback_item = fallback_by_id[identifier]
        elif index < len(fallback):
            possible = fallback[index]
            if isinstance(possible, Mapping):
                fallback_item = possible

        text = base.get("text")
        if not isinstance(text, str):
            fallback_text = (
                fallback_item.get("text")
                if isinstance(fallback_item, Mapping)
                else None
            )
            base["text"] = str(fallback_text or "")

        source = base.get("source")
        if not isinstance(source, str):
            fallback_source = (
                fallback_item.get("source")
                if isinstance(fallback_item, Mapping)
                else None
            )
            base["source"] = str(fallback_source or "")

        score = base.get("score")
        try:
            base["score"] = float(score)
        except (TypeError, ValueError):
            fallback_score = None
            if isinstance(fallback_item, Mapping):
                fallback_score = fallback_item.get("score")
            try:
                base["score"] = float(fallback_score)
            except (TypeError, ValueError):
                base["score"] = 0.0

        citation = base.get("citation")
        if not isinstance(citation, str):
            fallback_citation = None
            if isinstance(fallback_item, Mapping):
                fallback_citation = fallback_item.get("citation")
            if isinstance(fallback_citation, str):
                base["citation"] = fallback_citation
            elif isinstance(base.get("source"), str):
                base["citation"] = base["source"]

        normalised.append(base)

    return normalised


def _build_tool_context(meta: MutableMapping[str, Any]) -> ToolContext:
    try:
        return tool_context_from_meta(meta)
    except Exception as exc:
        raise ContextError(
            "scope_context or tool_context is required for retrieval graphs",
            field="scope_context",
        ) from exc


def _coerce_graph_input(
    state: Mapping[str, Any],
) -> RetrievalAugmentedGenerationInput:
    payload = {
        field: state[field]
        for field in RetrievalAugmentedGenerationInput.model_fields
        if field in state and state[field] is not None
    }
    return RetrievalAugmentedGenerationInput.model_validate(payload)


def _resolve_base_query(graph_input: RetrievalAugmentedGenerationInput) -> str:
    query = graph_input.query
    if isinstance(query, str) and query.strip():
        return query.strip()
    question = graph_input.question
    if isinstance(question, str) and question.strip():
        return question.strip()
    return ""


def _coerce_int(value: object, *, fallback: int) -> int:
    try:
        candidate = int(str(value))
    except (TypeError, ValueError):
        return fallback
    if candidate < 0:
        return fallback
    return candidate


def _coerce_float(value: object, *, fallback: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return fallback
    if candidate != candidate or candidate in (float("inf"), float("-inf")):
        return fallback
    return candidate


def _resolve_int_env(name: str, fallback: int) -> int:
    return _coerce_int(os.getenv(name), fallback=fallback)


def _resolve_float_env(name: str, fallback: float) -> float:
    return _coerce_float(os.getenv(name), fallback=fallback)


def _coerce_score(value: object) -> float:
    return _coerce_float(value, fallback=0.0)


def _match_key(match: Mapping[str, Any], index: int) -> tuple[str, ...]:
    meta = match.get("meta")
    if isinstance(meta, Mapping):
        for key in ("chunk_id", "document_id", "hash"):
            raw = meta.get(key)
            if raw is not None:
                text = str(raw).strip()
                if text:
                    return (key, text)
    for key in ("id", "hash", "source"):
        raw = match.get(key)
        if raw is not None:
            text = str(raw).strip()
            if text:
                return (key, text)
    text_value = match.get("text")
    if isinstance(text_value, str) and text_value.strip():
        return ("text", text_value.strip()[:80])
    return ("index", str(index))


def _dedupe_matches(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for index, match in enumerate(matches):
        if not isinstance(match, Mapping):
            continue
        key = _match_key(match, index)
        if key in aggregated:
            existing = aggregated[key]
            if _coerce_score(match.get("score")) > _coerce_score(existing.get("score")):
                aggregated[key] = dict(match)
            continue
        aggregated[key] = dict(match)
        order.append(key)

    ordered = [aggregated[key] for key in order]
    ordered.sort(
        key=lambda item: (-_coerce_score(item.get("score")), str(item.get("id") or ""))
    )
    return ordered


def _aggregate_retrieval_meta(
    outputs: Sequence[retrieve.RetrieveOutput],
    matches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not outputs:
        return {}
    last_meta = outputs[-1].meta
    meta_payload = last_meta.model_dump(mode="json", exclude_none=True)
    took_ms = sum(_coerce_int(out.meta.took_ms, fallback=0) for out in outputs)
    vector_candidates = sum(
        _coerce_int(out.meta.vector_candidates, fallback=0) for out in outputs
    )
    lexical_candidates = sum(
        _coerce_int(out.meta.lexical_candidates, fallback=0) for out in outputs
    )
    meta_payload["matches_returned"] = len(matches)
    meta_payload["took_ms"] = took_ms
    meta_payload["vector_candidates"] = vector_candidates
    meta_payload["lexical_candidates"] = lexical_candidates
    return meta_payload


def _confidence_metrics(snippets: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    scores = [_coerce_score(snippet.get("score")) for snippet in snippets]
    scores = [score for score in scores if score > 0.0]
    if not scores:
        return 0.0, 0.0
    scores.sort(reverse=True)
    top_score = scores[0]
    if len(scores) >= 3:
        delta = top_score - scores[2]
    elif len(scores) == 2:
        delta = top_score - scores[1]
    else:
        delta = top_score
    return top_score, delta


def _should_retry(
    *,
    top_score: float,
    delta: float,
    candidate_count: int,
    retry_count: int,
) -> bool:
    max_retries = _resolve_int_env("RAG_CONFIDENCE_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    if retry_count >= max_retries:
        return False
    score_threshold = _resolve_float_env(
        "RAG_CONFIDENCE_TOP_SCORE", DEFAULT_SCORE_THRESHOLD
    )
    delta_threshold = _resolve_float_env(
        "RAG_CONFIDENCE_SCORE_DELTA", DEFAULT_SCORE_DELTA
    )
    if candidate_count == 0:
        return True
    if top_score < score_threshold:
        return True
    if delta < delta_threshold:
        return True
    return False


def _broaden_hybrid_config(
    hybrid: Mapping[str, Any] | None, *, attempt: int
) -> Mapping[str, Any] | None:
    if hybrid is None or attempt <= 0:
        return None
    updated = dict(hybrid)
    min_sim = _coerce_float(updated.get("min_sim"), fallback=0.0)
    updated["min_sim"] = max(0.0, min_sim * (0.5**attempt))
    top_k = _coerce_int(updated.get("top_k"), fallback=0)
    if top_k > 0:
        updated["top_k"] = min(top_k + 2 * attempt, top_k * 2)
    max_candidates = _coerce_int(updated.get("max_candidates"), fallback=0)
    if max_candidates > 0:
        updated["max_candidates"] = min(
            max_candidates + 10 * attempt,
            max_candidates * 2,
        )
    return updated


def _build_compiled_graph(
    *,
    retrieve_node: RetrieveNode,
    compose_node: ComposeNode,
) -> Any:
    @observe_span(name="rag.contextualize")
    def _contextualize_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]

        try:
            graph_input = graph_state.get("graph_input")
            if graph_input is None:
                graph_input = _coerce_graph_input(working_state)
        except ValidationError as exc:
            raise ValueError(f"Invalid graph input: {exc.errors()}") from exc

        context = graph_state["context"]
        base_query = _resolve_base_query(graph_input)
        history = working_state.get("chat_history")
        if not isinstance(history, list):
            history_items: list[Mapping[str, Any]] = []
        else:
            history_items = [item for item in history if isinstance(item, Mapping)]

        result = rag_standalone.generate_standalone_question(
            base_query,
            history_items,
            context,
        )
        standalone_question = result.question or base_query

        if base_query and "raw_question" not in working_state:
            working_state["raw_question"] = base_query
        if standalone_question:
            working_state["question"] = standalone_question
            working_state["query"] = standalone_question

        update_observation(
            metadata={
                "rag.standalone_source": result.source,
                "rag.standalone_error": result.error,
            }
        )

        return {
            "state": working_state,
            "context": context,
            "graph_input": _coerce_graph_input(working_state),
            "standalone_question": standalone_question,
        }

    @observe_span(name="rag.cache_lookup")
    def _cache_lookup_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]

        try:
            graph_input = graph_state.get("graph_input")
            if graph_input is None:
                graph_input = _coerce_graph_input(working_state)
        except ValidationError as exc:
            raise ValueError(f"Invalid graph input: {exc.errors()}") from exc

        context = graph_state["context"]
        base_query = _resolve_base_query(graph_input)

        cache = semantic_cache.get_semantic_cache()
        lookup = cache.lookup(base_query, context)

        if lookup.hit and lookup.response:
            update_observation(
                metadata={
                    "rag.cache_hit": True,
                    "rag.cache_similarity": lookup.similarity,
                }
            )
            emit_event(
                {
                    "event": "rag.cache.hit",
                    "tenant_id": context.scope.tenant_id,
                    "case_id": context.business.case_id,
                    "collection_id": context.business.collection_id,
                    "workflow_id": context.business.workflow_id,
                    "similarity": lookup.similarity,
                }
            )
            return {
                "state": working_state,
                "context": context,
                "graph_input": graph_input,
                "cache_hit": True,
                "cache_response": lookup.response,
            }

        update_observation(
            metadata={
                "rag.cache_hit": False,
                "rag.cache_reason": lookup.reason,
                "rag.cache_similarity": lookup.similarity,
            }
        )
        emit_event(
            {
                "event": "rag.cache.miss",
                "tenant_id": context.scope.tenant_id,
                "case_id": context.business.case_id,
                "collection_id": context.business.collection_id,
                "workflow_id": context.business.workflow_id,
                "reason": lookup.reason,
                "similarity": lookup.similarity,
            }
        )
        return {
            "state": working_state,
            "context": context,
            "graph_input": graph_input,
            "cache_hit": False,
            "cache_embedding": lookup.embedding,
        }

    @observe_span(name="rag.cache_finalize")
    def _cache_finalize_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]
        cached_payload = graph_state.get("cache_response") or {}
        retrieval_payload = cached_payload.get("retrieval") or {}
        if isinstance(retrieval_payload, Mapping):
            retrieval_payload = dict(retrieval_payload)
        else:
            retrieval_payload = {}
        retrieval_payload["cache_hit"] = True

        snippets_payload = cached_payload.get("snippets") or []
        if not isinstance(snippets_payload, list):
            snippets_payload = []
        snippets_payload = _normalise_snippets(snippets_payload, [])

        answer = cached_payload.get("answer")
        prompt_version = cached_payload.get("prompt_version")
        reasoning = cached_payload.get("reasoning")
        used_sources = cached_payload.get("used_sources") or []
        suggested_followups = cached_payload.get("suggested_followups") or []
        debug_meta = cached_payload.get("debug_meta")

        result_payload = RetrievalAugmentedGenerationOutput(
            answer=answer,
            prompt_version=prompt_version,
            retrieval=retrieval_payload,
            snippets=snippets_payload,
            reasoning=reasoning,
            used_sources=used_sources,
            suggested_followups=suggested_followups,
            debug_meta=debug_meta,
        ).model_dump(mode="json")

        working_state["answer"] = answer
        working_state["snippets"] = snippets_payload
        working_state["retrieval"] = retrieval_payload

        return {"state": working_state, "result": result_payload}

    @observe_span(name="rag.transform")
    def _transform_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]

        try:
            graph_input = graph_state.get("graph_input")
            if graph_input is None:
                graph_input = _coerce_graph_input(working_state)
        except ValidationError as exc:
            raise ValueError(f"Invalid graph input: {exc.errors()}") from exc

        context = graph_state["context"]
        base_query = _resolve_base_query(graph_input)

        if not working_state.get("question"):
            if graph_input.question:
                working_state["question"] = graph_input.question
            elif base_query:
                working_state["question"] = base_query

        retry_count = _coerce_int(graph_state.get("retry_count"), fallback=0)
        variants = rag_strategy.generate_query_variants(
            base_query,
            context,
            max_variants=DEFAULT_QUERY_VARIANTS,
        )
        queries = variants.queries
        if retry_count > 0:
            queries = rag_strategy.expand_query_variants(
                base_query,
                queries,
                max_variants=RETRY_QUERY_VARIANTS,
            )
        if not queries:
            queries = [base_query]

        hybrid_override = _broaden_hybrid_config(
            working_state.get("hybrid"), attempt=retry_count
        )

        logger.debug(
            "rag.query_transform",
            extra={
                "query_count": len(queries),
                "retry_count": retry_count,
                "tenant_id": context.scope.tenant_id,
                "case_id": context.business.case_id,
            },
        )
        update_observation(
            metadata={
                "rag.query_count": len(queries),
                "rag.retry_count": retry_count,
                "rag.query_source": variants.source,
            }
        )

        return {
            "state": working_state,
            "context": context,
            "graph_input": graph_input,
            "query_variants": queries,
            "retry_count": retry_count,
            "hybrid_override": hybrid_override,
        }

    @observe_span(name="rag.retrieve")
    def _retrieve_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]
        graph_input = graph_state.get("graph_input") or _coerce_graph_input(
            working_state
        )
        context = graph_state["context"]
        base_query = _resolve_base_query(graph_input)

        queries = graph_state.get("query_variants") or []
        if not queries and base_query:
            queries = [base_query]

        outputs: list[retrieve.RetrieveOutput] = []
        errors: list[Exception] = []
        matches: list[Mapping[str, Any]] = []
        last_output: retrieve.RetrieveOutput | None = None

        for query in queries:
            if not isinstance(query, str):
                continue
            query_text = query.strip()
            params_payload = graph_input.model_dump(
                exclude={"schema_id", "schema_version", "question"}
            )
            params_payload["query"] = query_text
            hybrid_override = graph_state.get("hybrid_override")
            if hybrid_override is not None:
                params_payload["hybrid"] = dict(hybrid_override)
            params = retrieve.RetrieveInput.model_validate(params_payload)
            try:
                retrieve_output = retrieve_node(context, params)
            except NotFoundError as exc:
                errors.append(exc)
                continue
            outputs.append(retrieve_output)
            last_output = retrieve_output
            matches.extend(retrieve_output.matches)

        if not outputs:
            if errors:
                raise errors[0]
            raise ValueError("retrieval produced no results")

        deduped = _dedupe_matches(matches)
        retrieval_meta = _aggregate_retrieval_meta(outputs, deduped)
        top_k = _coerce_int(retrieval_meta.get("top_k_effective"), fallback=0)
        if top_k > 0:
            deduped = deduped[:top_k]

        working_state["matches"] = deduped
        working_state["snippets"] = deduped
        working_state["retrieval"] = retrieval_meta

        update_observation(
            metadata={
                "rag.candidate_count": len(deduped),
                "rag.retrieval_took_ms": retrieval_meta.get("took_ms"),
                "rag.vector_candidates": retrieval_meta.get("vector_candidates"),
                "rag.lexical_candidates": retrieval_meta.get("lexical_candidates"),
            }
        )
        return {
            "state": working_state,
            "context": context,
            "graph_input": graph_input,
            "retrieval_output": last_output,
        }

    @observe_span(name="rag.rerank")
    def _rerank_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]
        graph_input = graph_state.get("graph_input") or _coerce_graph_input(
            working_state
        )
        context = graph_state["context"]
        base_query = _resolve_base_query(graph_input)
        matches = working_state.get("matches")
        if not isinstance(matches, list):
            matches = []

        retrieval_meta = working_state.get("retrieval")
        top_k = None
        if isinstance(retrieval_meta, Mapping):
            top_k = (
                _coerce_int(retrieval_meta.get("top_k_effective"), fallback=0) or None
            )

        rerank_result = rag_rerank.rerank_chunks(
            matches,
            base_query,
            context,
            top_k=top_k,
        )
        working_state["snippets"] = rerank_result.chunks
        working_state["matches"] = rerank_result.chunks

        update_observation(
            metadata={
                "rag.rerank_mode": rerank_result.mode,
                "rag.rerank_count": len(rerank_result.chunks),
            }
        )
        return {
            "state": working_state,
            "context": context,
            "graph_input": graph_input,
            "rerank_result": rerank_result,
        }

    @observe_span(name="rag.confidence")
    def _confidence_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]
        retry_count = _coerce_int(graph_state.get("retry_count"), fallback=0)
        snippets = working_state.get("snippets")
        if not isinstance(snippets, list):
            snippets = []
        snippet_items = [
            snippet for snippet in snippets if isinstance(snippet, Mapping)
        ]

        top_score, delta = _confidence_metrics(snippet_items)
        should_retry = _should_retry(
            top_score=top_score,
            delta=delta,
            candidate_count=len(snippet_items),
            retry_count=retry_count,
        )
        logger.debug(
            "rag.confidence_check",
            extra={
                "top_score": top_score,
                "score_delta": delta,
                "candidate_count": len(snippet_items),
                "retry_count": retry_count,
                "retry": should_retry,
            },
        )

        if should_retry:
            retry_count += 1

        update_observation(
            metadata={
                "rag.top_score": top_score,
                "rag.score_delta": delta,
                "rag.retry_count": retry_count,
                "rag.retry": should_retry,
            }
        )
        if should_retry:
            emit_event(
                {
                    "event": "rag.confidence.retry",
                    "top_score": top_score,
                    "score_delta": delta,
                    "retry_count": retry_count,
                }
            )
        return {"retry": should_retry, "retry_count": retry_count}

    @observe_span(name="rag.compose")
    def _compose_step(
        graph_state: RetrievalAugmentedGenerationState,
    ) -> dict[str, Any]:
        working_state = graph_state["state"]
        retrieval_output = graph_state.get("retrieval_output")
        context = graph_state["context"]

        snippets_value = working_state.get("snippets")
        if isinstance(snippets_value, list):
            snippets = [
                snippet for snippet in snippets_value if isinstance(snippet, Mapping)
            ]
        else:
            snippets = []
        question_value = working_state.get("question")
        question = question_value if isinstance(question_value, str) else None
        stream_callback = graph_state["meta"].get("stream_callback")
        compose_params = compose.ComposeInput(
            question=question,
            snippets=snippets,
            stream_callback=stream_callback if callable(stream_callback) else None,
        )
        compose_result = compose_node(context, compose_params)

        retrieval_meta: dict[str, Any] = {}
        if retrieval_output is not None:
            retrieval_meta = retrieval_output.meta.model_dump(
                mode="json", exclude_none=True
            )
        elif isinstance(working_state.get("retrieval"), Mapping):
            retrieval_meta = dict(working_state.get("retrieval", {}))

        retrieval_override = compose_result.retrieval
        if retrieval_override is not None:
            retrieval_payload: Mapping[str, Any] = retrieval_override
        else:
            retrieval_payload = working_state.get("retrieval", retrieval_meta)
        if not isinstance(retrieval_payload, Mapping):
            retrieval_payload = retrieval_meta
        retrieval_payload = dict(retrieval_payload)

        took_ms_value = retrieval_payload.get("took_ms", retrieval_meta.get("took_ms"))
        try:
            retrieval_payload["took_ms"] = int(took_ms_value)
        except (TypeError, ValueError, OverflowError):
            retrieval_payload["took_ms"] = retrieval_meta.get("took_ms", 0)

        fallback_matches: list[Mapping[str, Any]] = []
        if retrieval_output is not None:
            fallback_matches = retrieval_output.matches
        elif isinstance(working_state.get("matches"), list):
            fallback_matches = cast(
                list[Mapping[str, Any]], working_state.get("matches")
            )

        snippets_override = compose_result.snippets
        if snippets_override is not None:
            snippets_payload = snippets_override
        else:
            snippets_payload = working_state.get("snippets", fallback_matches)
        snippets_payload = _normalise_snippets(snippets_payload, fallback_matches)

        guardrail_result = answer_guardrails.evaluate_answer_guardrails(
            [snippet for snippet in snippets_payload if isinstance(snippet, Mapping)]
        )
        retrieval_payload["answer_guardrail"] = guardrail_result.to_dict()
        if not guardrail_result.allowed:
            logger.warning(
                "rag.answer_guardrail.failed",
                extra={
                    "reason": guardrail_result.reason,
                    "snippet_count": guardrail_result.snippet_count,
                    "top_score": guardrail_result.top_score,
                },
            )
        update_observation(
            metadata={
                "rag.guardrail.allowed": guardrail_result.allowed,
                "rag.guardrail.reason": guardrail_result.reason,
                "rag.guardrail.snippet_count": guardrail_result.snippet_count,
                "rag.guardrail.top_score": guardrail_result.top_score,
            }
        )

        result_payload = RetrievalAugmentedGenerationOutput(
            answer=compose_result.answer,
            prompt_version=compose_result.prompt_version,
            retrieval=retrieval_payload,
            snippets=snippets_payload,
            reasoning=compose_result.reasoning,
            used_sources=compose_result.used_sources or [],
            suggested_followups=compose_result.suggested_followups or [],
            debug_meta=compose_result.debug_meta,
        ).model_dump(mode="json")

        if result_payload["prompt_version"] is None:
            logger.warning(
                "rag.compose.missing_prompt_version",
                extra={
                    "tenant_id": context.scope.tenant_id,
                    "case_id": context.business.case_id,
                    "graph": getattr(context, "graph_name", "rag.default"),
                },
            )

        if isinstance(working_state, MutableMapping):
            working_state["answer"] = compose_result.answer
            working_state["retrieval"] = retrieval_payload
            working_state["snippets"] = snippets_payload
            if compose_result.reasoning is not None:
                working_state["reasoning"] = compose_result.reasoning.model_dump(
                    mode="json"
                )
            if compose_result.used_sources is not None:
                working_state["used_sources"] = [
                    source.model_dump(mode="json")
                    for source in compose_result.used_sources
                ]
            if compose_result.suggested_followups is not None:
                working_state["suggested_followups"] = list(
                    compose_result.suggested_followups
                )
            if compose_result.debug_meta is not None:
                working_state["debug_meta"] = dict(compose_result.debug_meta)
        graph_input = graph_state.get("graph_input") or _coerce_graph_input(
            working_state
        )
        base_query = _resolve_base_query(graph_input)
        if base_query and compose_result.used_sources:
            try:
                enqueue_used_source_feedback(
                    context=context,
                    query_text=base_query,
                    snippets=snippets_payload,
                    used_sources=compose_result.used_sources,
                )
            except Exception:
                pass
        if base_query and compose_result.answer:
            cache = semantic_cache.get_semantic_cache()
            cache.store(
                base_query,
                context,
                result_payload,
                embedding=graph_state.get("cache_embedding"),
            )
            emit_event(
                {
                    "event": "rag.cache.store",
                    "tenant_id": context.scope.tenant_id,
                    "case_id": context.business.case_id,
                    "collection_id": context.business.collection_id,
                    "workflow_id": context.business.workflow_id,
                }
            )

        return {"state": working_state, "result": result_payload}

    workflow = StateGraph(RetrievalAugmentedGenerationState)
    workflow.add_node("contextualize", _contextualize_step)
    workflow.add_node("cache_lookup", _cache_lookup_step)
    workflow.add_node("cache_finalize", _cache_finalize_step)
    workflow.add_node("transform", _transform_step)
    workflow.add_node("retrieve", _retrieve_step)
    workflow.add_node("rerank", _rerank_step)
    workflow.add_node("confidence", _confidence_step)
    workflow.add_node("compose", _compose_step)
    workflow.set_entry_point("contextualize")

    workflow.add_edge("contextualize", "cache_lookup")

    def _route_after_cache(state: RetrievalAugmentedGenerationState) -> str:
        if state.get("cache_hit"):
            return "cache_finalize"
        return "transform"

    workflow.add_conditional_edges("cache_lookup", _route_after_cache)
    workflow.add_edge("cache_finalize", END)
    workflow.add_edge("transform", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "confidence")

    def _route_after_confidence(state: RetrievalAugmentedGenerationState) -> str:
        if state.get("retry"):
            return "transform"
        return "compose"

    workflow.add_conditional_edges("confidence", _route_after_confidence)
    workflow.add_edge("compose", END)

    graph = workflow.compile()
    setattr(graph, "io_spec", RAG_GRAPH_IO)
    return graph


@dataclass(frozen=True)
class RetrievalAugmentedGenerationGraph:
    """Graph executing the production RAG workflow (retrieve → compose).

    MVP 2025-10 — Breaking Contract v2: Response enthält answer, prompt_version, retrieval, snippets.
    """

    retrieve_node: RetrieveNode = retrieve.run
    compose_node: ComposeNode = compose.run
    io_spec: GraphIOSpec = RAG_GRAPH_IO
    _graph: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_graph",
            _build_compiled_graph(
                retrieve_node=self.retrieve_node,
                compose_node=self.compose_node,
            ),
        )

    def run(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        working_state = _ensure_mutable_state(state)
        working_meta = _ensure_mutable_meta(meta)
        context = _build_tool_context(working_meta)

        # M-5: RAG Graph manages history internally via Checkpointer
        thread_id = context.business.thread_id
        checkpointer = ThreadAwareCheckpointer()

        # We construct a GraphContext to access the checkpointer store
        # Note: We use "rag.default" as canonical name for now, or use meta?
        # Ideally this should match registry name logic, but file checkpointer
        # paths are derived from tenant/case/thread.
        graph_ctx = GraphContext(
            tool_context=context,
            graph_name="rag.default",
            graph_version=RAG_IO_VERSION_STRING,
        )

        # 1. Load History (if thread exists)
        if thread_id:
            try:
                loaded = checkpointer.load(graph_ctx)
                history = _load_history(loaded)
                # Inject history into working state for the graph execution
                working_state["chat_history"] = history
            except Exception:
                # Log but proceed (resilience)
                logger.warning(
                    "rag.history.load_failed",
                    extra={
                        "tenant_id": context.scope.tenant_id,
                        "thread_id": thread_id,
                    },
                )

        graph_state: RetrievalAugmentedGenerationState = {
            "state": working_state,
            "meta": working_meta,
            "context": context,
        }

        # 2. Invoke Graph
        final_state = self._graph.invoke(graph_state)

        updated_state = final_state.get("state", working_state)
        result_payload = final_state.get("result") or {}
        if not isinstance(result_payload, Mapping):
            result_payload = {}

        # 3. Save History (if thread exists and answer generated)
        if thread_id:
            try:
                base_query = _resolve_base_query(
                    final_state.get("graph_input") or _coerce_graph_input(updated_state)
                )
                answer = result_payload.get("answer")

                if base_query and answer:
                    history = _load_history(
                        updated_state
                    )  # Get current history from state
                    _append_history(history, role="user", content=base_query)
                    _append_history(history, role="assistant", content=answer)

                    limit = _resolve_history_limit()
                    history = _trim_history(history, limit=limit)

                    checkpointer.save(graph_ctx, {"chat_history": history})
            except Exception:
                logger.warning(
                    "rag.history.save_failed",
                    extra={
                        "tenant_id": context.scope.tenant_id,
                        "thread_id": thread_id,
                    },
                )

        return updated_state, result_payload


def build_graph() -> RetrievalAugmentedGenerationGraph:
    """Return a retrieval augmented generation graph instance."""

    return RetrievalAugmentedGenerationGraph()


def run(
    state: Mapping[str, Any] | MutableMapping[str, Any],
    meta: Mapping[str, Any] | MutableMapping[str, Any],
) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
    """Module-level convenience delegating to :func:`build_graph`.

    MVP 2025-10 — Breaking Contract v2: Response enthält answer, prompt_version, retrieval, snippets.
    """

    return build_graph().run(state, meta)


__all__ = ["RetrievalAugmentedGenerationGraph", "build_graph", "run"]
