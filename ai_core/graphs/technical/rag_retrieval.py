"""Retrieval-only graph with multi-query batching for internal RAG usage."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol, Sequence, Literal

from common.logging import get_logger

from ai_core.infra.observability import update_observation
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.nodes import retrieve
from ai_core.rag import rerank as rag_rerank
from ai_core.rag.query_planner import plan_query
from ai_core.tool_contracts import ToolContext


RAG_RETRIEVAL_SCHEMA_ID = "noesis.graphs.rag_retrieval"
RAG_RETRIEVAL_IO_VERSION = GraphIOVersion(major=1, minor=0, patch=0)
RAG_RETRIEVAL_IO_VERSION_STRING = RAG_RETRIEVAL_IO_VERSION.as_string()


class RagRetrievalGraphInput(BaseModel):
    """Boundary input model for the retrieval-only graph."""

    schema_id: Literal[RAG_RETRIEVAL_SCHEMA_ID] = RAG_RETRIEVAL_SCHEMA_ID
    schema_version: Literal[RAG_RETRIEVAL_IO_VERSION_STRING] = (
        RAG_RETRIEVAL_IO_VERSION_STRING
    )
    tool_context: ToolContext
    queries: list[str] = Field(..., min_length=1)
    retrieve: retrieve.RetrieveInput
    use_rerank: bool = False
    document_id: str | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class RagRetrievalGraphOutput(BaseModel):
    """Boundary output model for the retrieval-only graph."""

    schema_id: Literal[RAG_RETRIEVAL_SCHEMA_ID] = RAG_RETRIEVAL_SCHEMA_ID
    schema_version: Literal[RAG_RETRIEVAL_IO_VERSION_STRING] = (
        RAG_RETRIEVAL_IO_VERSION_STRING
    )
    matches: list[dict[str, Any]]
    snippets: list[dict[str, Any]]
    retrieval_meta: Mapping[str, Any]
    query_variants_used: list[str]
    rerank_meta: Mapping[str, Any] | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


RAG_RETRIEVAL_IO = GraphIOSpec(
    schema_id=RAG_RETRIEVAL_SCHEMA_ID,
    version=RAG_RETRIEVAL_IO_VERSION,
    input_model=RagRetrievalGraphInput,
    output_model=RagRetrievalGraphOutput,
)


logger = get_logger(__name__)


class RetrieveNode(Protocol):
    """Protocol describing the retrieve node callable."""

    def __call__(
        self,
        context: ToolContext,
        params: retrieve.RetrieveInput,
    ) -> retrieve.RetrieveOutput:
        """Execute retrieval and return the structured output payload."""


class RerankNode(Protocol):
    """Protocol describing the rerank callable."""

    def __call__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        query: str,
        context: ToolContext,
        *,
        top_k: int | None = None,
    ) -> rag_rerank.RerankResult:
        """Execute rerank and return the structured output payload."""


def _coerce_queries(queries: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for query in queries:
        if not isinstance(query, str):
            continue
        candidate = query.strip()
        if candidate:
            cleaned.append(candidate)
    if not cleaned:
        raise ValueError("queries must include at least one non-empty string")
    return cleaned


def _merge_filters(
    base_filters: Mapping[str, Any] | None, document_id: str | None
) -> dict[str, Any]:
    merged = dict(base_filters or {})
    if document_id:
        merged["id"] = document_id
    return merged


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


def _coerce_score(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
    took_ms = sum(int(out.meta.took_ms) for out in outputs)
    vector_candidates = sum(int(out.meta.vector_candidates) for out in outputs)
    lexical_candidates = sum(int(out.meta.lexical_candidates) for out in outputs)
    meta_payload["matches_returned"] = len(matches)
    meta_payload["took_ms"] = took_ms
    meta_payload["vector_candidates"] = vector_candidates
    meta_payload["lexical_candidates"] = lexical_candidates
    return meta_payload


def _coerce_int(value: object) -> int | None:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    return candidate


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_positive_int(value: object, *, default: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


def _normalize_reference_id(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None
    try:
        return str(uuid.UUID(candidate))
    except Exception:
        return candidate


def _extract_reference_ids(match: Mapping[str, Any]) -> list[str]:
    meta = match.get("meta")
    if not isinstance(meta, Mapping):
        return []
    raw = meta.get("reference_ids") or meta.get("references")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    references: list[str] = []
    for entry in raw:
        try:
            text = str(entry).strip()
        except Exception:
            text = ""
        if not text:
            continue
        normalized = _normalize_reference_id(text)
        if normalized:
            references.append(normalized)
    return references


def _collect_reference_ids(matches: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    collected: list[str] = []
    for match in matches:
        for ref_id in _extract_reference_ids(match):
            if ref_id in seen:
                continue
            seen.add(ref_id)
            collected.append(ref_id)
    return collected


@dataclass(frozen=True)
class RagRetrievalGraph:
    """Graph executing multi-query retrieval with optional rerank."""

    retrieve_node: RetrieveNode = retrieve.run
    rerank_node: RerankNode = rag_rerank.rerank_chunks
    io_spec: GraphIOSpec = RAG_RETRIEVAL_IO

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Execute the graph via the versioned boundary contract."""
        try:
            graph_input = RagRetrievalGraphInput.model_validate(state)
        except ValidationError as exc:
            msg = f"Invalid graph input: {exc.errors()}"
            return {"error": msg}

        output = self._run(graph_input)
        graph_output = RagRetrievalGraphOutput.model_validate(
            output.model_dump(mode="json")
        )
        return graph_output.model_dump(mode="json")

    def _run(self, graph_input: RagRetrievalGraphInput) -> RagRetrievalGraphOutput:
        context = graph_input.tool_context
        queries = _coerce_queries(graph_input.queries)
        document_id = graph_input.document_id
        plan = plan_query(
            queries[0],
            context=context,
            doc_class=graph_input.retrieve.doc_class,
            filters=graph_input.retrieve.filters,
        )
        if len(queries) > 1:
            plan = plan.model_copy(
                update={"queries": list(queries), "planner": "manual"}
            )
        elif plan.queries:
            queries = plan.queries

        outputs: list[retrieve.RetrieveOutput] = []
        errors: list[Exception] = []
        matches: list[Mapping[str, Any]] = []

        base_payload = graph_input.retrieve.model_dump(mode="json", exclude_none=True)
        for query in queries:
            params_payload = dict(base_payload)
            params_payload["query"] = query
            params_payload["filters"] = _merge_filters(
                params_payload.get("filters"), document_id
            )
            params = retrieve.RetrieveInput.model_validate(params_payload)
            try:
                retrieve_output = self.retrieve_node(context, params)
            except Exception as exc:
                errors.append(exc)
                continue
            outputs.append(retrieve_output)
            matches.extend(retrieve_output.matches)

        if not outputs:
            if errors:
                raise errors[0]
            raise ValueError("retrieval produced no results")

        deduped = _dedupe_matches(matches)
        retrieval_meta = _aggregate_retrieval_meta(outputs, deduped)
        retrieval_meta["query_plan"] = plan.model_dump(mode="json", exclude_none=True)
        try:
            logger.info(
                "rag.retrieval.query_plan",
                extra={
                    "planner": plan.planner,
                    "doc_type": plan.doc_type,
                    "queries": plan.queries,
                    "constraints": plan.constraints.model_dump(
                        mode="json", exclude_none=True
                    ),
                },
            )
            update_observation(
                metadata={
                    "rag.query_plan.planner": plan.planner,
                    "rag.query_plan.doc_type": plan.doc_type,
                    "rag.query_plan.query_count": len(plan.queries),
                }
            )
        except Exception:
            pass

        reference_meta: dict[str, Any] | None = None
        if _env_flag("RAG_REFERENCE_EXPANSION", default=False):
            ref_start = time.monotonic()
            reference_ids = _collect_reference_ids(deduped)
            ref_limit = _coerce_positive_int(
                os.getenv("RAG_REFERENCE_EXPANSION_LIMIT", "5"), default=5
            )
            reference_ids = reference_ids[:ref_limit]
            ref_matches: list[Mapping[str, Any]] = []
            ref_errors = 0
            if reference_ids:
                ref_query = " | ".join(queries)
                ref_top_k = _coerce_positive_int(
                    os.getenv("RAG_REFERENCE_EXPANSION_TOP_K", "3"), default=3
                )
                for ref_id in reference_ids:
                    params_payload = dict(base_payload)
                    params_payload["query"] = ref_query
                    ref_filters = dict(params_payload.get("filters") or {})
                    ref_filters["id"] = ref_id
                    params_payload["filters"] = ref_filters
                    params_payload["top_k"] = ref_top_k
                    params = retrieve.RetrieveInput.model_validate(params_payload)
                    try:
                        retrieve_output = self.retrieve_node(context, params)
                    except Exception:
                        ref_errors += 1
                        continue
                    ref_matches.extend(retrieve_output.matches)
            if ref_matches:
                deduped = _dedupe_matches([*deduped, *ref_matches])
            reference_meta = {
                "reference_ids": reference_ids,
                "reference_count": len(reference_ids),
                "expanded_matches": len(ref_matches),
                "errors": ref_errors,
                "took_ms": int((time.monotonic() - ref_start) * 1000),
            }
            retrieval_meta["reference_expansion"] = reference_meta
            try:
                logger.info(
                    "rag.retrieval.reference_expansion",
                    extra=reference_meta,
                )
                update_observation(
                    metadata={
                        "rag.reference_expansion.count": len(reference_ids),
                        "rag.reference_expansion.matches": len(ref_matches),
                    }
                )
            except Exception:
                pass

        top_k = _coerce_int(retrieval_meta.get("top_k_effective"))
        if top_k and top_k > 0:
            deduped = deduped[:top_k]
        retrieval_meta["matches_returned"] = len(deduped)

        rerank_meta: dict[str, Any] | None = None
        snippets = deduped
        if graph_input.use_rerank and deduped:
            rerank_query = " | ".join(queries)
            rerank_result = self.rerank_node(
                deduped,
                rerank_query,
                context,
                top_k=top_k,
            )
            snippets = rerank_result.chunks
            deduped = rerank_result.chunks
            rerank_meta = {
                "mode": rerank_result.mode,
                "prompt_version": rerank_result.prompt_version,
                "error": rerank_result.error,
                "scores": rerank_result.scores,
            }

        return RagRetrievalGraphOutput(
            matches=deduped,
            snippets=snippets,
            retrieval_meta=retrieval_meta,
            query_variants_used=queries,
            rerank_meta=rerank_meta,
        )


def build_graph() -> RagRetrievalGraph:
    """Build and return the retrieval-only graph instance."""
    return RagRetrievalGraph()


def run(
    state: Mapping[str, Any], meta: Mapping[str, Any]
) -> tuple[MutableMapping[str, Any], Mapping[str, Any]]:
    """GraphRunner adapter for registry execution."""
    graph = build_graph()
    result = graph.invoke(state)
    return dict(state), result


__all__ = ["RagRetrievalGraph", "build_graph", "run"]
