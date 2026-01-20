"""Retrieval-only graph with multi-query batching for internal RAG usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol, Sequence, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.nodes import retrieve
from ai_core.rag import rerank as rag_rerank
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
        top_k = _coerce_int(retrieval_meta.get("top_k_effective"))
        if top_k and top_k > 0:
            deduped = deduped[:top_k]

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
