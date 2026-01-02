"""Production retrieval augmented generation graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Tuple
from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.nodes import compose, retrieve
from ai_core.tool_contracts import ContextError, ToolContext
from ai_core.tool_contracts.base import tool_context_from_meta
from pydantic import BaseModel, ConfigDict, ValidationError


logger = logging.getLogger(__name__)


RAG_SCHEMA_ID = "noesis.graphs.retrieval_augmented_generation"
RAG_IO_VERSION = GraphIOVersion(major=1, minor=0, patch=0)
RAG_IO_VERSION_STRING = RAG_IO_VERSION.as_string()


class RetrievalAugmentedGenerationInput(retrieve.RetrieveInput):
    """Boundary input model for the retrieval augmented generation graph."""

    schema_id: Literal[RAG_SCHEMA_ID] = RAG_SCHEMA_ID
    schema_version: Literal[RAG_IO_VERSION_STRING] = RAG_IO_VERSION_STRING

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
        self, state: MutableMapping[str, Any], meta: MutableMapping[str, Any]
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        """Execute composition and return the updated state and payload."""


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


@dataclass(frozen=True)
class RetrievalAugmentedGenerationGraph:
    """Graph executing the production RAG workflow (retrieve → compose).

    MVP 2025-10 — Breaking Contract v2: Response enthält answer, prompt_version, retrieval, snippets.
    """

    retrieve_node: RetrieveNode = retrieve.run
    compose_node: ComposeNode = compose.run
    io_spec: GraphIOSpec = RAG_GRAPH_IO

    def run(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        working_state = _ensure_mutable_state(state)
        working_meta = _ensure_mutable_meta(meta)

        try:
            graph_input = RetrievalAugmentedGenerationInput.model_validate(
                working_state
            )
        except ValidationError as exc:
            # This graph fails fast on invalid input instead of returning state errors.
            raise ValueError(f"Invalid graph input: {exc.errors()}") from exc

        context = _build_tool_context(working_meta)
        params = retrieve.RetrieveInput.model_validate(
            graph_input.model_dump(exclude={"schema_id", "schema_version"})
        )
        retrieve_output = self.retrieve_node(context, params)
        retrieval_meta = retrieve_output.meta.model_dump(mode="json", exclude_none=True)
        took_ms = retrieval_meta.get("took_ms")
        try:
            retrieval_meta["took_ms"] = int(took_ms)
        except (TypeError, ValueError, OverflowError):
            retrieval_meta["took_ms"] = int(
                getattr(retrieve_output.meta, "took_ms", 0) or 0
            )
        working_state["matches"] = retrieve_output.matches
        working_state["snippets"] = retrieve_output.matches
        working_state["retrieval"] = retrieval_meta

        final_state, compose_result = self.compose_node(working_state, working_meta)

        retrieval_payload = final_state.get("retrieval", retrieval_meta)
        if not isinstance(retrieval_payload, Mapping):
            retrieval_payload = retrieval_meta
        else:
            retrieval_payload = dict(retrieval_payload)

        took_ms_value = retrieval_payload.get("took_ms", retrieval_meta.get("took_ms"))
        try:
            retrieval_payload["took_ms"] = int(took_ms_value)
        except (TypeError, ValueError, OverflowError):
            retrieval_payload["took_ms"] = retrieval_meta.get("took_ms", 0)

        snippets_payload = final_state.get("snippets", retrieve_output.matches)
        snippets_payload = _normalise_snippets(
            snippets_payload, retrieve_output.matches
        )

        result_payload = RetrievalAugmentedGenerationOutput(
            answer=compose_result.get("answer"),
            prompt_version=compose_result.get("prompt_version"),
            retrieval=retrieval_payload,
            snippets=snippets_payload,
        ).model_dump(mode="json")

        if result_payload["prompt_version"] is None:
            logger.warning(
                "rag.compose.missing_prompt_version",
                extra={
                    "tenant_id": context.tenant_id,
                    "case_id": context.case_id,
                    "graph": getattr(context, "graph_name", "rag.default"),
                },
            )

        if isinstance(final_state, MutableMapping):
            final_state["retrieval"] = retrieval_payload
            final_state["snippets"] = snippets_payload

        return final_state, result_payload


GRAPH = RetrievalAugmentedGenerationGraph()


def build_graph() -> RetrievalAugmentedGenerationGraph:
    """Return the shared retrieval augmented generation graph instance."""

    return GRAPH


def run(
    state: Mapping[str, Any] | MutableMapping[str, Any],
    meta: Mapping[str, Any] | MutableMapping[str, Any],
) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
    """Module-level convenience delegating to :data:`GRAPH`.

    MVP 2025-10 — Breaking Contract v2: Response enthält answer, prompt_version, retrieval, snippets.
    """

    return GRAPH.run(state, meta)


__all__ = ["RetrievalAugmentedGenerationGraph", "GRAPH", "build_graph", "run"]
