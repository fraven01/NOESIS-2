"""Production retrieval augmented generation graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from ai_core.nodes import compose, retrieve
from ai_core.rag.visibility import coerce_bool_flag
from ai_core.tool_contracts import ContextError, ToolContext


logger = logging.getLogger(__name__)


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
    tenant_raw = meta.get("tenant_id")
    tenant_text = str(tenant_raw or "").strip()
    if not tenant_text:
        raise ContextError(
            "tenant_id is required for retrieval graphs", field="tenant_id"
        )

    meta["tenant_id"] = tenant_text

    tenant_schema_raw = meta.get("tenant_schema")
    tenant_schema = (
        str(tenant_schema_raw).strip() if tenant_schema_raw is not None else None
    )

    case_raw = meta.get("case_id")
    case_id = str(case_raw).strip() if case_raw is not None else None
    if not case_id:
        raise ContextError("case_id is required for retrieval graphs", field="case_id")

    override_flag = meta.get("visibility_override_allowed")
    trace_raw = meta.get("trace_id")
    trace_id = str(trace_raw).strip() if trace_raw is not None else None

    return ToolContext(
        tenant_id=tenant_text,
        tenant_schema=tenant_schema,
        case_id=case_id,
        trace_id=trace_id,
        visibility_override_allowed=coerce_bool_flag(override_flag),
        metadata=dict(meta),
    )


@dataclass(frozen=True)
class RetrievalAugmentedGenerationGraph:
    """Graph executing the production RAG workflow (retrieve → compose).

    MVP 2025-10 — Breaking Contract v2: Response enthält answer, prompt_version, retrieval, snippets.
    """

    retrieve_node: RetrieveNode = retrieve.run
    compose_node: ComposeNode = compose.run

    def run(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        working_state = _ensure_mutable_state(state)
        working_meta = _ensure_mutable_meta(meta)

        context = _build_tool_context(working_meta)
        params = retrieve.RetrieveInput.from_state(working_state)
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

        result_payload = {
            "answer": compose_result.get("answer"),
            "prompt_version": compose_result.get("prompt_version"),
            "retrieval": retrieval_payload,
            "snippets": snippets_payload,
        }

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
