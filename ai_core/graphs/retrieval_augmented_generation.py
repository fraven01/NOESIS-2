"""Production retrieval augmented generation graph."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from ai_core.nodes import compose, retrieve
from ai_core.rag.visibility import coerce_bool_flag
from ai_core.tool_contracts import ContextError, ToolContext


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
    """Graph executing the production RAG workflow (retrieve → compose)."""

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
        working_state["matches"] = retrieve_output.matches
        working_state["snippets"] = retrieve_output.matches

        final_state, result = self.compose_node(working_state, working_meta)
        return final_state, result


GRAPH = RetrievalAugmentedGenerationGraph()


def build_graph() -> RetrievalAugmentedGenerationGraph:
    """Return the shared retrieval augmented generation graph instance."""

    return GRAPH


def run(
    state: Mapping[str, Any] | MutableMapping[str, Any],
    meta: Mapping[str, Any] | MutableMapping[str, Any],
) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
    """Module-level convenience delegating to :data:`GRAPH`."""

    return GRAPH.run(state, meta)


__all__ = ["RetrievalAugmentedGenerationGraph", "GRAPH", "build_graph", "run"]
