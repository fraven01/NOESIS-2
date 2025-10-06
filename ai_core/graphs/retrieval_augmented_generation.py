"""Production retrieval augmented generation graph."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol, Tuple

from ai_core.nodes import compose, retrieve


class RetrieveNode(Protocol):
    """Protocol describing the retrieve node callable."""

    def __call__(
        self,
        state: MutableMapping[str, Any],
        meta: Mapping[str, Any],
        *,
        top_k: int | None = None,
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        """Execute retrieval and return the updated state and payload."""


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


def _ensure_tenant_id(meta: MutableMapping[str, Any]) -> str:
    tenant_id = meta.get("tenant_id") or meta.get("tenant")
    if tenant_id is None:
        raise ValueError("tenant_id is required for retrieval graphs")
    tenant_id = str(tenant_id)
    meta["tenant_id"] = tenant_id
    return tenant_id


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

        _ensure_tenant_id(working_meta)

        intermediate_state, _ = self.retrieve_node(working_state, working_meta)
        final_state, result = self.compose_node(intermediate_state, working_meta)
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
