from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Tuple

from ai_core.graphs.technical.retrieval_augmented_generation import (
    RAG_IO_VERSION_STRING,
    RAG_SCHEMA_ID,
    run as run_retrieval_augmented_generation,
)
from ai_core.tool_contracts import ToolContext


class RagQueryService:
    """Shared service for executing the retrieval-augmented generation graph."""

    def __init__(self, stream_callback: Callable[[str], None] | None = None) -> None:
        self._stream_callback = stream_callback

    def execute(
        self,
        *,
        tool_context: ToolContext,
        question: str,
        hybrid: Mapping[str, Any] | None = None,
        chat_history: list[Mapping[str, Any]] | None = None,
        graph_state: Mapping[str, Any] | None = None,
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        """Run the RAG graph with the provided context and question."""
        state: MutableMapping[str, Any] = {
            "schema_id": RAG_SCHEMA_ID,
            "schema_version": RAG_IO_VERSION_STRING,
            "question": question,
            "query": question,
            "hybrid": hybrid or {},
            "chat_history": list(chat_history or []),
        }
        if graph_state:
            state.update(graph_state)

        meta: MutableMapping[str, Any] = {
            "scope_context": tool_context.scope.model_dump(
                mode="json", exclude_none=True
            ),
            "business_context": tool_context.business.model_dump(
                mode="json", exclude_none=True
            ),
            "tool_context": tool_context.model_dump(mode="json", exclude_none=True),
        }

        if self._stream_callback:
            meta["stream_callback"] = self._stream_callback

        return run_retrieval_augmented_generation(state, meta)
