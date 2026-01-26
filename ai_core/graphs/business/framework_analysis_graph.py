"""Legacy wrapper for the framework analysis graph (LangGraph-backed)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from ai_core.services.framework_analysis_capabilities import call_llm_json_prompt
from ai_core.graphs.technical import rag_retrieval
from ai_core.graphs.business.framework_analysis.graph import (
    FrameworkAnalysisStateGraph,
)
from ai_core.graphs.business.framework_analysis.io import (
    FRAMEWORK_ANALYSIS_IO,
    FRAMEWORK_ANALYSIS_IO_VERSION_STRING,
    FRAMEWORK_ANALYSIS_SCHEMA_ID,
    FrameworkAnalysisGraphInput,
    FrameworkAnalysisGraphOutput,
)
from ai_core.tool_contracts import ToolContext
from ai_core.tools.framework_contracts import FrameworkAnalysisInput


@dataclass(frozen=True)
class FrameworkAnalysisGraph:
    """Compatibility wrapper matching the legacy API."""

    retrieval_service: Any | None = None
    llm_service: Any | None = None
    io_spec = FRAMEWORK_ANALYSIS_IO

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_graph",
            FrameworkAnalysisStateGraph(
                retrieval_service=self.retrieval_service or rag_retrieval.build_graph(),
                llm_service=self.llm_service or call_llm_json_prompt,
            ),
        )

    def run(
        self,
        *,
        context: ToolContext,
        input_params: FrameworkAnalysisInput,
    ) -> FrameworkAnalysisGraphOutput:
        """Execute the graph and return the boundary output."""
        graph_request = FrameworkAnalysisGraphInput(
            schema_id=FRAMEWORK_ANALYSIS_SCHEMA_ID,
            schema_version=FRAMEWORK_ANALYSIS_IO_VERSION_STRING,
            input=input_params,
            tool_context=context,
        )
        state = graph_request.model_dump(mode="json")
        meta = {
            "tool_context": context.model_dump(mode="json", exclude_none=True),
            "scope_context": context.scope.model_dump(mode="json", exclude_none=True),
            "business_context": context.business.model_dump(
                mode="json", exclude_none=True
            ),
        }
        _, result = self._graph.run(state, meta)
        return FrameworkAnalysisGraphOutput.model_validate(result)

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Execute the graph via the boundary contract."""
        _, result = self._graph.run(state, {})
        return result


def build_graph(
    *,
    retrieval_service: Any | None = None,
    llm_service: Any | None = None,
) -> FrameworkAnalysisGraph:
    """Build and return a framework analysis graph instance."""
    return FrameworkAnalysisGraph(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
    )


def run(
    state: Mapping[str, Any], meta: Mapping[str, Any]
) -> tuple[MutableMapping[str, Any], Mapping[str, Any]]:
    """GraphRunner adapter for registry execution."""
    graph = build_graph()
    result = graph.invoke(state)
    return dict(state), result


__all__ = ["FrameworkAnalysisGraph", "build_graph", "run"]
