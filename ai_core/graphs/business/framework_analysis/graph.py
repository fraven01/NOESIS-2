"""LangGraph implementation for framework analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Tuple, Callable, Awaitable, TypedDict

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import asyncio

from langgraph.graph import END, StateGraph

from ai_core.graph.io import GraphIOSpec
from ai_core.infra.observability import emit_event, update_observation
from ai_core.tool_contracts import ToolContext
from ai_core.tools.framework_contracts import (
    FrameworkAnalysisDraft,
    FrameworkAnalysisDraftMetadata,
    FrameworkAnalysisInput,
    FrameworkStructure,
)
from common.logging import get_logger

from .io import (
    FRAMEWORK_ANALYSIS_IO,
    FrameworkAnalysisGraphInput,
    FrameworkAnalysisGraphOutput,
)
from .nodes import (
    assemble_profile_node,
    detect_type_node,
    init_and_fetch_node,
    locate_components_node,
    _error_payload,
    _get_ids,
)
from .protocols import FrameworkLLMService, FrameworkRetrievalService
from .state import FrameworkAnalysisState


logger = get_logger(__name__)


class FrameworkAnalysisGraphStateInput(TypedDict, total=False):
    """Boundary input keys for the framework analysis graph."""

    input: Mapping[str, Any]
    context: ToolContext
    retrieval_service: Any
    llm_service: Any
    component_queries: Mapping[str, Mapping[str, str]]
    runtime: Mapping[str, Any]
    tenant_id: str
    tenant_schema: str | None
    trace_id: str
    scope_context: Any
    document_collection_id: str
    document_id: str | None
    force_reanalysis: bool
    confidence_threshold: float
    transitions: list[Mapping[str, Any]]


class FrameworkAnalysisGraphStateOutput(TypedDict, total=False):
    """Boundary output keys for the framework analysis graph."""

    assembled_structure: Mapping[str, Any]
    missing_components: list[str]
    completeness_score: float
    hitl_required: bool
    hitl_reasons: list[str]
    errors: list[Mapping[str, Any]]
    agreement_type: str
    type_confidence: float
    gremium_name_raw: str
    gremium_identifier: str
    document_collection_id: str
    document_id: str | None
    confidence_threshold: float
    force_reanalysis: bool
    transitions: list[Mapping[str, Any]]


def _build_compiled_graph() -> Any:
    workflow = StateGraph(
        FrameworkAnalysisState,
        input_schema=FrameworkAnalysisGraphStateInput,
        output_schema=FrameworkAnalysisGraphStateOutput,
    )

    def _resolve_node_timeout(
        state: FrameworkAnalysisState, node_name: str
    ) -> float | None:
        runtime = state.get("runtime") or {}
        per_node = runtime.get("node_timeouts_s") or {}
        timeout = per_node.get(node_name)
        if timeout is None:
            timeout = runtime.get("node_timeout_s")
        try:
            timeout_val = float(timeout) if timeout is not None else None
        except (TypeError, ValueError):
            return None
        if timeout_val is None or timeout_val <= 0:
            return None
        return timeout_val

    def _run_with_timeout(
        func: Callable[[], dict[str, Any]], timeout_s: float | None
    ) -> dict[str, Any]:
        if timeout_s is None:
            return func()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_s)
            except FutureTimeout as exc:
                raise TimeoutError("node_timeout") from exc

    def _safe(node_name: str, func):
        def _wrapped(state: FrameworkAnalysisState) -> dict[str, Any]:
            try:
                timeout_s = _resolve_node_timeout(state, node_name)
                return _run_with_timeout(lambda: func(state), timeout_s)
            except Exception as exc:
                return {
                    "errors": [_error_payload(node_name, exc)],
                    "hitl_required": True,
                    "hitl_reasons": [f"{node_name}: error"],
                    "early_exit": True,
                }

        return _wrapped

    workflow.add_node("init_and_fetch", _safe("init_and_fetch", init_and_fetch_node))
    workflow.add_node("detect_type", _safe("detect_type", detect_type_node))
    workflow.add_node(
        "locate_components", _safe("locate_components", locate_components_node)
    )
    workflow.add_node(
        "assemble_profile", _safe("assemble_profile", assemble_profile_node)
    )

    workflow.set_entry_point("init_and_fetch")
    workflow.add_edge("init_and_fetch", "detect_type")

    def _route_after_detect(state: FrameworkAnalysisState) -> str:
        if state.get("early_exit"):
            return END
        return "locate_components"

    workflow.add_conditional_edges("detect_type", _route_after_detect)
    workflow.add_edge("locate_components", "assemble_profile")
    workflow.add_edge("assemble_profile", END)

    graph = workflow.compile()
    setattr(graph, "io_spec", FRAMEWORK_ANALYSIS_IO)
    return graph


def _build_component_queries() -> dict[str, dict[str, str]]:
    return {
        "kbv": {
            "systembeschreibung": "Systembeschreibung technische Beschreibung",
            "funktionsbeschreibung": "Funktionsbeschreibung Funktionen",
            "auswertungen": "KBV Auswertungen Berichte Reports",
            "zugriffsrechte": "Zugriffsrechte Berechtigungen Rollen",
        },
        "gbv": {
            "systembeschreibung": "Gesamtbetriebsvereinbarung Systembeschreibung",
            "funktionsbeschreibung": "Gesamtbetriebsvereinbarung Funktionsbeschreibung",
            "auswertungen": "GBV Auswertungen Berichte Reports",
            "zugriffsrechte": "Zugriffsrechte Berechtigungen Rollen",
        },
        "bv": {
            "systembeschreibung": "Betriebsvereinbarung Systembeschreibung",
            "funktionsbeschreibung": "Betriebsvereinbarung Funktionsbeschreibung",
            "auswertungen": "Betriebsvereinbarung Auswertungen Berichte",
            "zugriffsrechte": "Zugriffsrechte Berechtigungen Rollen",
        },
        "dv": {
            "systembeschreibung": "Dienstvereinbarung Systembeschreibung",
            "funktionsbeschreibung": "Dienstvereinbarung Funktionsbeschreibung",
            "auswertungen": "Dienstvereinbarung Auswertungen Berichte",
            "zugriffsrechte": "Zugriffsrechte Berechtigungen Rollen",
        },
        "other": {
            "systembeschreibung": "Systembeschreibung technische Beschreibung",
            "funktionsbeschreibung": "Funktionsbeschreibung Funktionen",
            "auswertungen": "Auswertungen Berichte Reports",
            "zugriffsrechte": "Zugriffsrechte Berechtigungen Rollen",
        },
    }


@dataclass(frozen=True)
class FrameworkAnalysisStateGraph:
    """LangGraph-backed framework analysis execution."""

    retrieval_service: FrameworkRetrievalService
    llm_service: FrameworkLLMService
    io_spec: GraphIOSpec = FRAMEWORK_ANALYSIS_IO
    _graph: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_graph", _build_compiled_graph())

    def run(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        """Execute the framework analysis graph via boundary contracts."""
        try:
            graph_input = FrameworkAnalysisGraphInput.model_validate(state)
        except Exception as exc:
            msg = f"Invalid graph input: {exc}"
            logger.error(msg)
            return dict(state), {"error": msg}

        runtime = graph_input.runtime or {}
        retrieval_service = runtime.get("retrieval_service") or self.retrieval_service
        llm_service = runtime.get("llm_service") or self.llm_service

        context = graph_input.tool_context
        input_params: FrameworkAnalysisInput = graph_input.input

        initial_state: FrameworkAnalysisState = {
            "input": input_params.model_dump(),
            "context": context,
            "retrieval_service": retrieval_service,
            "llm_service": llm_service,
            "runtime": runtime,
            "tenant_id": context.scope.tenant_id,
            "tenant_schema": context.scope.tenant_schema,
            "trace_id": context.scope.trace_id,
            "scope_context": context.scope,
            "document_collection_id": context.business.collection_id,
            "document_id": context.business.document_id,
            "force_reanalysis": input_params.force_reanalysis,
            "confidence_threshold": input_params.confidence_threshold,
            "component_queries": _build_component_queries(),
            "transitions": [],
        }
        ids = _get_ids(context)
        logger.info(
            "framework_graph_starting",
            extra={
                **ids,
                "tenant_schema": context.scope.tenant_schema,
                "document_collection_id": context.business.collection_id,
                "force_reanalysis": input_params.force_reanalysis,
            },
        )
        emit_event(
            {
                "event": "framework.graph_started",
                **ids,
            }
        )

        graph_timeout_s = runtime.get("graph_timeout_s")
        try:
            final_state: FrameworkAnalysisState = _run_graph_with_timeout(
                self._graph.invoke, initial_state, graph_timeout_s
            )
        except TimeoutError:
            timeout_state = _timeout_state(initial_state, "graph_timeout")
            output = _build_output(timeout_state)
            graph_output = FrameworkAnalysisGraphOutput.model_validate(
                output.model_dump(mode="json")
            )
            return dict(state), graph_output.model_dump(mode="json")

        output = _build_output(final_state)
        graph_output = FrameworkAnalysisGraphOutput.model_validate(
            output.model_dump(mode="json")
        )
        update_observation(
            metadata={
                "framework.completeness_score": output.completeness_score,
                "framework.hitl_required": output.hitl_required,
            }
        )
        logger.info(
            "framework_graph_completed",
            extra={
                **ids,
                "gremium_identifier": output.gremium_identifier,
                "completeness_score": output.completeness_score,
                "hitl_required": output.hitl_required,
                "nodes_executed": len(final_state.get("transitions", [])),
            },
        )
        emit_event(
            {
                "event": "framework.graph_completed",
                **ids,
                "gremium_identifier": output.gremium_identifier,
                "completeness_score": output.completeness_score,
                "hitl_required": output.hitl_required,
            }
        )
        return dict(state), graph_output.model_dump(mode="json")

    async def arun(
        self,
        state: Mapping[str, Any] | MutableMapping[str, Any],
        meta: Mapping[str, Any] | MutableMapping[str, Any],
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        """Async execution path for the framework analysis graph."""
        try:
            graph_input = FrameworkAnalysisGraphInput.model_validate(state)
        except Exception as exc:
            msg = f"Invalid graph input: {exc}"
            logger.error(msg)
            return dict(state), {"error": msg}

        runtime = graph_input.runtime or {}
        retrieval_service = runtime.get("retrieval_service") or self.retrieval_service
        llm_service = runtime.get("llm_service") or self.llm_service

        context = graph_input.tool_context
        input_params: FrameworkAnalysisInput = graph_input.input

        initial_state: FrameworkAnalysisState = {
            "input": input_params.model_dump(),
            "context": context,
            "retrieval_service": retrieval_service,
            "llm_service": llm_service,
            "runtime": runtime,
            "tenant_id": context.scope.tenant_id,
            "tenant_schema": context.scope.tenant_schema,
            "trace_id": context.scope.trace_id,
            "scope_context": context.scope,
            "document_collection_id": context.business.collection_id,
            "document_id": context.business.document_id,
            "force_reanalysis": input_params.force_reanalysis,
            "confidence_threshold": input_params.confidence_threshold,
            "component_queries": _build_component_queries(),
            "transitions": [],
        }

        graph_timeout_s = runtime.get("graph_timeout_s")
        try:
            final_state: FrameworkAnalysisState = await _run_graph_with_timeout_async(
                self._graph.ainvoke, initial_state, graph_timeout_s
            )
        except TimeoutError:
            timeout_state = _timeout_state(initial_state, "graph_timeout")
            output = _build_output(timeout_state)
            graph_output = FrameworkAnalysisGraphOutput.model_validate(
                output.model_dump(mode="json")
            )
            return dict(state), graph_output.model_dump(mode="json")

        output = _build_output(final_state)
        graph_output = FrameworkAnalysisGraphOutput.model_validate(
            output.model_dump(mode="json")
        )
        return dict(state), graph_output.model_dump(mode="json")


def _build_output(state: FrameworkAnalysisState) -> FrameworkAnalysisDraft:
    if "assembled_structure" not in state:
        from .nodes import _empty_framework_structure

        empty_structure, missing_components = _empty_framework_structure()
        state.setdefault("assembled_structure", empty_structure)
        state.setdefault("missing_components", missing_components)
        state.setdefault("completeness_score", 0.0)
        state.setdefault("hitl_required", True)
        state.setdefault("hitl_reasons", ["error_fallback"])

    agreement_type = state.get("agreement_type", "other")
    type_confidence = state.get("type_confidence", 0.0)
    gremium_name_raw = state.get("gremium_name_raw", "")
    gremium_identifier = state.get("gremium_identifier", "unknown")

    analysis_metadata = FrameworkAnalysisDraftMetadata(
        detected_type=agreement_type,
        type_confidence=type_confidence,
        gremium_name_raw=gremium_name_raw,
        gremium_identifier=gremium_identifier,
        completeness_score=state["completeness_score"],
        missing_components=state["missing_components"],
    )

    structure = FrameworkStructure(**state["assembled_structure"])
    partial_results = structure if state.get("errors") else None

    return FrameworkAnalysisDraft(
        gremium_identifier=gremium_identifier,
        gremium_name_raw=gremium_name_raw,
        agreement_type=agreement_type,
        document_collection_id=state["document_collection_id"],
        document_id=state.get("document_id"),
        structure=structure,
        completeness_score=state["completeness_score"],
        missing_components=state["missing_components"],
        hitl_required=state["hitl_required"],
        hitl_reasons=state.get("hitl_reasons", []),
        analysis_metadata=analysis_metadata,
        confidence_threshold=state["confidence_threshold"],
        force_reanalysis=state["force_reanalysis"],
        partial_results=partial_results,
        errors=state.get("errors", []),
    )


def _run_graph_with_timeout(
    func: Callable[[FrameworkAnalysisState], FrameworkAnalysisState],
    state: FrameworkAnalysisState,
    timeout_s: Any,
) -> FrameworkAnalysisState:
    try:
        timeout_val = float(timeout_s) if timeout_s is not None else None
    except (TypeError, ValueError):
        timeout_val = None
    if timeout_val is None or timeout_val <= 0:
        return func(state)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, state)
        try:
            return future.result(timeout=timeout_val)
        except FutureTimeout as exc:
            raise TimeoutError("graph_timeout") from exc


async def _run_graph_with_timeout_async(
    func: Callable[[FrameworkAnalysisState], Awaitable[FrameworkAnalysisState]],
    state: FrameworkAnalysisState,
    timeout_s: Any,
) -> FrameworkAnalysisState:
    try:
        timeout_val = float(timeout_s) if timeout_s is not None else None
    except (TypeError, ValueError):
        timeout_val = None
    if timeout_val is None or timeout_val <= 0:
        return await func(state)
    try:
        return await asyncio.wait_for(func(state), timeout=timeout_val)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("graph_timeout") from exc


def _timeout_state(
    state: FrameworkAnalysisState, message: str
) -> FrameworkAnalysisState:
    timeout_error = {
        "node": "graph",
        "message": message,
        "error_type": "TimeoutError",
    }
    state = dict(state)
    state.setdefault("errors", [])
    state["errors"] = list(state.get("errors", [])) + [timeout_error]
    state["hitl_required"] = True
    state["hitl_reasons"] = [message]
    state["early_exit"] = True
    return state


def build_graph(
    *,
    retrieval_service: FrameworkRetrievalService,
    llm_service: FrameworkLLMService,
) -> FrameworkAnalysisStateGraph:
    """Build a new framework analysis state graph."""
    return FrameworkAnalysisStateGraph(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
    )


__all__ = ["FrameworkAnalysisStateGraph", "build_graph"]
