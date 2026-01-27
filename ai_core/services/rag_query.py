from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Tuple

from ai_core.agent.harness.diff import diff_artifacts
from ai_core.agent.harness.gates import gate_artifact
from ai_core.agent.runtime import AgentRuntime
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.graphs.technical.retrieval_augmented_generation import (
    RAG_IO_VERSION_STRING,
    RAG_SCHEMA_ID,
    run as run_retrieval_augmented_generation,
)
from ai_core.tool_contracts import ToolContext
from ai_core.rag.filter_spec import FilterSpec, build_filter_spec


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
        mode: str = "legacy",
        top_k: int | None = None,
    ) -> Tuple[MutableMapping[str, Any], Mapping[str, Any]]:
        """Run the RAG graph with the provided context and question."""
        if mode == "runtime":
            runtime = AgentRuntime()
            flow_input = {
                "question": question,
                "top_k": top_k or (graph_state or {}).get("top_k", 10),
            }
            return runtime.start(
                tool_context=tool_context,
                runtime_config=RuntimeConfig(execution_scope="TENANT"),
                flow_name="rag_query",
                flow_input=flow_input,
            )
        if mode == "compare":
            legacy_state, legacy_result = self.execute(
                tool_context=tool_context,
                question=question,
                hybrid=hybrid,
                chat_history=chat_history,
                graph_state=graph_state,
                mode="legacy",
                top_k=top_k,
            )
            runtime_record = self.execute(
                tool_context=tool_context,
                question=question,
                hybrid=hybrid,
                chat_history=chat_history,
                graph_state=graph_state,
                mode="runtime",
                top_k=top_k,
            )
            legacy_artifact = _artifact_from_legacy(legacy_state, legacy_result)
            runtime_artifact = _artifact_from_runtime(runtime_record)
            diff = diff_artifacts(legacy_artifact, runtime_artifact)
            passed, reasons = gate_artifact(runtime_artifact)
            return {
                "legacy": {"artifact": legacy_artifact},
                "runtime": {"artifact": runtime_artifact},
                "diff": diff,
                "gate": {"passed": passed, "reasons": reasons},
            }

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

        raw_filters = state.get("filters")
        if raw_filters is not None:
            if isinstance(raw_filters, FilterSpec):
                state["filters"] = raw_filters
            else:
                filter_mapping = raw_filters if isinstance(raw_filters, Mapping) else {}
                state["filters"] = build_filter_spec(
                    tenant_id=tool_context.scope.tenant_id,
                    case_id=tool_context.business.case_id,
                    collection_id=tool_context.business.collection_id,
                    document_id=tool_context.business.document_id,
                    document_version_id=tool_context.business.document_version_id,
                    raw_filters=filter_mapping,
                )

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


def _extract_answer(result: Mapping[str, Any]) -> str:
    return str(result.get("answer") or result.get("output", {}).get("answer", ""))


def _extract_citations(result: Mapping[str, Any]) -> list[Any]:
    return list(
        result.get("citations") or result.get("output", {}).get("citations", [])
    )


def _extract_claim_map(result: Mapping[str, Any]) -> dict[str, Any]:
    claim_map = result.get("claim_to_citation") or result.get("output", {}).get(
        "claim_to_citation", {}
    )
    return dict(claim_map) if isinstance(claim_map, Mapping) else {}


def _diff_summary(
    legacy_result: Mapping[str, Any], runtime_record: Mapping[str, Any]
) -> dict[str, Any]:
    legacy_answer = _extract_answer(legacy_result)
    runtime_answer = _extract_answer(runtime_record)
    legacy_citations = _extract_citations(legacy_result)
    runtime_citations = _extract_citations(runtime_record)
    legacy_claim_map = _extract_claim_map(legacy_result)
    runtime_claim_map = _extract_claim_map(runtime_record)

    return {
        "answer_text_equal": legacy_answer == runtime_answer,
        "citations_count_legacy": len(legacy_citations),
        "citations_count_runtime": len(runtime_citations),
        "has_claim_map_legacy": bool(legacy_claim_map),
        "has_claim_map_runtime": bool(runtime_claim_map),
    }


def _artifact_base(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "inputs_hash": "",
        "decision_log": [],
    }


def _artifact_from_runtime(runtime_record: Mapping[str, Any]) -> dict[str, Any]:
    output = runtime_record.get("output", {})
    retrieval_matches = output.get("retrieval_matches", [])
    return {
        **_artifact_base(runtime_record.get("run_id", "runtime")),
        "stop_decision": runtime_record.get("stop_decision", {}),
        "retrieval": {
            "matches_count": len(retrieval_matches),
            "matches_sample": retrieval_matches[:3],
        },
        "answer": {"text": output.get("answer", "")},
        "citations": output.get("citations", []),
        "claim_to_citation": output.get("claim_to_citation", {}),
    }


def _artifact_from_legacy(
    legacy_state: Mapping[str, Any], legacy_result: Mapping[str, Any]
) -> dict[str, Any]:
    _ = legacy_state
    return {
        **_artifact_base("legacy"),
        "stop_decision": {"status": "succeeded", "reason": "legacy completed"},
        "answer": {"text": legacy_result.get("answer", "")},
        "citations": legacy_result.get("citations", []),
        "claim_to_citation": legacy_result.get("claim_to_citation", {}),
    }
