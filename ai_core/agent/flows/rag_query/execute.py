from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel

from ai_core.agent.capabilities import registry as capability_registry
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.agent.flows.rag_query.contract import InputModel
from ai_core.tool_contracts.base import ToolContext

_FIXED_TS = "2026-01-01T00:00:00Z"


def _capability_event(
    run_id: str, name: str, status: str, phase: str
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ts": _FIXED_TS,
        "kind": phase,
        "status": status,
        "capability_name": name,
    }


def execute(
    tool_context: ToolContext,
    runtime_config: RuntimeConfig,
    flow_input: BaseModel | Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(flow_input, BaseModel):
        payload = flow_input.model_dump()
    else:
        payload = dict(flow_input)
    validated = InputModel.model_validate(payload)

    events: list[dict[str, Any]] = []

    events.append(
        _capability_event(run_id, "rag.retrieve", "started", "capability_start")
    )
    retrieval = capability_registry.execute(
        "rag.retrieve",
        tool_context,
        runtime_config,
        {"query": validated.question, "top_k": validated.top_k},
    )
    events.append(
        _capability_event(run_id, "rag.retrieve", "succeeded", "capability_end")
    )

    snippets = [
        {
            "text": match.get("text", ""),
            "source": match.get("source_id") or match.get("chunk_id"),
        }
        for match in retrieval.matches
    ]

    events.append(
        _capability_event(run_id, "rag.compose", "started", "capability_start")
    )
    composed = capability_registry.execute(
        "rag.compose",
        tool_context,
        runtime_config,
        {"question": validated.question, "snippets": snippets},
    )
    events.append(
        _capability_event(run_id, "rag.compose", "succeeded", "capability_end")
    )

    citations = [
        {
            "id": match.get("source_id") or match.get("chunk_id"),
            "snippet": match.get("text", ""),
            "score": match.get("score"),
        }
        for match in retrieval.matches
    ]

    events.append(
        _capability_event(run_id, "rag.evidence", "started", "capability_start")
    )
    evidence = capability_registry.execute(
        "rag.evidence",
        tool_context,
        runtime_config,
        {"answer": composed.answer, "citations": citations},
    )
    events.append(
        _capability_event(run_id, "rag.evidence", "succeeded", "capability_end")
    )

    output = {
        "answer": composed.answer,
        "citations": citations,
        "claim_to_citation": evidence.claim_to_citation,
        "retrieval_matches": retrieval.matches,
    }

    return output, events
