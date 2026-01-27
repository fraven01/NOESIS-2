from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_core.agent.capabilities.base import CapabilitySpec
from ai_core.agent.capabilities.registry import register_capability
from ai_core.agent.runtime import get_runtime_dependencies
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.nodes import retrieve
from ai_core.tool_contracts.base import ToolContext


class RagRetrieveInput(BaseModel):
    query: str
    top_k: int = 10
    filters: dict[str, Any] | None = None
    retrieval_plan: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def clamp_top_k(cls, data: object) -> object:
        if isinstance(data, Mapping):
            payload = dict(data)
            top_k = payload.get("top_k", 10)
            try:
                value = int(top_k)
            except (TypeError, ValueError):
                return data
            if value < 1:
                payload["top_k"] = 1
            elif value > 50:
                payload["top_k"] = 50
            return payload
        return data


class RagRetrieveOutput(BaseModel):
    matches: list[dict[str, Any]]
    telemetry: dict[str, Any] = Field(default_factory=dict)
    routing: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _retrieve(
    tool_context: ToolContext,
    params: RagRetrieveInput,
) -> retrieve.RetrieveOutput:
    deps = get_runtime_dependencies()
    router = deps.get("vector_router")
    if router is None:
        raise RuntimeError("vector_router dependency missing")
    node_input = retrieve.RetrieveInput(
        query=params.query,
        filters=params.filters,
        top_k=params.top_k,
        hybrid={},
    )
    previous_router = retrieve._ROUTER
    retrieve._ROUTER = router
    try:
        return retrieve.run(tool_context, node_input)
    finally:
        retrieve._ROUTER = previous_router


def _normalize_match(match: Any) -> dict[str, Any]:
    if isinstance(match, Mapping):
        payload = dict(match)
    else:
        payload = match.__dict__.copy() if hasattr(match, "__dict__") else {}

    chunk_id = payload.get("chunk_id") or payload.get("source_id")
    source_id = payload.get("source_id") or payload.get("chunk_id")
    score = payload.get("score", 0.0)
    text = payload.get("text") or payload.get("snippet") or payload.get("content") or ""

    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "score": score,
        "text": text,
    }


def execute(
    tool_context: ToolContext,
    runtime_config: RuntimeConfig,
    input_model: RagRetrieveInput,
) -> RagRetrieveOutput:
    _ = runtime_config
    result = _retrieve(tool_context, input_model)
    matches = [_normalize_match(match) for match in result.matches]
    telemetry = {}
    routing = None
    if hasattr(result, "meta") and result.meta is not None:
        meta_payload = result.meta.model_dump()
        telemetry = meta_payload.get("telemetry", {})
        routing = meta_payload.get("routing")
    return RagRetrieveOutput(matches=matches, telemetry=telemetry, routing=routing)


IO_SPEC = GraphIOSpec(
    schema_id="capability.rag.retrieve",
    version=GraphIOVersion(major=0, minor=1, patch=0),
    input_model=RagRetrieveInput,
    output_model=RagRetrieveOutput,
)

CAPABILITY = CapabilitySpec(
    name="rag.retrieve",
    version="0.1.0",
    io_spec_version=IO_SPEC.version_string,
    input_model=RagRetrieveInput,
    output_model=RagRetrieveOutput,
    execute=execute,
    io_spec=IO_SPEC,
)

register_capability(CAPABILITY)


__all__ = ["RagRetrieveInput", "RagRetrieveOutput", "execute", "CAPABILITY"]
