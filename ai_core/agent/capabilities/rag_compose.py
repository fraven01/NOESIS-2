from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_core.agent.capabilities.base import CapabilitySpec
from ai_core.agent.capabilities.registry import register_capability
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.nodes import compose
from ai_core.tool_contracts.base import ToolContext


class RagComposeInput(BaseModel):
    question: str
    snippets: list[dict[str, Any]]
    constraints: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="after")
    def non_empty_question(self) -> "RagComposeInput":
        if not self.question.strip():
            raise ValueError("question must not be empty")
        return self


class RagComposeOutput(BaseModel):
    answer: str
    used_sources: list[str] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _compose(
    tool_context: ToolContext, params: RagComposeInput
) -> compose.ComposeOutput:
    node_input = compose.ComposeInput(
        question=params.question, snippets=params.snippets
    )
    return compose.run(tool_context, node_input)


def execute(
    tool_context: ToolContext,
    runtime_config: RuntimeConfig,
    input_model: RagComposeInput,
) -> RagComposeOutput:
    _ = runtime_config
    result = _compose(tool_context, input_model)
    telemetry = {}
    if hasattr(result, "debug_meta") and result.debug_meta is not None:
        telemetry = result.debug_meta
    used_sources = list(getattr(result, "used_sources", []) or [])
    return RagComposeOutput(
        answer=result.answer, used_sources=used_sources, telemetry=telemetry
    )


IO_SPEC = GraphIOSpec(
    schema_id="capability.rag.compose",
    version=GraphIOVersion(major=0, minor=1, patch=0),
    input_model=RagComposeInput,
    output_model=RagComposeOutput,
)

CAPABILITY = CapabilitySpec(
    name="rag.compose",
    version="0.1.0",
    io_spec_version=IO_SPEC.version_string,
    input_model=RagComposeInput,
    output_model=RagComposeOutput,
    execute=execute,
    io_spec=IO_SPEC,
)

register_capability(CAPABILITY)


__all__ = ["RagComposeInput", "RagComposeOutput", "execute", "CAPABILITY"]
