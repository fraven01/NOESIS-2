from __future__ import annotations

from typing import Any, Dict

from ai_core.nodes._prompt_runner import run_prompt_node
from ai_core.tool_contracts import ToolContext
from pydantic import BaseModel, ConfigDict, Field


class AssessInput(BaseModel):
    """Structured input parameters for the assess node."""

    text: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class AssessOutput(BaseModel):
    """Structured output payload returned by the assess node."""

    risk: str
    prompt_version: str | None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


def run(
    context: ToolContext,
    params: AssessInput,
) -> AssessOutput:
    """Assess risks for the provided text."""
    result = run_prompt_node(
        trace_name="assess",
        prompt_alias="assess/risk",
        llm_label="analyze",
        context=context,
        text=params.text,
        metadata=context.metadata,
    )
    return AssessOutput(
        risk=str(result.value),
        prompt_version=result.prompt_version,
        metadata=result.metadata,
    )
