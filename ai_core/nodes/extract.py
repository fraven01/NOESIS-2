from __future__ import annotations

from typing import Any, Dict

from ai_core.nodes._prompt_runner import run_prompt_node
from ai_core.tool_contracts import ToolContext
from pydantic import BaseModel, ConfigDict, Field


class ExtractInput(BaseModel):
    """Structured input parameters for the extract node."""

    text: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExtractOutput(BaseModel):
    """Structured output payload returned by the extract node."""

    items: str
    prompt_version: str | None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


def run(
    context: ToolContext,
    params: ExtractInput,
) -> ExtractOutput:
    """Extract items and facts from text using the LLM."""
    result = run_prompt_node(
        trace_name="extract",
        prompt_alias="extract/items",
        llm_label="extract",
        context=context,
        text=params.text,
        metadata=context.metadata,
    )
    return ExtractOutput(
        items=str(result.value),
        prompt_version=result.prompt_version,
        metadata=result.metadata,
    )
