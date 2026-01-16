from __future__ import annotations

from typing import Any, Dict

from ai_core.nodes._prompt_runner import run_prompt_node
from ai_core.tool_contracts import ToolContext
from pydantic import BaseModel, ConfigDict, Field


class ClassifyInput(BaseModel):
    """Structured input parameters for the classify node."""

    text: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class ClassifyOutput(BaseModel):
    """Structured output payload returned by the classify node."""

    classification: str
    prompt_version: str | None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


def run(
    context: ToolContext,
    params: ClassifyInput,
) -> ClassifyOutput:
    """Classify text regarding co-determination."""
    result = run_prompt_node(
        trace_name="classify",
        prompt_alias="classify/mitbestimmung",
        llm_label="classify",
        context=context,
        text=params.text,
        metadata=context.metadata,
    )
    return ClassifyOutput(
        classification=str(result.value),
        prompt_version=result.prompt_version,
        metadata=result.metadata,
    )
