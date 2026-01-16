from __future__ import annotations

from typing import Any, Dict

from ai_core.infra.mask_prompt import mask_prompt, mask_response
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.prompts import load
from ai_core.infra.observability import observe_span
from ai_core.llm import client
from ai_core.tool_contracts import ToolContext
from pydantic import BaseModel, ConfigDict, Field


class DraftBlocksInput(BaseModel):
    """Structured input parameters for the draft blocks node."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DraftBlocksOutput(BaseModel):
    """Structured output payload returned by the draft blocks node."""

    draft: str
    prompt_version: str | None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


def _build_llm_metadata(
    context: ToolContext,
    *,
    prompt_version: str,
) -> dict[str, Any]:
    metadata = {
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,
        "trace_id": context.scope.trace_id,
        "user_id": context.scope.user_id,
        "prompt_version": prompt_version,
    }
    key_alias = context.metadata.get("key_alias")
    if key_alias:
        metadata["key_alias"] = key_alias
    ledger_logger = context.metadata.get("ledger_logger")
    if ledger_logger:
        metadata["ledger_logger"] = ledger_logger
    return metadata


def run(
    context: ToolContext,
    params: DraftBlocksInput,
) -> DraftBlocksOutput:
    """Generate draft blocks using system and function prompts."""
    system = load("draft/system")
    functions = load("draft/functions")
    clause = load("draft/clause_standard")
    metadata = _build_llm_metadata(context, prompt_version=clause["version"])
    return _run(system, functions, clause, params, metadata=metadata)


@observe_span(name="draft_blocks")
def _run(
    system: Dict[str, str],
    functions: Dict[str, str],
    clause: Dict[str, str],
    params: DraftBlocksInput,
    *,
    metadata: Dict[str, Any],
) -> DraftBlocksOutput:
    combined = "\n".join([system["text"], functions["text"], clause["text"]])
    pii_config = get_pii_config()
    masked = mask_prompt(combined, config=pii_config)
    result = client.call("draft", masked, metadata)
    draft_text = mask_response(result["text"], config=pii_config)
    return DraftBlocksOutput(
        draft=draft_text,
        prompt_version=clause["version"],
        metadata={},
    )
