from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from ai_core.agent.flows.dummy_flow.contract import InputModel
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.tool_contracts.base import ToolContext


def execute(
    tool_context: ToolContext,
    runtime_config: RuntimeConfig,
    flow_input: BaseModel | Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    _ = run_id
    if isinstance(flow_input, BaseModel):
        payload = flow_input.model_dump()
    else:
        payload = dict(flow_input)
    validated = InputModel.model_validate(payload)
    return {"result": validated.query}
