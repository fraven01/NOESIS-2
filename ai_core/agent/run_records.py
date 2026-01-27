from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ai_core.agent.decision_log import StopDecision


class AgentRunRecord(BaseModel):
    run_id: str
    terminal_decision: StopDecision

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


__all__ = ["AgentRunRecord"]
