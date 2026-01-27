from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class AgentState(BaseModel):
    flow_name: str
    flow_version: str
    decision_log: list[dict[str, Any]]
    plan: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "AgentState":
        return cls.model_validate_json(data)


__all__ = ["AgentState"]
