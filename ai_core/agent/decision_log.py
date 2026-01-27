from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)


class StopDecision(BaseModel):
    status: str
    reason: str
    evidence_refs: list[str]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DecisionLogEntry(BaseModel):
    run_id: str
    event_id: str
    ts: str
    kind: str
    status: str
    tool_context_hash: str
    reason: str | None = None
    evidence_refs: list[str] | None = None
    stop_decision: StopDecision | None = None
    metadata: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("ts")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        try:
            normalized = value.replace("Z", "+00:00")
            datetime.fromisoformat(normalized)
        except (TypeError, ValueError) as exc:
            raise ValueError("ts must be an ISO-8601 timestamp string") from exc
        return value

    @field_validator("reason", "evidence_refs")
    @classmethod
    def validate_stop_fields_presence(
        cls, value: object, info: ValidationInfo
    ) -> object:
        field_name = info.field_name or "field"
        if value is None:
            return value
        if field_name == "evidence_refs" and not isinstance(value, list):
            raise ValueError("evidence_refs must be a list of strings")
        return value

    @model_validator(mode="after")
    def validate_stop_event(self) -> "DecisionLogEntry":
        if self.kind != "stop":
            return self
        if not self.reason:
            raise ValueError("stop events require reason")
        if self.evidence_refs is None:
            raise ValueError("stop events require evidence_refs")
        if self.stop_decision is None:
            raise ValueError("stop events require stop_decision")
        if self.stop_decision.reason != self.reason:
            raise ValueError("stop_decision.reason must match reason")
        if self.stop_decision.evidence_refs != self.evidence_refs:
            raise ValueError("stop_decision.evidence_refs must match evidence_refs")
        return self


__all__ = ["DecisionLogEntry", "StopDecision"]
