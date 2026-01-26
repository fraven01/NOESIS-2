"""Plan contract models for agentic workflows."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_core.contracts.plans.evidence import Evidence

SCHEMA_VERSION_PATTERN = r"^(v0|\d+\.\d+\.\d+)$"
PLAN_KEY_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "noesis.plan")


def _normalize_gremium_identifier(value: str) -> str:
    return " ".join(value.strip().lower().split())


class PlanScope(BaseModel):
    """Scope tuple used for deterministic plan key derivation."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    tenant_id: str
    gremium_identifier: str
    framework_profile_id: str | None = None
    framework_profile_version: str | None = None
    case_id: str | None = None
    workflow_id: str
    run_id: str

    @model_validator(mode="after")
    def _validate_scope(self) -> "PlanScope":
        for field_name in ("tenant_id", "gremium_identifier", "workflow_id", "run_id"):
            value = getattr(self, field_name)
            if not value:
                raise ValueError(f"{field_name} must be provided")

        has_profile_id = bool(self.framework_profile_id)
        has_profile_version = bool(self.framework_profile_version)
        if has_profile_id == has_profile_version:
            raise ValueError(
                "Exactly one of framework_profile_id or framework_profile_version is required"
            )
        return self


def _profile_ref(scope: PlanScope) -> str:
    if scope.framework_profile_id:
        return f"id:{scope.framework_profile_id}"
    if scope.framework_profile_version:
        return f"version:{scope.framework_profile_version}"
    raise ValueError("framework_profile_id or framework_profile_version is required")


def plan_scope_tuple(scope: PlanScope) -> tuple[str, ...]:
    """Return the ordered scope tuple used to derive plan keys."""

    return (
        scope.tenant_id,
        _normalize_gremium_identifier(scope.gremium_identifier),
        _profile_ref(scope),
        scope.case_id or "",
        scope.workflow_id,
        scope.run_id,
    )


def derive_plan_key(scope: PlanScope) -> str:
    """Return a deterministic plan key derived from the plan scope tuple."""

    canonical = json.dumps(
        plan_scope_tuple(scope), separators=(",", ":"), ensure_ascii=True
    )
    return str(uuid.uuid5(PLAN_KEY_NAMESPACE, canonical))


class Slot(BaseModel):
    """Structured slot value with optional provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    key: str
    status: str = Field(default="pending")
    value: Any | None = None
    slot_type: str | None = None
    json_schema_ref: str | None = None
    provenance: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_slot_type(self) -> "Slot":
        if self.slot_type and self.json_schema_ref:
            raise ValueError("slot_type and json_schema_ref are mutually exclusive")
        return self


class Task(BaseModel):
    """Ordered task with preconditions and evidence outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    key: str
    status: str = Field(default="pending")
    description: str | None = None
    preconditions: list[str] = Field(default_factory=list)
    outputs: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Gate(BaseModel):
    """Gate decision that may require review."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    key: str
    status: str = Field(default="pending")
    required: bool = False
    rationale: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Deviation(BaseModel):
    """Recorded deviation from the plan with rationale."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    key: str
    status: str = Field(default="pending")
    summary: str
    rationale: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImplementationPlan(BaseModel):
    """Plan schema representing tasks, gates, and evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    schema_version: str = Field(default="v0", pattern=SCHEMA_VERSION_PATTERN)
    plan_key: str
    scope: PlanScope
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    slots: list[Slot] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    gates: list[Gate] = Field(default_factory=list)
    deviations: list[Deviation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_plan_key(self) -> "ImplementationPlan":
        expected = derive_plan_key(self.scope)
        if self.plan_key != expected:
            raise ValueError("plan_key must match the derived plan scope key")
        return self


__all__ = [
    "SCHEMA_VERSION_PATTERN",
    "PLAN_KEY_NAMESPACE",
    "PlanScope",
    "Slot",
    "Task",
    "Gate",
    "Deviation",
    "ImplementationPlan",
    "plan_scope_tuple",
    "derive_plan_key",
]
