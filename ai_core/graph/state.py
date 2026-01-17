"""Persisted graph state envelope."""

from __future__ import annotations

from collections.abc import Mapping
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ai_core.contracts.plans import (
    Evidence,
    ImplementationPlan,
    PlanScope,
    derive_plan_key,
)
from ai_core.infra.object_store import read_json, sanitize_identifier
from ai_core.tool_contracts import ToolContext


class PersistedGraphState(BaseModel):
    """Standardized format for persisted graph state."""

    tool_context: ToolContext
    state: dict[str, Any]
    plan: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] | None = None
    graph_name: str
    graph_version: str = "v0"
    checkpoint_at: datetime

    model_config = ConfigDict(frozen=True)


def plan_state_path(tenant_id: str, plan_key: str) -> str:
    """Return the object store path for a persisted plan checkpoint."""
    safe_tenant = sanitize_identifier(tenant_id)
    safe_plan_key = sanitize_identifier(plan_key)
    return f"{safe_tenant}/workflow-executions/{safe_plan_key}/state.json"


def extract_plan_envelope(
    state: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None]:
    """Return validated plan payload, evidence payload, and derived plan_key."""
    plan_value = state.get("plan")
    if plan_value is None:
        return None, None, None
    try:
        plan = ImplementationPlan.model_validate(plan_value)
    except ValidationError:
        return None, None, None
    plan_payload = plan.model_dump(mode="json")
    evidence_payload = [item.model_dump(mode="json") for item in plan.evidence]
    return plan_payload, evidence_payload, plan.plan_key


def load_plan_from_key(
    *, tenant_id: str, plan_key: str
) -> tuple[ImplementationPlan, list[Evidence]] | None:
    """Load the latest persisted plan/evidence payload for a plan_key."""
    path = plan_state_path(tenant_id, plan_key)
    try:
        payload = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    try:
        persisted = PersistedGraphState.model_validate(payload)
    except ValidationError:
        return None
    plan_payload = persisted.plan or persisted.state.get("plan")
    if not plan_payload:
        return None
    plan = ImplementationPlan.model_validate(plan_payload)
    evidence_payload = persisted.evidence
    if evidence_payload is None:
        evidence_payload = [item.model_dump(mode="json") for item in plan.evidence]
    evidence = [Evidence.model_validate(item) for item in evidence_payload]
    return plan, evidence


def load_plan_from_scope(
    scope: PlanScope,
) -> tuple[ImplementationPlan, list[Evidence]] | None:
    """Load the latest persisted plan/evidence using a derived plan_key."""
    plan_key = derive_plan_key(scope)
    return load_plan_from_key(tenant_id=scope.tenant_id, plan_key=plan_key)
