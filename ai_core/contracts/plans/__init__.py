"""Contracts for plan, blueprint, and evidence schemas."""

from ai_core.contracts.plans.blueprint import Blueprint, BlueprintGate, SlotSpec
from ai_core.contracts.plans.evidence import (
    Confidence,
    ConfidenceLabel,
    Evidence,
    EvidenceRefType,
)
from ai_core.contracts.plans.plan import (
    PLAN_KEY_NAMESPACE,
    Deviation,
    Gate,
    ImplementationPlan,
    PlanScope,
    Slot,
    Task,
    derive_plan_key,
    plan_scope_tuple,
)


def export_plan_schemas() -> dict[str, dict[str, object]]:
    """Return JSON schema exports for plan-related contracts."""

    return {
        "ImplementationPlan": ImplementationPlan.model_json_schema(),
        "Blueprint": Blueprint.model_json_schema(),
        "Evidence": Evidence.model_json_schema(),
    }


__all__ = [
    "PLAN_KEY_NAMESPACE",
    "PlanScope",
    "Slot",
    "Task",
    "Gate",
    "Deviation",
    "ImplementationPlan",
    "plan_scope_tuple",
    "derive_plan_key",
    "Blueprint",
    "BlueprintGate",
    "SlotSpec",
    "Evidence",
    "EvidenceRefType",
    "Confidence",
    "ConfidenceLabel",
    "export_plan_schemas",
]
