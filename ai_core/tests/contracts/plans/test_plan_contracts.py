"""Tests for plan contract models."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from ai_core.contracts.plans import (
    Blueprint,
    BlueprintGate,
    Confidence,
    Deviation,
    Evidence,
    Gate,
    ImplementationPlan,
    PlanScope,
    Slot,
    SlotSpec,
    Task,
    derive_plan_key,
    export_plan_schemas,
)


def _build_scope(
    *,
    gremium_identifier: str = "Board A",
    framework_profile_id: str | None = "profile-1",
    framework_profile_version: str | None = None,
    case_id: str | None = "case-1",
) -> PlanScope:
    return PlanScope(
        tenant_id="tenant-1",
        gremium_identifier=gremium_identifier,
        framework_profile_id=framework_profile_id,
        framework_profile_version=framework_profile_version,
        case_id=case_id,
        workflow_id="workflow-1",
        run_id="run-1",
    )


def _build_evidence() -> Evidence:
    return Evidence(
        ref_type="url",
        ref_id="https://example.com/doc",
        summary="Primary source",
        confidence=Confidence(score=0.86),
    )


def test_plan_key_derivation_is_deterministic_and_normalized() -> None:
    scope_a = _build_scope(gremium_identifier="Board A")
    scope_b = _build_scope(gremium_identifier="  board   a ")
    scope_version = _build_scope(
        gremium_identifier="Board A",
        framework_profile_id=None,
        framework_profile_version="v1",
    )

    assert derive_plan_key(scope_a) == derive_plan_key(scope_b)
    assert derive_plan_key(scope_a) != derive_plan_key(scope_version)


def test_plan_scope_requires_single_framework_reference() -> None:
    with pytest.raises(ValueError, match="framework_profile"):
        _build_scope(framework_profile_id="profile-1", framework_profile_version="v1")
    with pytest.raises(ValueError, match="framework_profile"):
        _build_scope(framework_profile_id=None, framework_profile_version=None)


def test_slot_type_requires_single_reference() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Slot(
            key="source_url",
            slot_type="source_url",
            json_schema_ref="#/schemas/source-url",
        )


def test_slot_spec_type_requires_single_reference() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SlotSpec(
            key="source_url",
            slot_type="source_url",
            json_schema_ref="#/schemas/source-url",
        )


def test_plan_round_trip_serialization() -> None:
    scope = _build_scope()
    plan_key = derive_plan_key(scope)
    evidence = _build_evidence()
    plan = ImplementationPlan(
        plan_key=plan_key,
        scope=scope,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        slots=[
            Slot(
                key="source_url",
                status="filled",
                value="https://example.com/doc",
                slot_type="source_url",
                provenance=[evidence],
            )
        ],
        tasks=[
            Task(
                key="acquire_sources",
                status="completed",
                outputs=[evidence],
            )
        ],
        gates=[
            Gate(
                key="review_gate",
                status="approved",
                required=True,
                rationale="Auto-approved",
                evidence=[evidence],
            )
        ],
        deviations=[
            Deviation(
                key="missing_source",
                status="approved",
                summary="Fallback to secondary source",
                rationale="Primary source unavailable",
                evidence=[evidence],
            )
        ],
        evidence=[evidence],
    )

    payload = plan.model_dump(mode="json")
    restored = ImplementationPlan.model_validate(payload)
    assert restored.model_dump(mode="json") == payload


def test_plan_requires_matching_plan_key() -> None:
    scope = _build_scope()
    with pytest.raises(ValueError, match="plan_key"):
        ImplementationPlan(
            plan_key="mismatch",
            scope=scope,
            slots=[],
            tasks=[],
            gates=[],
            deviations=[],
            evidence=[],
        )


def test_blueprint_round_trip_serialization() -> None:
    blueprint = Blueprint(
        name="collection_search",
        version="0.1.0",
        slots=[
            SlotSpec(
                key="source_url",
                required=True,
                description="Primary source url",
                slot_type="source_url",
            )
        ],
        gates=[
            BlueprintGate(
                key="review_gate",
                description="Review before execution",
                criteria={"hitl_required": True},
            )
        ],
    )

    payload = blueprint.model_dump(mode="json")
    restored = Blueprint.model_validate(payload)
    assert restored.model_dump(mode="json") == payload


def test_exported_schemas_avoid_unapproved_ids() -> None:
    schemas = export_plan_schemas()
    assert "ImplementationPlan" in schemas
    assert "Blueprint" in schemas
    assert "Evidence" in schemas
    schema_text = json.dumps(schemas, sort_keys=True)
    assert "execution_case_id" not in schema_text
    assert "plan_id" not in schema_text
    assert '"format": "uuid"' not in schema_text
    assert "schema_version" in schemas["ImplementationPlan"]["properties"]
