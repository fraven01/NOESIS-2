from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.agent.decision_log import DecisionLogEntry, StopDecision
from ai_core.agent.run_records import AgentRunRecord


def _base_stop_payload() -> dict[str, object]:
    return {
        "run_id": "run-001",
        "event_id": "event-001",
        "ts": "2026-01-27T12:00:00Z",
        "kind": "stop",
        "status": "succeeded",
        "tool_context_hash": "hash-001",
        "reason": "complete",
        "evidence_refs": [],
        "stop_decision": {
            "status": "succeeded",
            "reason": "complete",
            "evidence_refs": [],
        },
    }


def test_decision_log_stop_requires_reason_and_evidence_refs():
    payload = _base_stop_payload()
    payload.pop("reason")
    with pytest.raises(ValidationError):
        DecisionLogEntry.model_validate(payload)

    payload = _base_stop_payload()
    payload.pop("evidence_refs")
    with pytest.raises(ValidationError):
        DecisionLogEntry.model_validate(payload)


def test_run_record_terminal_decision_equals_stop_event():
    stop_event = DecisionLogEntry.model_validate(_base_stop_payload())
    run_record = AgentRunRecord(
        run_id=stop_event.run_id,
        terminal_decision=stop_event.stop_decision,
    )

    assert run_record.terminal_decision == stop_event.stop_decision


def test_decision_log_deterministic_ordering():
    expected_order = [
        "run_id",
        "event_id",
        "ts",
        "kind",
        "status",
        "tool_context_hash",
        "reason",
        "evidence_refs",
        "stop_decision",
        "metadata",
    ]
    assert list(DecisionLogEntry.model_fields.keys()) == expected_order

    stop_decision = StopDecision(
        status="succeeded",
        reason="complete",
        evidence_refs=[],
    )
    payload = _base_stop_payload()
    payload["stop_decision"] = stop_decision.model_dump()
    entry = DecisionLogEntry.model_validate(payload)
    assert list(entry.model_dump().keys()) == expected_order
