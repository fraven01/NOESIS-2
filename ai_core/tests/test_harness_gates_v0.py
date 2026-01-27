from __future__ import annotations

from ai_core.agent.harness.gates import gate_artifact, gate_diff


def test_gate_artifact_fails_without_citations():
    artifact = {
        "run_id": "r",
        "inputs_hash": "h",
        "decision_log": [],
        "stop_decision": {"status": "succeeded"},
        "answer": {"text": "hello"},
        "citations": [],
        "claim_to_citation": {"claim": ["c1"]},
        "ungrounded_claims": [],
    }

    passed, reasons = gate_artifact(artifact)
    assert passed is False
    assert "citations_below_minimum" in reasons


def test_gate_diff_fails_when_runtime_claim_map_missing():
    diff = {
        "stop_status_runtime": "succeeded",
        "citations_count_runtime": 1,
        "has_claim_map_runtime": False,
        "ungrounded_claims_count_runtime": 0,
    }

    passed, reasons = gate_diff(diff)
    assert passed is False
    assert "runtime_claim_map_missing" in reasons
