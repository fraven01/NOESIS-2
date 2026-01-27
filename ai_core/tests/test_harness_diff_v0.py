from __future__ import annotations

from ai_core.agent.harness.diff import diff_artifacts


def test_diff_reports_citation_and_claim_map_metrics():
    a = {
        "run_id": "a",
        "inputs_hash": "h",
        "decision_log": [],
        "stop_decision": {"status": "succeeded"},
        "citations": [{"id": "c1"}, {"id": "c2"}],
        "claim_to_citation": {"claim": ["c1"]},
        "ungrounded_claims": ["u1"],
        "answer": {"text": "hello"},
    }
    b = {
        "run_id": "b",
        "inputs_hash": "h",
        "decision_log": [],
        "stop_decision": {"status": "failed"},
        "citations": [],
        "claim_to_citation": {},
        "answer": {"text": "hi"},
    }

    diff = diff_artifacts(a, b)

    assert diff["citations_count_a"] == 2
    assert diff["citations_count_b"] == 0
    assert diff["has_claim_map_a"] is True
    assert diff["has_claim_map_b"] is False
    assert diff["ungrounded_claims_count_a"] == 1
    assert diff["ungrounded_claims_count_b"] == 0
    assert diff["answer_length_a"] == 5
    assert diff["answer_length_b"] == 2


def test_diff_is_deterministic():
    a = {
        "run_id": "a",
        "inputs_hash": "h",
        "decision_log": [],
        "stop_decision": {"status": "succeeded"},
        "citations": [{"id": "c1"}],
        "claim_to_citation": {"claim": ["c1"]},
        "answer": {"text": "hello"},
    }
    b = {
        "run_id": "b",
        "inputs_hash": "h",
        "decision_log": [],
        "stop_decision": {"status": "succeeded"},
        "citations": [{"id": "c1"}],
        "claim_to_citation": {"claim": ["c1"]},
        "answer": {"text": "hello"},
    }

    diff1 = diff_artifacts(a, b)
    diff2 = diff_artifacts(a, b)

    assert diff1 == diff2
