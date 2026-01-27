from __future__ import annotations

from typing import Any

DEFAULT_POLICY = {
    "require_stop_status": "succeeded",
    "require_claim_map": True,
    "max_ungrounded_claims": 2,
    "min_citations": 1,
}


def _artifact_answer_exists(artifact: dict[str, Any]) -> bool:
    answer = artifact.get("answer")
    if isinstance(answer, dict):
        text = answer.get("text")
        return isinstance(text, str) and bool(text.strip())
    return False


def gate_artifact(
    artifact: dict[str, Any], policy: dict | None = None
) -> tuple[bool, list[str]]:
    rules = DEFAULT_POLICY if policy is None else policy
    reasons: list[str] = []

    stop_status = (artifact.get("stop_decision") or {}).get("status")
    if rules.get("require_stop_status") and stop_status != rules["require_stop_status"]:
        reasons.append("stop_status_not_succeeded")

    citations = artifact.get("citations")
    citations_count = len(citations) if isinstance(citations, list) else 0
    if citations_count < int(rules.get("min_citations", 0)):
        reasons.append("citations_below_minimum")

    ungrounded = artifact.get("ungrounded_claims")
    ungrounded_count = len(ungrounded) if isinstance(ungrounded, list) else 0
    if ungrounded_count > int(rules.get("max_ungrounded_claims", 0)):
        reasons.append("ungrounded_claims_above_max")

    if rules.get("require_claim_map") and _artifact_answer_exists(artifact):
        claim_map = artifact.get("claim_to_citation")
        if not isinstance(claim_map, dict) or not claim_map:
            reasons.append("claim_map_missing_or_empty")

    return (len(reasons) == 0), reasons


def gate_diff(
    diff: dict[str, Any], policy: dict | None = None
) -> tuple[bool, list[str]]:
    rules = DEFAULT_POLICY if policy is None else policy
    reasons: list[str] = []

    if rules.get("require_stop_status"):
        if diff.get("stop_status_runtime") != rules["require_stop_status"]:
            reasons.append("runtime_stop_status_not_succeeded")

    if rules.get("require_claim_map"):
        if not diff.get("has_claim_map_runtime", False):
            reasons.append("runtime_claim_map_missing")

    if diff.get("citations_count_runtime", 0) < int(rules.get("min_citations", 0)):
        reasons.append("runtime_citations_below_minimum")

    if diff.get("ungrounded_claims_count_runtime", 0) > int(
        rules.get("max_ungrounded_claims", 0)
    ):
        reasons.append("runtime_ungrounded_claims_above_max")

    return (len(reasons) == 0), reasons


__all__ = ["DEFAULT_POLICY", "gate_artifact", "gate_diff"]
