from __future__ import annotations

from typing import Any

from ai_core.agent.harness.validate import ValidationError, validate_artifact_v0


def _schema_ok(payload: dict[str, Any]) -> bool:
    try:
        validate_artifact_v0(payload)
    except ValidationError:
        return False
    return True


def _count_citations(payload: dict[str, Any]) -> int:
    citations = payload.get("citations")
    if isinstance(citations, list):
        return len(citations)
    return 0


def _has_claim_map(payload: dict[str, Any]) -> bool:
    claim_map = payload.get("claim_to_citation")
    return isinstance(claim_map, dict) and bool(claim_map)


def _count_ungrounded(payload: dict[str, Any]) -> int:
    ungrounded = payload.get("ungrounded_claims")
    if isinstance(ungrounded, list):
        return len(ungrounded)
    return 0


def _answer_length(payload: dict[str, Any]) -> int:
    answer = payload.get("answer")
    if isinstance(answer, dict):
        text = answer.get("text")
        if isinstance(text, str):
            return len(text)
    return 0


def _changed_fields(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    keys = set(a.keys()) | set(b.keys())
    changed = sorted(key for key in keys if a.get(key) != b.get(key))
    return changed


def diff_artifacts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "schema_ok_a": _schema_ok(a),
        "schema_ok_b": _schema_ok(b),
        "stop_status_a": (a.get("stop_decision") or {}).get("status"),
        "stop_status_b": (b.get("stop_decision") or {}).get("status"),
        "citations_count_a": _count_citations(a),
        "citations_count_b": _count_citations(b),
        "has_claim_map_a": _has_claim_map(a),
        "has_claim_map_b": _has_claim_map(b),
        "ungrounded_claims_count_a": _count_ungrounded(a),
        "ungrounded_claims_count_b": _count_ungrounded(b),
        "answer_length_a": _answer_length(a),
        "answer_length_b": _answer_length(b),
        "changed_fields": _changed_fields(a, b),
    }
    return summary


__all__ = ["diff_artifacts"]
