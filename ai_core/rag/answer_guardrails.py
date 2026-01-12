from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_MIN_SNIPPETS = 1
DEFAULT_MIN_TOP_SCORE = 0.2


@dataclass(frozen=True)
class AnswerGuardrailResult:
    allowed: bool
    reason: str
    snippet_count: int
    top_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "snippet_count": self.snippet_count,
            "top_score": self.top_score,
        }


def _coerce_int(value: object, *, fallback: int) -> int:
    try:
        candidate = int(str(value))
    except (TypeError, ValueError):
        return fallback
    if candidate < 0:
        return fallback
    return candidate


def _coerce_float(value: object, *, fallback: float) -> float:
    try:
        candidate = float(str(value))
    except (TypeError, ValueError):
        return fallback
    if candidate != candidate:
        return fallback
    return candidate


def _max_score(snippets: Sequence[Mapping[str, Any]]) -> float:
    top_score = 0.0
    for snippet in snippets:
        score = _coerce_float(snippet.get("score"), fallback=0.0)
        if score > top_score:
            top_score = score
    return top_score


def evaluate_answer_guardrails(
    snippets: Sequence[Mapping[str, Any]],
) -> AnswerGuardrailResult:
    min_snippets = _coerce_int(
        os.getenv("RAG_GUARDRAIL_MIN_SNIPPETS"), fallback=DEFAULT_MIN_SNIPPETS
    )
    min_top_score = _coerce_float(
        os.getenv("RAG_GUARDRAIL_MIN_TOP_SCORE"), fallback=DEFAULT_MIN_TOP_SCORE
    )
    snippet_count = len(snippets)
    top_score = _max_score(snippets)

    if snippet_count < min_snippets:
        return AnswerGuardrailResult(
            allowed=False,
            reason="insufficient_snippets",
            snippet_count=snippet_count,
            top_score=top_score,
        )
    if top_score < min_top_score:
        return AnswerGuardrailResult(
            allowed=False,
            reason="low_top_score",
            snippet_count=snippet_count,
            top_score=top_score,
        )
    return AnswerGuardrailResult(
        allowed=True,
        reason="ok",
        snippet_count=snippet_count,
        top_score=top_score,
    )


__all__ = ["AnswerGuardrailResult", "evaluate_answer_guardrails"]
