"""Auto-ingest selection helpers for collection search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(key)
    return getattr(result, key, None)


def select_auto_ingest_urls(
    ranked: Sequence[Any],
    *,
    top_k: int,
    min_score: float,
    candidate_by_id: Mapping[str, Any] | None = None,
) -> list[str]:
    """Select URLs from ranked items for auto-ingestion.

    Args:
        ranked: Sequence of scored items (e.g., LLMScoredItem objects).
        top_k: Maximum number of URLs to select.
        min_score: Minimum score threshold (0-100 scale).
        candidate_by_id: Optional mapping from candidate_id to candidate objects
            containing the URL. Required when ranked items only have candidate_id
            but not the URL directly (e.g., LLMScoredItem from HybridResult).
    """
    selected: list[str] = []
    candidate_lookup = candidate_by_id or {}

    for item in ranked:
        if len(selected) >= top_k:
            break
        if (_result_value(item, "score") or 0.0) < min_score:
            continue

        # Try to get URL directly from item first
        url = _result_value(item, "url")

        # If no direct URL, resolve via candidate_id lookup
        if not url:
            candidate_id = _result_value(item, "candidate_id")
            if candidate_id and candidate_id in candidate_lookup:
                candidate = candidate_lookup[candidate_id]
                url = _result_value(candidate, "url")

        if url:
            selected.append(url)

    return selected
