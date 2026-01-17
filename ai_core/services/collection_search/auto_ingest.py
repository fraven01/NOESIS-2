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
) -> list[str]:
    selected: list[str] = []
    for item in ranked:
        if len(selected) >= top_k:
            break
        if (_result_value(item, "score") or 0.0) < min_score:
            continue
        url = _result_value(item, "url")
        if url:
            selected.append(url)
    return selected
