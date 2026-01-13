"""Auto-ingest selection helpers for collection search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def select_auto_ingest_urls(
    ranked: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    min_score: float,
) -> list[str]:
    selected: list[str] = []
    for item in ranked:
        if len(selected) >= top_k:
            break
        if item.get("score", 0.0) < min_score:
            continue
        url = item.get("url")
        if url:
            selected.append(url)
    return selected
