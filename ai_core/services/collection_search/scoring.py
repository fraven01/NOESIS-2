"""Scoring helpers for collection search."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def calculate_generic_heuristics(
    result: Mapping[str, Any],
    query: str,
) -> float:
    """Calculate generic quality heuristics for a search result (0-100 score)."""
    score = 0.0

    title = str(result.get("title") or "").lower()
    snippet = str(result.get("snippet") or "").lower()
    url = str(result.get("url") or "").lower()
    query_lower = query.lower()

    # 1. Title relevance (0-30 points)
    if query_lower in title:
        score += 30.0
    elif any(word in title for word in query_lower.split() if len(word) > 3):
        score += 15.0

    # 2. Snippet quality (0-25 points)
    snippet_words = len(snippet.split())
    score += min(snippet_words / 20.0, 25.0)  # More context = better

    # 3. Query coverage in snippet (0-20 points)
    query_mentions = snippet.count(query_lower)
    score += min(query_mentions * 10.0, 20.0)

    # 4. URL quality penalties (0 to -20 points)
    if any(
        x in url
        for x in ["login", "signup", "register", "cookie-policy", "privacy-policy"]
    ):
        score -= 20.0

    # 5. Source position boost (small bonus for early results)
    position = result.get("position", 0)
    if position < 3:
        score += 5.0

    return max(0.0, min(score, 100.0))
