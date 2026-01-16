"""Selection helpers for web acquisition results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def is_blocked_domain(url: str, blocked_domains: Sequence[str] | None) -> bool:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if not hostname or not blocked_domains:
        return False
    for domain in blocked_domains:
        blocked = domain.lower()
        if hostname == blocked or hostname.endswith(f".{blocked}"):
            return True
    return False


def select_search_candidates(
    *,
    results: Sequence[Mapping[str, Any]],
    preselected_results: Sequence[Mapping[str, Any]] | None,
    search_config: Mapping[str, Any] | None,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None]:
    config = search_config or {}

    min_len = config.get("min_snippet_length", 40)
    blocked = config.get("blocked_domains", [])
    top_n = config.get("top_n", 5)
    prefer_pdf = config.get("prefer_pdf", True)

    preselected_urls = {
        item["url"] for item in (preselected_results or []) if item.get("url")
    }

    validated: list[Mapping[str, Any]] = []
    for raw in results:
        url = raw.get("url")
        snippet = raw.get("snippet", "")

        if not url:
            continue

        if url not in preselected_urls and len(snippet) < min_len:
            continue

        if is_blocked_domain(url, blocked):
            continue

        lowered = snippet.lower()
        if "noindex" in lowered and "robot" in lowered:
            continue

        validated.append(raw)

    shortlisted = validated[:top_n]
    selected = None

    if shortlisted:
        if prefer_pdf:
            for cand in shortlisted:
                if cand.get("is_pdf"):
                    selected = cand
                    break
        if not selected:
            selected = shortlisted[0]

    return shortlisted, selected
