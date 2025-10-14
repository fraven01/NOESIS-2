"""Utilities for managing parent node payloads.

These helpers ensure that parent node metadata stays within a predictable
payload size, preventing excessively large JSON documents when parents are
persisted alongside document metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from django.conf import settings


def _resolve_parent_cap() -> int:
    """Return the configured parent payload cap in bytes."""

    try:
        value = getattr(settings, "RAG_PARENT_MAX_BYTES", 0)
    except Exception:  # pragma: no cover - defensive guard
        return 0

    try:
        cap = int(value)
    except (TypeError, ValueError):
        return 0

    return cap if cap > 0 else 0


def _truncate_text(text: str, limit: int) -> tuple[str, bool, int]:
    """Truncate *text* to *limit* bytes, preserving UTF-8 boundaries."""

    encoded = text.encode("utf-8")
    byte_length = len(encoded)
    if limit <= 0 or byte_length <= limit:
        return text, False, byte_length

    truncated = encoded[:limit]
    preview = truncated.decode("utf-8", errors="ignore")
    return preview, True, byte_length


def limit_parent_payload(parents: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return a copy of *parents* with oversized content truncated.

    The truncation respects the ``settings.RAG_PARENT_MAX_BYTES`` cap. When a
    parent node's content is truncated a ``content_truncated`` flag is added
    alongside the original byte length so downstream consumers can react if
    necessary.
    """

    cap = _resolve_parent_cap()
    limited: Dict[str, Dict[str, Any]] = {}

    for parent_id, payload in parents.items():
        if not isinstance(payload, Mapping):
            continue

        node: Dict[str, Any] = dict(payload)
        content = node.get("content")
        if isinstance(content, str):
            preview, truncated, byte_length = _truncate_text(content, cap)
            node["content"] = preview
            if truncated:
                node["content_truncated"] = True
                node.setdefault("content_length", len(content))
                node["content_bytes"] = byte_length

        limited[str(parent_id)] = node

    return limited

