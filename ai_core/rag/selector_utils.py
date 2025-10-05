"""Utilities for normalising routing selector dimensions."""

from __future__ import annotations

__all__ = ["normalise_selector_value"]


def normalise_selector_value(value: object | None) -> str | None:
    """Return a lowercase, trimmed representation of routing dimensions."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    return text.lower()
