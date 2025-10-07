"""Visibility options and helpers for RAG retrieval."""

from __future__ import annotations

from enum import Enum


class Visibility(str, Enum):
    """Enumerated visibility modes supported by the retriever stack."""

    ACTIVE = "active"
    ALL = "all"
    DELETED = "deleted"


DEFAULT_VISIBILITY: Visibility = Visibility.ACTIVE
_ALLOWED_TEXT = {value.value for value in Visibility}
_TRUE_LITERALS = {"1", "true", "yes", "y", "on"}
_FALSE_LITERALS = {"0", "false", "no", "n", "off", ""}


def normalize_visibility(value: object | None) -> tuple[Visibility, str]:
    """Return the normalized visibility and its source identifier."""

    if value is None:
        return DEFAULT_VISIBILITY, "from_default"

    if isinstance(value, Visibility):
        return value, "from_state"

    text = str(value).strip().lower()
    if not text:
        return DEFAULT_VISIBILITY, "from_default"

    if text in _ALLOWED_TEXT:
        return Visibility(text), "from_state"

    raise ValueError(f"unsupported visibility '{value}'")


def coerce_bool_flag(value: object | None) -> bool:
    """Coerce *value* to a boolean flag understood by visibility guards."""

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in _TRUE_LITERALS:
        return True
    if text in _FALSE_LITERALS:
        return False

    return bool(text)


__all__ = [
    "DEFAULT_VISIBILITY",
    "Visibility",
    "coerce_bool_flag",
    "normalize_visibility",
]
