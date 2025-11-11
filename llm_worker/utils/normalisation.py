from __future__ import annotations

"""Shared normalisation helpers for reranking models and graphs."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar


EnumT = TypeVar("EnumT", bound=Enum)


def clamp_score(value: Any, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp numeric scores into a bounded interval."""

    try:
        score = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive conversion
        raise ValueError("score must be numeric") from exc
    if not minimum <= score <= maximum:
        raise ValueError(f"score must be between {minimum} and {maximum}")
    return score


def coerce_enum(value: Any, enum_cls: type[EnumT]) -> EnumT | None:
    """Attempt to coerce arbitrary values into the provided Enum type."""

    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except Exception:
        return None


def ensure_aware_utc(value: Any) -> datetime | None:
    """Normalise timestamps to timezone-aware UTC datetimes."""

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        candidate = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            candidate = datetime.fromisoformat(text)
        except ValueError:
            return None
    if candidate.tzinfo is None or candidate.tzinfo.utcoffset(candidate) is None:
        return None
    return candidate.astimezone(timezone.utc)
