"""Normalization helpers for RAG search parameters."""

from __future__ import annotations

import math
import os
from enum import Enum
from typing import Any, TypeVar, cast

Number = TypeVar("Number", int, float)

__all__ = [
    "CandidatePoolPolicy",
    "clamp_fraction",
    "get_limit_setting",
    "normalize_max_candidates",
    "normalize_top_k",
    "resolve_candidate_pool_policy",
]


class CandidatePoolPolicy(str, Enum):
    """Policy describing how to handle undersized candidate pools."""

    NORMALIZE = "normalize"
    ERROR = "error"


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):  # bool inherits from int – guard explicitly
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return int(value)
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return int(float(candidate)) if "." in candidate else int(candidate)
        except ValueError:
            return None
    return None


def _coerce_to_type(value: str, default: Number) -> Number | None:
    try:
        if isinstance(default, bool):  # pragma: no cover - defensive
            return cast(Number, type(default)(value))  # type: ignore[call-arg]
        if isinstance(default, int) and not isinstance(default, bool):
            return cast(Number, int(value))
        if isinstance(default, float):
            return cast(Number, float(value))
    except (TypeError, ValueError):
        return None
    return None


def get_limit_setting(name: str, default: Number) -> Number:
    """Resolve a numeric setting from environment or Django settings."""

    env_value = os.getenv(name)
    if env_value is not None:
        coerced = _coerce_to_type(env_value, default)
        if coerced is not None:
            return coerced

    try:  # pragma: no cover - requires Django settings
        from django.conf import settings  # type: ignore

        configured = getattr(settings, name, default)
    except Exception:
        return default

    if isinstance(default, float):
        try:
            return cast(Number, float(configured))
        except (TypeError, ValueError):
            return default
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return cast(Number, int(configured))
        except (TypeError, ValueError):
            return default
    return default


def resolve_candidate_pool_policy(
    *, default: CandidatePoolPolicy = CandidatePoolPolicy.ERROR
) -> CandidatePoolPolicy:
    """Return the configured candidate pool policy.

    The policy can be set via the ``RAG_CANDIDATE_POLICY`` environment variable or
    Django setting. Accepted values are ``"normalize"`` and ``"error"``; any other
    value falls back to ``default``.
    """

    configured: str | None = None
    env_value = os.getenv("RAG_CANDIDATE_POLICY")
    if env_value:
        configured = env_value
    else:  # pragma: no cover - requires Django settings
        try:
            from django.conf import settings  # type: ignore

            configured = getattr(settings, "RAG_CANDIDATE_POLICY", None)
        except Exception:
            configured = None

    if configured is None:
        return default

    normalized = str(configured).strip().lower()
    for policy in CandidatePoolPolicy:
        if policy.value == normalized:
            return policy
    return default


def clamp_fraction(
    value: Any,
    *,
    default: float,
    return_source: bool = False,
) -> float | tuple[float, str]:
    """Clamp ``value`` to the inclusive range [0, 1] or fall back to ``default``."""

    default_value = float(default)
    source = "from_default"
    coerced = _coerce_float(value)
    if coerced is not None and math.isfinite(coerced) and 0.0 <= coerced <= 1.0:
        normalized = float(coerced)
        source = "from_state"
    else:
        normalized = default_value
    if return_source:
        return normalized, source
    return normalized


def normalize_top_k(
    requested: Any,
    *,
    default: int = 5,
    minimum: int = 1,
    maximum: int = 10,
    return_source: bool = False,
) -> int | tuple[int, str]:
    """Normalise ``top_k`` requests to sane integer bounds."""

    default_value = int(default)
    candidate = _coerce_int(requested)
    if candidate is None:
        normalized = default_value
        source = "from_default"
    else:
        normalized = candidate
        source = "from_state"
    normalized = max(minimum, min(maximum, normalized))
    if return_source:
        return normalized, source
    return normalized


def normalize_max_candidates(
    top_k: int,
    requested: Any,
    cap: int | None,
    *,
    return_source: bool = False,
) -> int | tuple[int, str]:
    """Ensure ``max_candidates`` respects ``top_k`` and optional caps."""

    effective_cap = int(cap) if cap is not None else None
    default_value = effective_cap if effective_cap is not None else max(top_k, 1)
    candidate = _coerce_int(requested)
    if candidate is None:
        normalized = default_value
        source = "from_default"
    else:
        normalized = candidate
        source = "from_state"
    normalized = max(top_k, normalized)
    if effective_cap is not None:
        normalized = min(effective_cap, normalized)
    if return_source:
        return normalized, source
    return normalized
