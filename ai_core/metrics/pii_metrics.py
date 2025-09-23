"""Instrumentation helpers for PII masking detections."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter as _PromCounter  # type: ignore
except Exception:  # pragma: no cover - fallback path exercised in tests
    _PromCounter = None


class _LabelCounter:
    def __init__(
        self,
        store: Dict[Tuple[Tuple[str, str], ...], float],
        key: Tuple[Tuple[str, str], ...],
    ):
        self._store = store
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        self._store[self._key] += amount

    @property
    def value(self) -> float:
        return self._store[self._key]


class _FallbackCounter:
    def __init__(self) -> None:
        self._store: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)

    def labels(self, **labels: str) -> _LabelCounter:
        key = tuple(sorted((k, v) for k, v in labels.items()))
        return _LabelCounter(self._store, key)

    def value(self, **labels: str) -> float:
        key = tuple(sorted((k, v) for k, v in labels.items()))
        return self._store.get(key, 0.0)


if _PromCounter is not None:  # pragma: no cover - integration path
    PII_DETECTIONS = _PromCounter(
        "pii_detections_total",
        "Total number of PII detections performed by the masking layer.",
        ["tag"],
    )
    PII_FALSE_POSITIVES = _PromCounter(
        "pii_false_positive_total",
        "Review hook counter for reported false positives.",
        ["tag"],
    )
    PII_FALSE_NEGATIVES = _PromCounter(
        "pii_false_negative_total",
        "Review hook counter for reported false negatives.",
        ["tag"],
    )
else:  # pragma: no cover - fallback used in unit tests
    PII_DETECTIONS = _FallbackCounter()
    PII_FALSE_POSITIVES = _FallbackCounter()
    PII_FALSE_NEGATIVES = _FallbackCounter()


def _emit_statsd(metric: str, *, tag: str) -> None:
    """Placeholder for a StatsD hook that mirrors Prometheus counters."""

    # StatsD integration will be wired once the agent-side emitter is available.
    # The function exists so callers do not need to change once the hook lands.
    return None


def record_detection(tag: str) -> None:
    """Increase the detection counter for a masked PII class."""

    PII_DETECTIONS.labels(tag=tag).inc()
    _emit_statsd("pii.detections", tag=tag)


def record_false_positive(tag: str) -> None:
    """Increment the review hook counter for a reported false positive."""

    PII_FALSE_POSITIVES.labels(tag=tag).inc()
    _emit_statsd("pii.false_positive", tag=tag)


def record_false_negative(tag: str) -> None:
    """Increment the review hook counter for a reported false negative."""

    PII_FALSE_NEGATIVES.labels(tag=tag).inc()
    _emit_statsd("pii.false_negative", tag=tag)


__all__ = [
    "PII_DETECTIONS",
    "PII_FALSE_NEGATIVES",
    "PII_FALSE_POSITIVES",
    "record_detection",
    "record_false_negative",
    "record_false_positive",
]
