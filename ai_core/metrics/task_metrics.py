"""Instrumentation helpers for task retry events."""

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
    ) -> None:
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
    AGENT_TASK_RETRIES_TOTAL = _PromCounter(
        "agent_task_retries_total",
        "Total number of task retries executed by agents/ingestion workers.",
        ["agent_id", "reason_category"],
    )
else:  # pragma: no cover - fallback used in unit tests
    AGENT_TASK_RETRIES_TOTAL = _FallbackCounter()


def record_task_retry(*, agent_id: str, reason_category: str) -> None:
    """Increment the retry counter for the provided agent/task identifier."""

    if not agent_id:
        agent_id = "unknown"
    if not reason_category:
        reason_category = "unknown"
    AGENT_TASK_RETRIES_TOTAL.labels(
        agent_id=str(agent_id),
        reason_category=str(reason_category),
    ).inc()


__all__ = ["AGENT_TASK_RETRIES_TOTAL", "record_task_retry"]
