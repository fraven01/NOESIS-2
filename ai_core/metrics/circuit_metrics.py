"""Prometheus metrics for circuit breaker state tracking."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

try:  # pragma: no cover - optional dependency
    from prometheus_client import Gauge as _PromGauge  # type: ignore
except Exception:  # pragma: no cover - fallback path exercised in tests
    _PromGauge = None


class _LabelGauge:
    def __init__(
        self,
        store: Dict[Tuple[Tuple[str, str], ...], float],
        key: Tuple[Tuple[str, str], ...],
    ) -> None:
        self._store = store
        self._key = key

    def set(self, value: float) -> None:
        self._store[self._key] = float(value)

    @property
    def value(self) -> float:
        return self._store[self._key]


class _FallbackGauge:
    def __init__(self) -> None:
        self._store: Dict[Tuple[Tuple[str, str], ...], float] = defaultdict(float)

    def labels(self, **labels: str) -> _LabelGauge:
        key = tuple(sorted((k, v) for k, v in labels.items()))
        return _LabelGauge(self._store, key)

    def value(self, **labels: str) -> float:
        key = tuple(sorted((k, v) for k, v in labels.items()))
        return self._store.get(key, 0.0)


if _PromGauge is not None:  # pragma: no cover - integration path
    CIRCUIT_BREAKER_STATE = _PromGauge(
        "circuit_breaker_state",
        "Circuit breaker state (1 for current state).",
        ["name", "state"],
    )
else:  # pragma: no cover - fallback used in unit tests
    CIRCUIT_BREAKER_STATE = _FallbackGauge()

_LAST_STATE: Dict[str, str] = {}


def set_circuit_breaker_state(*, name: str, state: str) -> None:
    """Set the circuit breaker state gauge for a named breaker."""

    name = name or "unknown"
    state = state or "unknown"
    previous = _LAST_STATE.get(name)
    if previous and previous != state:
        try:
            CIRCUIT_BREAKER_STATE.labels(name=name, state=previous).set(0)
        except Exception:
            pass
    try:
        CIRCUIT_BREAKER_STATE.labels(name=name, state=state).set(1)
    except Exception:
        return
    _LAST_STATE[name] = state


__all__ = ["CIRCUIT_BREAKER_STATE", "set_circuit_breaker_state"]
