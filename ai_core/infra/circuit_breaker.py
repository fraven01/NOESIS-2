"""Simple in-process circuit breaker used for LiteLLM calls."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

from ai_core.infra.observability import emit_event, update_observation
from ai_core.metrics.circuit_metrics import set_circuit_breaker_state
from common.logging import get_logger

logger = get_logger(__name__)

_LITELLM_BREAKER: "CircuitBreaker | None" = None
_LITELLM_LOCK = threading.Lock()


def _coerce_int(value: Any, default: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


def _load_setting(name: str, default: Any) -> Any:
    try:  # pragma: no cover - optional settings in unit tests
        from django.conf import settings

        return getattr(settings, name, default)
    except Exception:
        return default


class CircuitBreaker:
    """Minimal circuit breaker with exponential backoff open window."""

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int,
        base_backoff_s: float,
        max_backoff_s: float,
    ) -> None:
        self._name = name
        self._failure_threshold = max(1, failure_threshold)
        self._base_backoff_s = max(1.0, base_backoff_s)
        self._max_backoff_s = max(1.0, max_backoff_s)
        self._lock = threading.Lock()
        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._next_retry_at: float | None = None
        self._open_count = 0
        self._half_open_in_flight = False
        set_circuit_breaker_state(name=self._name, state=self._state)

    @property
    def state(self) -> str:
        return self._state

    @property
    def next_retry_at(self) -> float | None:
        return self._next_retry_at

    def allow_request(self) -> bool:
        now = time.time()
        with self._lock:
            if self._state == "open":
                if self._next_retry_at is None or now < self._next_retry_at:
                    return False
                self._transition("half_open", reason="cooldown_elapsed")
            if self._state == "half_open":
                if self._half_open_in_flight:
                    return False
                self._half_open_in_flight = True
                return True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._half_open_in_flight = False
            if self._state != "closed":
                self._transition("closed", reason="success")

    def record_failure(self, *, reason: Optional[str] = None) -> None:
        with self._lock:
            if self._state == "half_open":
                self._half_open_in_flight = False
                self._open(reason=reason or "half_open_failure")
                return
            if self._state == "open":
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._open(reason=reason or "threshold_exceeded")

    def _open(self, *, reason: str) -> None:
        now = time.time()
        self._open_count += 1
        backoff = min(
            self._max_backoff_s,
            self._base_backoff_s * (2 ** max(0, self._open_count - 1)),
        )
        self._opened_at = now
        self._next_retry_at = now + backoff
        self._transition(
            "open",
            reason=reason,
            extra={
                "backoff_s": round(backoff, 2),
                "next_retry_at": self._next_retry_at,
            },
        )

    def _transition(self, state: str, *, reason: str, extra: Optional[dict] = None):
        if self._state == state:
            return
        self._state = state
        payload = {
            "circuit_breaker": self._name,
            "state": state,
            "reason": reason,
            "failures": self._consecutive_failures,
            "open_count": self._open_count,
        }
        if extra:
            payload.update(extra)
        set_circuit_breaker_state(name=self._name, state=state)
        try:
            update_observation(
                metadata={
                    "circuit_breaker.state": state,
                    f"circuit_breaker.{self._name}.state": state,
                }
            )
        except Exception:
            pass
        try:
            emit_event("circuit_breaker.state", payload)
        except Exception:
            pass
        logger.warning("circuit_breaker.state", extra=payload)


def get_litellm_circuit_breaker() -> CircuitBreaker:
    global _LITELLM_BREAKER
    if _LITELLM_BREAKER is not None:
        return _LITELLM_BREAKER
    with _LITELLM_LOCK:
        if _LITELLM_BREAKER is None:
            threshold = _coerce_int(
                _load_setting("LITELLM_CIRCUIT_FAILURE_THRESHOLD", 5), 5
            )
            testing = bool(_load_setting("TESTING", False)) or bool(
                os.getenv("PYTEST_CURRENT_TEST")
            )
            if testing:
                threshold = max(threshold, 1000)
            base_backoff = _coerce_float(
                _load_setting("LITELLM_CIRCUIT_BASE_BACKOFF_S", 60.0), 60.0
            )
            max_backoff = _coerce_float(
                _load_setting("LITELLM_CIRCUIT_MAX_BACKOFF_S", 600.0), 600.0
            )
            _LITELLM_BREAKER = CircuitBreaker(
                name="litellm",
                failure_threshold=threshold,
                base_backoff_s=base_backoff,
                max_backoff_s=max_backoff,
            )
    return _LITELLM_BREAKER


__all__ = ["CircuitBreaker", "get_litellm_circuit_breaker"]
