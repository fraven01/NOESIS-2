from __future__ import annotations

import time

from ai_core.infra.circuit_breaker import CircuitBreaker


def test_circuit_breaker_opens_and_recovers() -> None:
    breaker = CircuitBreaker(
        name="test",
        failure_threshold=2,
        base_backoff_s=1,
        max_backoff_s=2,
    )

    assert breaker.state == "closed"
    assert breaker.allow_request() is True

    breaker.record_failure(reason="failed")
    assert breaker.state == "closed"

    breaker.record_failure(reason="failed")
    assert breaker.state == "open"
    assert breaker.allow_request() is False

    # Force cooldown to elapse without waiting in real time.
    breaker._next_retry_at = time.time() - 1

    assert breaker.allow_request() is True
    assert breaker.state == "half_open"
    assert breaker.allow_request() is False

    breaker.record_success()
    assert breaker.state == "closed"
