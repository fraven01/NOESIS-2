"""Fixtures for chaos test scenarios."""

from __future__ import annotations

import os

import pytest

from dataclasses import dataclass, field
from typing import Any, Dict, List

_CHAOS_ENV_VARS = {
    "REDIS_DOWN": "0",
    "SQL_DOWN": "0",
    "SLOW_NET": "0",
    "LANGFUSE_SAMPLE_RATE": "1.0",
    "DEMO_SEED_PROFILE": "demo",
    "DEMO_SEED_SEED": "1337",
}

CHAOS_ENV_REGISTRY: Dict[str, "ChaosEnvironment"] = {}


@dataclass
class ChaosEnvironment:
    """Helper to flip runtime toggles for fault-injection tests."""

    _monkeypatch: pytest.MonkeyPatch
    _values: Dict[str, str]

    def set_redis_down(self, enabled: bool = True) -> None:
        self._set_flag("REDIS_DOWN", enabled)

    def set_sql_down(self, enabled: bool = True) -> None:
        self._set_flag("SQL_DOWN", enabled)

    def set_slow_network(self, enabled: bool = True) -> None:
        self._set_flag("SLOW_NET", enabled)

    def set_seed_profile(self, profile: str) -> None:
        """Set the default demo seed profile used in chaos scenarios."""

        self._set_value("DEMO_SEED_PROFILE", profile)

    def set_seed_value(self, seed: int) -> None:
        """Set the deterministic seed for demo data generation."""

        self._set_value("DEMO_SEED_SEED", str(seed))

    def reset(self) -> None:
        """Reset all toggles to their default values."""

        for key, value in _CHAOS_ENV_VARS.items():
            self._monkeypatch.setenv(key, value)
            self._values[key] = value

    def _set_flag(self, name: str, enabled: bool) -> None:
        value = "1" if enabled else "0"
        self._set_value(name, value)

    def _set_value(self, name: str, value: str) -> None:
        self._monkeypatch.setenv(name, value)
        self._values[name] = value

    @property
    def values(self) -> Dict[str, str]:
        """Return the currently applied toggle values."""

        return dict(self._values)


@dataclass
class LangfuseSpan:
    """Container for recorded Langfuse span metadata."""

    trace_id: str
    operation: str
    metadata: Dict[str, Any]


@dataclass
class MockLangfuseClient:
    """In-memory Langfuse client used to assert tracing metadata in tests."""

    spans: List[LangfuseSpan] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def record_span(
        self, trace_id: str, operation: str, metadata: Dict[str, Any]
    ) -> None:
        self.spans.append(
            LangfuseSpan(trace_id=trace_id, operation=operation, metadata=metadata)
        )

    def record_event(self, payload: Dict[str, Any]) -> None:
        self.events.append(dict(payload))

    @property
    def sample_rate(self) -> str | None:
        """Return the currently configured Langfuse sample rate."""

        return os.getenv("LANGFUSE_SAMPLE_RATE")


@pytest.fixture
def chaos_env(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> ChaosEnvironment:
    """Provision a configurable chaos environment.

    The fixture normalises all fault-injection flags to ``"0"`` (disabled) and
    exposes helper methods for tests to enable Redis downtime, Cloud SQL
    unavailability or simulated network latency.
    """

    values = dict(_CHAOS_ENV_VARS)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    env = ChaosEnvironment(monkeypatch, values)
    nodeid = request.node.nodeid
    CHAOS_ENV_REGISTRY[nodeid] = env

    def _record_state() -> None:
        request.node.user_properties.append(("chaos_env", env.values))

    request.addfinalizer(_record_state)
    try:
        yield env
    finally:
        CHAOS_ENV_REGISTRY.pop(nodeid, None)
        env.reset()


@pytest.fixture
def langfuse_mock(monkeypatch: pytest.MonkeyPatch) -> MockLangfuseClient:
    """Provide a mock Langfuse client capturing spans and events during tests."""

    from ai_core.infra import tracing

    client = MockLangfuseClient()

    def _capture_dispatch(
        trace_id: str, node_name: str, metadata: Dict[str, Any]
    ) -> None:
        client.record_span(trace_id, node_name, metadata)

    def _capture_log(payload: Dict[str, Any]) -> None:
        client.record_event(payload)

    monkeypatch.setattr(tracing, "emit_span", _capture_dispatch)
    monkeypatch.setattr(tracing, "emit_event", _capture_log)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "1.0")

    return client
