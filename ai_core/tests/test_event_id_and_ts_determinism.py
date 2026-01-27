from __future__ import annotations

from ai_core.agent.runtime import AgentRuntime
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.agent.time_source import EventIdSource, TimeSource
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


class FixedTimeSource:
    def __init__(self, values: list[str]) -> None:
        self._values = list(values)
        self._index = 0

    def now_iso(self) -> str:
        value = self._values[self._index]
        self._index += 1
        return value


class FixedEventIdSource:
    def __init__(self, values: list[str]) -> None:
        self._values = list(values)
        self._index = 0

    def next_event_id(self) -> str:
        value = self._values[self._index]
        self._index += 1
        return value


def _tool_context() -> ToolContext:
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="run",
    )
    return ToolContext(scope=scope, business=BusinessContext(), metadata={})


def test_runtime_emits_deterministic_ts_and_event_ids_when_injected() -> None:
    time_source: TimeSource = FixedTimeSource(["ts-1", "ts-2"])
    event_id_source: EventIdSource = FixedEventIdSource(["evt-1", "evt-2"])
    runtime = AgentRuntime(time_source=time_source, event_id_source=event_id_source)

    record = runtime.start(
        tool_context=_tool_context(),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="dummy_flow",
        flow_input={"query": "alpha"},
    )

    events = record["decision_log"]
    assert events[0]["event_id"] == "evt-1"
    assert events[0]["ts"] == "ts-1"
    assert events[-1]["event_id"] == "evt-2"
    assert events[-1]["ts"] == "ts-2"
