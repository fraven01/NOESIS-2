from __future__ import annotations

from ai_core.agent.runtime import AgentRuntime
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


def _tool_context() -> ToolContext:
    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="scope-run",
    )
    return ToolContext(scope=scope, business=BusinessContext(), metadata={})


def test_runtime_executes_dummy_flow_and_returns_stop_decision():
    runtime = AgentRuntime()
    record = runtime.start(
        tool_context=_tool_context(),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="dummy_flow",
        flow_input={"query": "ping"},
    )

    assert record["output"]["result"] == "ping"
    assert record["stop_decision"]["status"] == "succeeded"
    assert record["stop_decision"]["reason"] == "flow completed"


def test_runtime_emits_start_and_stop_events():
    runtime = AgentRuntime()
    record = runtime.start(
        tool_context=_tool_context(),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="dummy_flow",
        flow_input={"query": "ping"},
    )

    decision_log = record["decision_log"]
    assert len(decision_log) == 2
    assert decision_log[0]["kind"] == "start"
    assert decision_log[1]["kind"] == "stop"
