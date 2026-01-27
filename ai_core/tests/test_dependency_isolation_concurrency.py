from __future__ import annotations

from ai_core.agent.runtime import AgentRuntime
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


def _tool_context(tenant_id: str) -> ToolContext:
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="run",
    )
    return ToolContext(scope=scope, business=BusinessContext(), metadata={})


def _extract_router_id(record: dict) -> int:
    for event in record.get("decision_log", []):
        if event.get("kind") == "start":
            telemetry = event.get("telemetry") or {}
            return int(telemetry.get("router_instance_id"))
    raise AssertionError("router_instance_id missing from decision_log")


def test_router_instance_is_isolated_per_run() -> None:
    runtime_a = AgentRuntime()
    runtime_b = AgentRuntime()

    record_a = runtime_a.start(
        tool_context=_tool_context("tenant-a"),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="dummy_flow",
        flow_input={"query": "alpha"},
    )
    record_b = runtime_b.start(
        tool_context=_tool_context("tenant-b"),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="dummy_flow",
        flow_input={"query": "beta"},
    )

    router_id_a = _extract_router_id(record_a)
    router_id_b = _extract_router_id(record_b)

    assert router_id_a != router_id_b
