from __future__ import annotations

from pathlib import Path

from ai_core.agent.run_store import FileRunStore
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
        run_id="run",
    )
    return ToolContext(scope=scope, business=BusinessContext(), metadata={})


def test_resume_loads_existing_run_and_appends_events(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path / "runs")
    runtime = AgentRuntime(run_store=store)

    record = runtime.start(
        tool_context=_tool_context(),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="dummy_flow",
        flow_input={"query": "alpha"},
    )
    initial_events = list(record["decision_log"])

    resumed = runtime.resume(
        run_id=record["run_id"],
        tool_context=_tool_context(),
        runtime_config=RuntimeConfig(execution_scope="TENANT"),
        flow_name="dummy_flow",
        flow_input={"query": "beta"},
        resume_input={"note": "resume"},
    )

    assert len(resumed["decision_log"]) > len(initial_events)
    stored = store.get(record["run_id"])
    assert stored is not None
    assert len(stored["decision_log"]) == len(resumed["decision_log"])
