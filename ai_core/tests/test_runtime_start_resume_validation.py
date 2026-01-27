from __future__ import annotations

import pytest

from ai_core.agent.runtime import AgentRuntime
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


def _make_tool_context() -> ToolContext:
    scope = ScopeContext(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        user_id="00000000-0000-0000-0000-000000000001",
        run_id="run-1",
    )
    business = BusinessContext(case_id="case-1")
    return ToolContext(scope=scope, business=business)


def test_runtime_rejects_dict_tool_context():
    runtime = AgentRuntime()
    config = RuntimeConfig(execution_scope="CASE")

    with pytest.raises(TypeError):
        runtime.start(
            tool_context={"scope": {}},  # type: ignore[arg-type]
            runtime_config=config,
            flow_name="dummy_flow",
            flow_input={"query": "hi"},
        )


def test_runtime_config_is_id_free():
    with pytest.raises(ValueError):
        RuntimeConfig(case_id="case-1")  # type: ignore[call-arg]


def test_runtime_rejects_flow_input_with_case_id_field():
    runtime = AgentRuntime()
    config = RuntimeConfig(execution_scope="CASE")

    with pytest.raises(ValueError):
        runtime.start(
            tool_context=_make_tool_context(),
            runtime_config=config,
            flow_name="dummy_flow",
            flow_input={"query": "hi", "case_id": "case-1"},
        )
