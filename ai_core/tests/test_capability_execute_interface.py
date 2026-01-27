from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from ai_core.agent.capabilities.base import CapabilitySpec
from ai_core.agent.capabilities.registry import execute, register_capability
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


class DummyInput(BaseModel):
    value: str


class DummyOutput(BaseModel):
    value: str


def test_capability_registry_rejects_missing_schemas():
    def _noop(
        _context: ToolContext, _config: RuntimeConfig, _input: DummyInput
    ) -> DummyOutput:
        return DummyOutput(value=_input.value)

    with pytest.raises(ValueError, match="missing input/output schema"):
        register_capability(
            CapabilitySpec(
                name="invalid",
                version="0.0.1",
                io_spec_version="0.0.1",
                execute=_noop,
                io_spec=object(),  # type: ignore[arg-type]
            )
        )


def test_capability_registry_executes_stub_echo_with_validation():
    def _echo(
        _context: ToolContext, _config: RuntimeConfig, _input: DummyInput
    ) -> DummyOutput:
        return DummyOutput(value=_input.value)

    register_capability(
        CapabilitySpec(
            name="echo",
            version="0.1.0",
            io_spec_version="0.1.0",
            input_model=DummyInput,
            output_model=DummyOutput,
            execute=_echo,
            io_spec=object(),  # type: ignore[arg-type]
        )
    )

    scope = ScopeContext(
        tenant_id="tenant",
        trace_id="trace",
        invocation_id="invocation",
        service_id="svc",
        run_id="run",
    )
    context = ToolContext(scope=scope, business=BusinessContext(), metadata={})
    runtime_config = RuntimeConfig(execution_scope="TENANT")

    output = execute("echo", context, runtime_config, {"value": "hi"})
    assert output.value == "hi"

    with pytest.raises(ValidationError):
        execute("echo", context, runtime_config, {"not_value": "bad"})
