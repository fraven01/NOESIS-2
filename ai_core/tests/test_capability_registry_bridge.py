from __future__ import annotations

import pytest
from pydantic import BaseModel

from ai_core.agent.capabilities.registry import (
    CapabilitySpec,
    get_capability,
    register_capability,
)
import ai_core.agent.capabilities.stub_echo  # noqa: F401


class DummyInput(BaseModel):
    value: str


class DummyOutput(BaseModel):
    value: str


def test_runtime_resolves_capability_by_name():
    spec = get_capability("echo")
    assert spec.name == "echo"
    assert spec.version == "0.1.0"


def test_capability_requires_iospec_binding():
    def _noop(_input: DummyInput) -> DummyOutput:
        return DummyOutput(value=_input.value)

    with pytest.raises(ValueError, match="missing io_spec"):
        register_capability(
            CapabilitySpec(
                name="invalid",
                version="0.0.1",
                io_spec_version="0.0.1",
                entrypoint=_noop,
                io_spec=None,  # type: ignore[arg-type]
                input_schema=DummyInput,
                output_schema=DummyOutput,
            )
        )

    with pytest.raises(ValueError, match="missing input/output schema"):
        register_capability(
            CapabilitySpec(
                name="invalid2",
                version="0.0.1",
                io_spec_version="0.0.1",
                entrypoint=_noop,
                io_spec=object(),  # type: ignore[arg-type]
                input_schema=None,  # type: ignore[arg-type]
                output_schema=None,  # type: ignore[arg-type]
            )
        )
