from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ai_core.agent.capabilities.registry import CapabilitySpec, register_capability
from ai_core.graph.io import GraphIOSpec, GraphIOVersion


class EchoInput(BaseModel):
    message: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class EchoOutput(BaseModel):
    message: str

    model_config = ConfigDict(extra="forbid", frozen=True)


IO_SPEC = GraphIOSpec(
    schema_id="capability.echo",
    version=GraphIOVersion(major=0, minor=1, patch=0),
    input_model=EchoInput,
    output_model=EchoOutput,
)


def run(input_model: EchoInput) -> EchoOutput:
    return EchoOutput(message=input_model.message)


CAPABILITY = CapabilitySpec(
    name="echo",
    version="0.1.0",
    io_spec_version=IO_SPEC.version_string,
    entrypoint=run,
    io_spec=IO_SPEC,
    input_schema=EchoInput,
    output_schema=EchoOutput,
)

register_capability(CAPABILITY)
