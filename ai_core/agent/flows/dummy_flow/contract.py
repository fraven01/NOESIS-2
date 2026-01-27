from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

flow_name = "dummy_flow"
flow_version = "0.1.0"
required_capabilities = ["dummy.capability"]
supported_scopes = ["CASE", "TENANT"]


class InputModel(BaseModel):
    query: str = Field(..., description="Dummy input payload")

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OutputModel(BaseModel):
    result: str = Field(..., description="Dummy output payload")

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


__all__ = [
    "flow_name",
    "flow_version",
    "required_capabilities",
    "supported_scopes",
    "InputModel",
    "OutputModel",
]
