from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from pydantic import BaseModel

from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.graph.io import GraphIOSpec
from ai_core.tool_contracts.base import ToolContext


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    version: str
    io_spec_version: str
    input_model: Type[BaseModel] | None = None
    output_model: Type[BaseModel] | None = None
    execute: Callable[[ToolContext, RuntimeConfig, BaseModel], BaseModel] | None = None
    io_spec: GraphIOSpec | None = None
    entrypoint: Callable[[BaseModel], BaseModel] | None = None
    input_schema: Type[BaseModel] | None = None
    output_schema: Type[BaseModel] | None = None


__all__ = ["CapabilitySpec"]
