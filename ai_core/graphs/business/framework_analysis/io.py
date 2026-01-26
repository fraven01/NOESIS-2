"""I/O contracts for framework analysis graph."""

from __future__ import annotations

from typing import Literal, Mapping, Any

from pydantic import BaseModel, ConfigDict, field_validator

from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.tool_contracts import ToolContext
from ai_core.tools.framework_contracts import (
    FrameworkAnalysisDraft,
    FrameworkAnalysisInput,
)


FRAMEWORK_ANALYSIS_SCHEMA_ID = "noesis.graphs.framework_analysis"
FRAMEWORK_ANALYSIS_IO_VERSION = GraphIOVersion(major=1, minor=0, patch=0)
FRAMEWORK_ANALYSIS_IO_VERSION_STRING = FRAMEWORK_ANALYSIS_IO_VERSION.as_string()


class FrameworkAnalysisGraphInput(BaseModel):
    """Boundary input model for the framework analysis graph."""

    schema_id: Literal[FRAMEWORK_ANALYSIS_SCHEMA_ID]
    schema_version: str
    input: FrameworkAnalysisInput
    tool_context: ToolContext
    runtime: Mapping[str, Any] | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("schema_version must be a string")
        parts = value.strip().split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("schema_version must be in MAJOR.MINOR.PATCH form")
        major = int(parts[0])
        if major != FRAMEWORK_ANALYSIS_IO_VERSION.major:
            raise ValueError("schema_version major must match")
        return value.strip()


class FrameworkAnalysisGraphOutput(FrameworkAnalysisDraft):
    """Boundary output model for the framework analysis graph."""

    schema_id: Literal[FRAMEWORK_ANALYSIS_SCHEMA_ID] = FRAMEWORK_ANALYSIS_SCHEMA_ID
    schema_version: Literal[FRAMEWORK_ANALYSIS_IO_VERSION_STRING] = (
        FRAMEWORK_ANALYSIS_IO_VERSION_STRING
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


FRAMEWORK_ANALYSIS_IO = GraphIOSpec(
    schema_id=FRAMEWORK_ANALYSIS_SCHEMA_ID,
    version=FRAMEWORK_ANALYSIS_IO_VERSION,
    input_model=FrameworkAnalysisGraphInput,
    output_model=FrameworkAnalysisGraphOutput,
)

# Ensure Pydantic resolves forward references for FrameworkStructure.
FrameworkAnalysisGraphOutput.model_rebuild()


__all__ = [
    "FRAMEWORK_ANALYSIS_SCHEMA_ID",
    "FRAMEWORK_ANALYSIS_IO_VERSION",
    "FRAMEWORK_ANALYSIS_IO_VERSION_STRING",
    "FrameworkAnalysisGraphInput",
    "FrameworkAnalysisGraphOutput",
    "FRAMEWORK_ANALYSIS_IO",
]
