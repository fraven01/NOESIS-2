from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel, ConfigDict


class GraphIOVersion(BaseModel):
    """Semantic version for graph I/O contracts."""

    major: int
    minor: int = 0
    patch: int = 0

    model_config = ConfigDict(frozen=True, extra="forbid")

    def as_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class GraphIOSpec:
    """Declarative I/O contract for a graph boundary."""

    schema_id: str
    version: GraphIOVersion
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]

    @property
    def version_string(self) -> str:
        return self.version.as_string()
