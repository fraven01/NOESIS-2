"""Blueprint contract models for plan construction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION_PATTERN = r"^(v0|\d+\.\d+\.\d+)$"


class SlotSpec(BaseModel):
    """Slot specification for blueprint inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    key: str
    required: bool = True
    description: str | None = None
    slot_type: str | None = None
    json_schema_ref: str | None = None
    constraints: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_slot_type(self) -> "SlotSpec":
        if self.slot_type and self.json_schema_ref:
            raise ValueError("slot_type and json_schema_ref are mutually exclusive")
        return self


class BlueprintGate(BaseModel):
    """Gate definition for blueprint validation."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    key: str
    description: str | None = None
    criteria: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Blueprint(BaseModel):
    """Blueprint schema describing allowed slot and gate structure."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    schema_version: str = Field(default="v0", pattern=SCHEMA_VERSION_PATTERN)
    name: str
    version: str
    description: str | None = None
    slots: list[SlotSpec] = Field(default_factory=list)
    gates: list[BlueprintGate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "SCHEMA_VERSION_PATTERN",
    "SlotSpec",
    "BlueprintGate",
    "Blueprint",
]
