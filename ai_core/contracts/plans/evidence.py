"""Evidence models for plan contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION_PATTERN = r"^(v0|\d+\.\d+\.\d+)$"
EvidenceRefType = Literal[
    "url",
    "repo_doc",
    "repo_chunk",
    "object_store",
    "confluence",
    "screenshot",
]
ConfidenceLabel = Literal["low", "medium", "high"]


class Confidence(BaseModel):
    """Confidence attached to evidence references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    label: ConfidenceLabel | None = None

    @model_validator(mode="after")
    def _validate_confidence(self) -> "Confidence":
        if self.score is None and self.label is None:
            raise ValueError("confidence requires score or label")
        return self


class Evidence(BaseModel):
    """Evidence reference for plan outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    schema_version: str = Field(default="v0", pattern=SCHEMA_VERSION_PATTERN)
    ref_type: EvidenceRefType
    ref_id: str
    summary: str | None = None
    confidence: Confidence | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "SCHEMA_VERSION_PATTERN",
    "EvidenceRefType",
    "ConfidenceLabel",
    "Confidence",
    "Evidence",
]
