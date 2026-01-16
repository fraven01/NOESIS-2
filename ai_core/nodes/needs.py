from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from ai_core.tool_contracts import ToolContext
from pydantic import BaseModel, ConfigDict, Field


class NeedsInput(BaseModel):
    """Structured input parameters for the needs node."""

    info_state: Mapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


class NeedsOutput(BaseModel):
    """Structured output payload returned by the needs node."""

    filled: list[str]
    missing: list[str]
    ignored: list[str]

    model_config = ConfigDict(extra="forbid", frozen=True)


def run(
    context: ToolContext,
    params: NeedsInput,
) -> NeedsOutput:
    """Map info_state against tenant profile and report filled/missing/ignored."""
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "profiles"
        / "tenant_default.yaml"
    )
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    system = profile.get("system", {})
    required = system.get("required", [])
    optional = system.get("optional", [])
    allowed = set(required + optional)
    info_state = params.info_state or {}
    filled = [k for k in allowed if k in info_state]
    missing = [k for k in required if k not in info_state]
    ignored = [k for k in info_state.keys() if k not in allowed]
    return NeedsOutput(filled=filled, missing=missing, ignored=ignored)
