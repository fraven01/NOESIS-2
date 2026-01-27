from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ai_core.agent.harness.schema import ArtifactV0


def validate_artifact_v0(obj: dict[str, Any]) -> ArtifactV0:
    return ArtifactV0.model_validate(obj)


__all__ = ["ArtifactV0", "ValidationError", "validate_artifact_v0"]
