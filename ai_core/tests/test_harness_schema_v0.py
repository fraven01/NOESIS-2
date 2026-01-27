from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_core.agent.harness.schema import ArtifactV0
from ai_core.agent.harness.validate import validate_artifact_v0


def _minimal_payload() -> dict[str, object]:
    return {
        "run_id": "run-123",
        "inputs_hash": "hash-abc",
        "decision_log": [{"event": "started"}],
        "stop_decision": {"reason": "complete"},
    }


def test_harness_artifact_v0_valid_minimal():
    artifact = validate_artifact_v0(_minimal_payload())

    assert isinstance(artifact, ArtifactV0)
    assert artifact.run_id == "run-123"


def test_harness_artifact_v0_rejects_missing_run_id():
    payload = _minimal_payload()
    payload.pop("run_id")

    with pytest.raises(ValidationError):
        validate_artifact_v0(payload)


def test_harness_artifact_v0_serialization_roundtrip():
    artifact = validate_artifact_v0(_minimal_payload())
    serialized = artifact.model_dump()

    assert validate_artifact_v0(serialized) == artifact
