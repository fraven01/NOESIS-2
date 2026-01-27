from __future__ import annotations

import yaml
import pytest

from ai_core.agent.contracts.manifest import Manifest, load_manifest


def test_manifest_loads_and_validates_required_fields():
    manifest = load_manifest()

    assert isinstance(manifest, Manifest)
    assert isinstance(manifest.agent_state_schema_version, str)
    assert isinstance(manifest.decision_log_schema_version, str)
    assert isinstance(manifest.harness_artifact_schema_version, str)
    assert isinstance(manifest.tool_context_hash_version, str)
    assert isinstance(manifest.capability_iospec_versions, dict)
    assert isinstance(manifest.flow_versions, dict)


def test_manifest_rejects_missing_required_field(tmp_path):
    data = load_manifest().model_dump()
    data.pop("decision_log_schema_version")

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_manifest(manifest_path)

    assert "decision_log_schema_version" in str(excinfo.value)


def test_manifest_rejects_invalid_semver(tmp_path):
    data = load_manifest().model_dump()
    data["agent_state_schema_version"] = "1.0"

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_manifest(manifest_path)

    message = str(excinfo.value)
    assert "agent_state_schema_version" in message
    assert "semver" in message.lower()
