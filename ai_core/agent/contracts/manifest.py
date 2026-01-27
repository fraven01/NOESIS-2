from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

_SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")


def _ensure_semver(value: str, field_name: str) -> str:
    if not _SEMVER_PATTERN.match(value):
        raise ValueError(f"{field_name} must be a semver string like X.Y.Z")
    return value


class Manifest(BaseModel):
    agent_state_schema_version: str
    decision_log_schema_version: str
    harness_artifact_schema_version: str
    tool_context_hash_version: str
    capability_iospec_versions: dict[str, str]
    flow_versions: dict[str, str]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator(
        "agent_state_schema_version",
        "decision_log_schema_version",
        "harness_artifact_schema_version",
        "tool_context_hash_version",
    )
    @classmethod
    def validate_semver_fields(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "version"
        return _ensure_semver(value, field_name)

    @field_validator("capability_iospec_versions", "flow_versions")
    @classmethod
    def validate_version_maps(
        cls, value: dict[str, str], info: ValidationInfo
    ) -> dict[str, str]:
        field_name = info.field_name or "version_map"
        for key, version in value.items():
            _ensure_semver(version, f"{field_name}.{key}")
        return value


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> Manifest:
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Contract manifest at {manifest_path} must be a mapping object"
        )
    return Manifest.model_validate(raw)


__all__ = ["DEFAULT_MANIFEST_PATH", "Manifest", "load_manifest"]
