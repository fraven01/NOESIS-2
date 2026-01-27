from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ai_core.agent.harness.validate import validate_artifact_v0


def _load_agent_run_module() -> object:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "agent_run.py"
    spec = importlib.util.spec_from_file_location("agent_run", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dev_runner_emits_valid_harness_artifact(tmp_path):
    module = _load_agent_run_module()
    output_path = tmp_path / "artifact.json"
    module.run_agent("dummy_flow", output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    artifact = validate_artifact_v0(payload)

    assert artifact.run_id
    assert artifact.inputs_hash
    assert isinstance(artifact.decision_log, list)
    assert isinstance(artifact.stop_decision, dict)
