from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from ai_core.agent.capabilities import registry as capability_registry
from ai_core.agent.harness.validate import validate_artifact_v0


def _load_module() -> object:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "harness_run.py"
    spec = importlib.util.spec_from_file_location("harness_run", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_harness_runner_emits_report_with_valid_artifacts(tmp_path, monkeypatch):
    def _fake_execute(name, _ctx, _cfg, payload):
        if name == "rag.retrieve":
            return SimpleNamespace(matches=[], telemetry={}, routing=None)
        if name == "rag.compose":
            return SimpleNamespace(answer="answer", used_sources=[], telemetry={})
        if name == "rag.evidence":
            return SimpleNamespace(
                claim_to_citation={"claim": []}, ungrounded_claims=[]
            )
        raise AssertionError(f"unexpected capability {name}")

    monkeypatch.setattr(capability_registry, "execute", _fake_execute)

    module = _load_module()
    output_path = tmp_path / "report.json"
    report = module.run_harness(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["passed_count"] == len(payload["queries"])

    for item in payload["queries"]:
        validate_artifact_v0(item["artifact"])

    assert report["run_id"] == payload["run_id"]
