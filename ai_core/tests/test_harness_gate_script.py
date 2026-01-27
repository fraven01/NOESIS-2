from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module() -> object:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "harness_gate.py"
    spec = importlib.util.spec_from_file_location("harness_gate", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_harness_gate_exits_nonzero_on_failure(tmp_path, monkeypatch) -> None:
    def _fake_run_harness(_path: Path):
        return {
            "queries": [
                {"id": "q1", "artifact": {"stop_decision": {"status": "failed"}}}
            ]
        }

    def _fake_gate(_artifact):
        return False, ["fail"]

    module = _load_module()
    monkeypatch.setattr(module, "run_harness", _fake_run_harness)
    monkeypatch.setattr(module, "gate_artifact", _fake_gate)

    exit_code = module.main(["--output", str(tmp_path / "report.json")])
    assert exit_code == 1
