from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


from ai_core.agent.capabilities import registry as capability_registry
from ai_core.agent.harness.validate import validate_artifact_v0


def _load_agent_run_module() -> object:
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "agent_run.py"
    spec = importlib.util.spec_from_file_location("agent_run", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dev_runner_rag_query_emits_valid_harness_artifact_with_optional_fields(
    tmp_path, monkeypatch
):
    def _fake_execute(name, _ctx, _cfg, payload):
        if name == "rag.retrieve":
            return SimpleNamespace(
                matches=[
                    {"chunk_id": "c1", "score": 0.9, "text": "alpha"},
                    {"chunk_id": "c2", "score": 0.8, "text": "beta"},
                ],
                telemetry={},
                routing=None,
            )
        if name == "rag.compose":
            return SimpleNamespace(
                answer="final answer", used_sources=["c1"], telemetry={}
            )
        if name == "rag.evidence":
            return SimpleNamespace(
                claim_to_citation={"claim": ["c1"]}, ungrounded_claims=[]
            )
        raise AssertionError(f"unexpected capability {name}")

    monkeypatch.setattr(capability_registry, "execute", _fake_execute)

    module = _load_agent_run_module()
    output_path = tmp_path / "artifact.json"
    module.run_agent("rag_query", output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    artifact = validate_artifact_v0(payload)

    assert artifact.retrieval is not None
    assert artifact.answer is not None
    assert artifact.citations is not None
    assert artifact.claim_to_citation is not None
