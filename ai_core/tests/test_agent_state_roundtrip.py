from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_core.agent.state import AgentState


def _base_state() -> AgentState:
    return AgentState(
        flow_name="dummy_flow",
        flow_version="0.1.0",
        decision_log=[{"event": "started"}],
        plan={"step": "noop"},
        checkpoint={"node": "start"},
        identity={"user_id": "user-1"},
    )


def test_agent_state_roundtrip_json():
    state = _base_state()
    payload = state.to_json()
    restored = AgentState.from_json(payload)

    assert restored == state


def test_agent_state_rejects_unknown_fields():
    payload = _base_state().model_dump()
    payload["unknown"] = "nope"

    with pytest.raises(ValidationError):
        AgentState.model_validate(payload)


def test_no_implicit_dict_merge_helpers_used():
    state_path = Path(__file__).resolve().parents[1] / "agent" / "state.py"
    content = state_path.read_text(encoding="utf-8").lower()

    assert "merge" not in content
