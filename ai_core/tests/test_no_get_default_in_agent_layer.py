from __future__ import annotations

from pathlib import Path


def test_agent_layer_has_no_get_default_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    agent_root = root / "ai_core" / "agent"
    forbidden = ("get_default_router", "get_default_")
    offenders: list[str] = []

    for path in sorted(agent_root.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
