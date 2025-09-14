from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Map info_state against tenant profile and report filled/missing/ignored."""
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "profiles"
        / "tenant_default.yaml"
    )
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    system = profile.get("system", {})
    required = system.get("required", [])
    optional = system.get("optional", [])
    allowed = set(required + optional)
    info_state = state.get("info_state", {})
    filled = [k for k in allowed if k in info_state]
    missing = [k for k in required if k not in info_state]
    ignored = [k for k in info_state.keys() if k not in allowed]
    result = {"filled": filled, "missing": missing, "ignored": ignored}
    new_state = dict(state)
    new_state["needs"] = result
    return new_state, result
