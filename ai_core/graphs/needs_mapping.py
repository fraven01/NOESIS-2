from __future__ import annotations

from typing import Dict, Tuple


def run(state: Dict, meta: Dict) -> Tuple[Dict, Dict]:
    """Map provided needs and stop early if information is missing."""

    new_state = dict(state)
    missing = new_state.get("missing") or []
    if missing:
        # Early exit when previous steps reported missing information.
        return new_state, {"missing": missing}

    needs_input = new_state.get("needs_input")
    if not needs_input:
        new_state["missing"] = ["needs_input"]
        return new_state, {"missing": ["needs_input"]}

    new_state["needs"] = needs_input
    new_state["missing"] = []
    return new_state, {"mapped": True}
