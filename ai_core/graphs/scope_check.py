from __future__ import annotations

from typing import Dict, Tuple


def run(state: Dict, meta: Dict) -> Tuple[Dict, Dict]:
    """Validate that the workflow scope is defined.

    The scope check is a gate that never produces a draft.
    Missing items are recorded under ``missing``.
    """

    new_state = dict(state)
    missing = []
    if not new_state.get("scope"):
        missing.append("scope")
    new_state["missing"] = missing
    result = {"missing": missing}
    return new_state, result
