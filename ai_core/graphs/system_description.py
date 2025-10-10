from __future__ import annotations

from typing import Dict, Tuple


def run(state: Dict, meta: Dict) -> Tuple[Dict, Dict]:
    """Produce a system description only if no information is missing."""

    new_state = dict(state)
    missing = new_state.get("missing") or []
    if missing:
        return new_state, {"skipped": True, "missing": missing}

    description = f"System for tenant {meta['tenant_id']} case {meta['case_id']}"
    new_state["description"] = description
    return new_state, {"description": description}
