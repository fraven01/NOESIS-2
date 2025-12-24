from __future__ import annotations

from typing import Dict, Tuple


def run(state: Dict, meta: Dict) -> Tuple[Dict, Dict]:
    """Record incoming meta information.

    Parameters
    ----------
    state:
        Mutable workflow state.
    meta:
        Context containing ``scope_context``.

    Returns
    -------
    tuple
        A tuple of ``(new_state, result)``.
    """

    new_state = dict(state)
    new_state.setdefault("meta", meta)
    scope_context = meta.get("scope_context", {})
    result = {
        "received": True,
        "tenant_id": scope_context.get("tenant_id"),
        "case_id": scope_context.get("case_id"),
    }
    return new_state, result
