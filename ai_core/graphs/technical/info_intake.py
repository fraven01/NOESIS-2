from __future__ import annotations

from typing import Dict, Tuple


def run(state: Dict, meta: Dict) -> Tuple[Dict, Dict]:
    """Record incoming meta information.

    Parameters
    ----------
    state:
        Mutable workflow state.
    meta:
        Context containing ``tenant_id``, ``case_id`` and ``trace_id``.

    Returns
    -------
    tuple
        A tuple of ``(new_state, result)``.
    """

    new_state = dict(state)
    new_state.setdefault("meta", meta)
    result = {
        "received": True,
        "tenant_id": meta.get("tenant_id"),
        "case_id": meta.get("case_id"),
    }
    return new_state, result
