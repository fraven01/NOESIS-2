from __future__ import annotations

from typing import Dict, Tuple


def run(state: Dict, meta: Dict) -> Tuple[Dict, Dict]:
    """Record incoming meta information.

    BREAKING CHANGE (Option A - Strict Separation):
    case_id is a business identifier, extracted from business_context.

    Parameters
    ----------
    state:
        Mutable workflow state.
    meta:
        Context containing ``scope_context`` and ``business_context``.

    Returns
    -------
    tuple
        A tuple of ``(new_state, result)``.
    """

    new_state = dict(state)
    new_state.setdefault("meta", meta)
    scope_context = meta.get("scope_context", {})
    # BREAKING CHANGE (Option A): Extract business_context for business IDs
    business_context = meta.get("business_context", {})

    result = {
        "received": True,
        "tenant_id": scope_context.get("tenant_id"),
        "case_id": business_context.get(
            "case_id"
        ),  # BREAKING CHANGE: from business_context
    }
    return new_state, result
