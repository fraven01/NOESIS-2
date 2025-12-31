from __future__ import annotations

from typing import Dict, Tuple

from ai_core.tool_contracts.base import tool_context_from_meta


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
    context = tool_context_from_meta(meta)

    result = {
        "received": True,
        "tenant_id": context.scope.tenant_id,
        "case_id": context.business.case_id,  # BREAKING CHANGE: from business_context
    }
    return new_state, result
