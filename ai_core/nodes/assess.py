from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.nodes._prompt_runner import run_prompt_node


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assess risks for the provided text."""
    return run_prompt_node(
        trace_name="assess",
        prompt_alias="assess/risk",
        llm_label="analyze",
        state_key="risk",
        state=state,
        meta=meta,
    )
