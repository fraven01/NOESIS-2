from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.nodes._prompt_runner import run_prompt_node


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract items and facts from text using the LLM."""
    return run_prompt_node(
        trace_name="extract",
        prompt_alias="extract/items",
        llm_label="extract",
        state_key="items",
        state=state,
        meta=meta,
    )
