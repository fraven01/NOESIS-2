from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.nodes._prompt_runner import run_prompt_node


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Classify text regarding co-determination."""
    return run_prompt_node(
        trace_name="classify",
        prompt_alias="classify/mitbestimmung",
        llm_label="classify",
        state_key="classification",
        state=state,
        meta=meta,
    )
