from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.mask_prompt import mask_prompt, mask_response
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.prompts import load
from ai_core.infra.observability import observe_span
from ai_core.llm import client


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate draft blocks using system and function prompts."""
    system = load("draft/system")
    functions = load("draft/functions")
    clause = load("draft/clause_standard")
    meta["prompt_version"] = clause["version"]
    meta_with_version = dict(meta)
    return _run(system, functions, clause, state, meta=meta_with_version)


@observe_span(name="draft_blocks")
def _run(
    system: Dict[str, str],
    functions: Dict[str, str],
    clause: Dict[str, str],
    state: Dict[str, Any],
    *,
    meta: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    combined = "\n".join([system["text"], functions["text"], clause["text"]])
    pii_config = get_pii_config()
    masked = mask_prompt(combined, config=pii_config)
    result = client.call("draft", masked, meta)
    draft_text = mask_response(result["text"], config=pii_config)
    new_state = dict(state)
    new_state["draft"] = draft_text
    return new_state, {"draft": draft_text, "prompt_version": clause["version"]}
