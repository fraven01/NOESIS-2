from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask_prompt
from ai_core.infra.tracing import trace
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


@trace("draft_blocks")
def _run(
    system: Dict[str, str],
    functions: Dict[str, str],
    clause: Dict[str, str],
    state: Dict[str, Any],
    *,
    meta: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    combined = "\n".join([system["text"], functions["text"], clause["text"]])
    masked = mask_prompt(combined)
    result = client.call("draft", masked, meta)
    new_state = dict(state)
    new_state["draft"] = result["text"]
    return new_state, {"draft": result["text"], "prompt_version": clause["version"]}
