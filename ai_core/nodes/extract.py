from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask
from ai_core.infra.tracing import trace
from ai_core.llm import client


@trace("extract")
def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract items and facts from text using the LLM."""
    prompt = load("extract/items")
    meta["prompt_version"] = prompt["version"]
    base = state.get("text", "")
    full_prompt = f"{prompt['text']}\n\n{base}"
    masked = mask(full_prompt)
    result = client.call("extract", masked, meta)
    new_state = dict(state)
    new_state["items"] = result["text"]
    return new_state, {"items": result["text"], "prompt_version": prompt["version"]}
