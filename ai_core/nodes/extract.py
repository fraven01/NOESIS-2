from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask_prompt
from ai_core.infra.tracing import trace
from ai_core.llm import client


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract items and facts from text using the LLM."""
    prompt = load("extract/items")
    meta["prompt_version"] = prompt["version"]
    meta_with_version = dict(meta)
    return _run(prompt, state, meta=meta_with_version)


@trace("extract")
def _run(
    prompt: Dict[str, str], state: Dict[str, Any], *, meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = state.get("text", "")
    full_prompt = f"{prompt['text']}\n\n{base}"
    masked = mask_prompt(full_prompt)
    result = client.call("extract", masked, meta)
    new_state = dict(state)
    new_state["items"] = result["text"]
    return new_state, {"items": result["text"], "prompt_version": prompt["version"]}
