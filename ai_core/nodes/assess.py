from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask
from ai_core.infra.tracing import trace
from ai_core.llm import client


@trace("assess")
def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assess risk based on extracted facts."""
    prompt = load("assess/risk")
    meta["prompt_version"] = prompt["version"]
    base = state.get("text", "")
    full_prompt = f"{prompt['text']}\n\n{base}"
    masked = mask(full_prompt)
    result = client.call("analyze", masked, meta)
    new_state = dict(state)
    new_state["risk"] = result["text"]
    return new_state, {"risk": result["text"], "prompt_version": prompt["version"]}
