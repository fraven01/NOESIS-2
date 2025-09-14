from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask
from ai_core.infra.tracing import trace
from ai_core.llm import client


@trace("classify")
def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Classify text regarding co-determination."""
    prompt = load("classify/mitbestimmung")
    meta["prompt_version"] = prompt["version"]
    base = state.get("text", "")
    full_prompt = f"{prompt['text']}\n\n{base}"
    masked = mask(full_prompt)
    result = client.call("classify", masked, meta)
    new_state = dict(state)
    new_state["classification"] = result["text"]
    return new_state, {
        "classification": result["text"],
        "prompt_version": prompt["version"],
    }
