from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask_prompt
from ai_core.infra.tracing import trace
from ai_core.llm import client


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assess risks for the provided text."""
    prompt = load("assess/risk")
    meta["prompt_version"] = prompt["version"]
    meta_with_version = dict(meta)
    return _run(prompt, state, meta=meta_with_version)


@trace("assess")
def _run(
    prompt: Dict[str, str], state: Dict[str, Any], *, meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = state.get("text", "")
    full_prompt = f"{prompt['text']}\n\n{base}"
    masked = mask_prompt(full_prompt)
    result = client.call("analyze", masked, meta)
    new_state = dict(state)
    new_state["risk"] = result["text"]
    return new_state, {"risk": result["text"], "prompt_version": prompt["version"]}
