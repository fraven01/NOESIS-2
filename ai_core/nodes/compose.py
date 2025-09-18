from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.prompts import load
from ai_core.infra.pii import mask_prompt
from ai_core.infra.tracing import trace
from ai_core.llm import client


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compose an answer from retrieved snippets using the LLM."""
    prompt = load("retriever/answer")
    meta["prompt_version"] = prompt["version"]
    meta_with_version = dict(meta)
    return _run(prompt, state, meta=meta_with_version)


@trace("compose")
def _run(
    prompt: Dict[str, str], state: Dict[str, Any], *, meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snippets_text = "\n".join(s.get("text", "") for s in state.get("snippets", []))
    question = state.get("question", "")
    full_prompt = (
        f"{prompt['text']}\n\nQuestion: {question}\nContext:\n{snippets_text}"
    )
    masked = mask_prompt(full_prompt)
    result = client.call("synthesize", masked, meta)
    new_state = dict(state)
    new_state["answer"] = result["text"]
    return new_state, {"answer": result["text"], "prompt_version": prompt["version"]}
