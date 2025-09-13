"""Stub compose node for assembling answers."""

from __future__ import annotations

from typing import Dict, List

from ..decorators.tracing import trace
from ...llm import client as llm
from ...prompts import load_prompt
from ...infra import pii

PROMPT_ALIAS = "retriever/answer"


@trace("compose")
def run(chunks: List[str], *, meta: Dict) -> Dict[str, List]:
    prompt = load_prompt(PROMPT_ALIAS)
    masked = pii.mask(prompt["text"])
    try:
        llm.call("synthesize", masked, {**meta, "prompt_version": prompt["version"]})
    except Exception:
        pass
    text = " ".join(chunks)
    return {
        "text": text,
        "citations": [],
        "gaps": [],
        "prompt_version": prompt["version"],
    }
