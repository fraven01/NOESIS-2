"""Stub retrieve node calling the LLM client."""

from __future__ import annotations

from typing import Dict, List

from ..decorators.tracing import trace
from ...llm import client as llm
from ...prompts import load_prompt
from ...infra import pii

PROMPT_ALIAS = "retriever/answer"


@trace("retrieve")
def run(query: str, *, meta: Dict) -> Dict[str, List]:
    """Return placeholder chunks for a query."""
    prompt = load_prompt(PROMPT_ALIAS)
    masked = pii.mask(prompt["text"])
    try:
        llm.call(
            "simple-query",
            masked,
            {**meta, "prompt_version": prompt["version"]},
        )
    except Exception:
        pass
    return {"chunks": [], "gaps": [], "prompt_version": prompt["version"]}
