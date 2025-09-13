"""Stub extract node."""

from __future__ import annotations

from typing import Dict, List

from ..decorators.tracing import trace
from ...llm import client as llm
from ...prompts import load_prompt
from ...infra import pii

PROMPT_ALIAS = "extract/items"


@trace("extract")
def run(text: str, *, meta: Dict) -> Dict[str, List]:
    prompt = load_prompt(PROMPT_ALIAS)
    masked = pii.mask(prompt["text"])
    try:
        llm.call("extract", masked, {**meta, "prompt_version": prompt["version"]})
    except Exception:
        pass
    return {"items": [], "gaps": [], "prompt_version": prompt["version"]}
