"""Stub assess node."""

from __future__ import annotations

from typing import Dict, List

from ..decorators.tracing import trace
from ...llm import client as llm
from ...prompts import load_prompt
from ...infra import pii

PROMPT_ALIAS = "assess/risk"


@trace("assess")
def run(labels: List[str], *, meta: Dict) -> Dict[str, List]:
    prompt = load_prompt(PROMPT_ALIAS)
    masked = pii.mask(prompt["text"])
    try:
        llm.call("analyze", masked, {**meta, "prompt_version": prompt["version"]})
    except Exception:
        pass
    return {"risk": "low", "gaps": [], "prompt_version": prompt["version"]}
