"""Stub compose node for precheck scoring."""

from __future__ import annotations

from typing import Dict, List

from ..decorators.tracing import trace
from ...llm import client as llm
from ...prompts import load_prompt
from ...infra import pii

PROMPT_ALIAS = "precheck/score"


@trace("compose_precheck")
def run(chunks: List[str], *, meta: Dict) -> Dict[str, List]:
    prompt = load_prompt(PROMPT_ALIAS)
    masked = pii.mask(prompt["text"])
    try:
        llm.call("analyze", masked, {**meta, "prompt_version": prompt["version"]})
    except Exception:
        pass
    return {"score": 0, "gaps": [], "prompt_version": prompt["version"]}
