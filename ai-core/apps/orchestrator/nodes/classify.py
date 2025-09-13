"""Stub classify node."""

from __future__ import annotations

from typing import Dict, List

from ..decorators.tracing import trace
from ...llm import client as llm
from ...prompts import load_prompt
from ...infra import pii

PROMPT_ALIAS = "classify/mitbestimmung"


@trace("classify")
def run(items: List[str], *, meta: Dict) -> Dict[str, List]:
    prompt = load_prompt(PROMPT_ALIAS)
    masked = pii.mask(prompt["text"])
    try:
        llm.call("classify", masked, {**meta, "prompt_version": prompt["version"]})
    except Exception:
        pass
    return {"labels": [], "gaps": [], "prompt_version": prompt["version"]}
