"""Stub draft blocks node selecting prompt by type."""

from __future__ import annotations

from typing import Dict

from ..decorators.tracing import trace
from ...llm import client as llm
from ...prompts import load_prompt
from ...infra import pii

PROMPT_ALIASES = {
    "system": "draft/system",
    "functions": "draft/functions",
    "clauses": "draft/clause_standard",
}


@trace("draft_blocks")
def run(data: Dict, *, meta: Dict) -> Dict[str, list]:
    alias = PROMPT_ALIASES.get(data.get("type", ""), "")
    prompt = load_prompt(alias) if alias else {"version": "", "text": ""}
    masked = pii.mask(prompt["text"])
    try:
        llm.call("draft", masked, {**meta, "prompt_version": prompt["version"]})
    except Exception:
        pass
    return {"blocks": [], "gaps": [], "prompt_version": prompt["version"]}
