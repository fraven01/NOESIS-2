"""Solve graph returning solution options."""

from __future__ import annotations

from typing import Dict

from ..nodes import solutions


def run(issue: str, meta: Dict) -> Dict:
    result = solutions.run(issue, meta=meta)
    return {
        "solutions": result["solutions"],
        "gaps": result["gaps"],
        "prompt_version": result["prompt_version"],
    }
