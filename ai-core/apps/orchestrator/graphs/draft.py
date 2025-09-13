"""Draft graph returning blocks based on type."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..nodes import draft_blocks


def run(draft_type: str, inputs: Optional[Any], meta: Dict) -> Dict:
    data = {"type": draft_type, "inputs": inputs}
    result = draft_blocks.run(data, meta=meta)
    return {
        "draft": result["blocks"],
        "gaps": result["gaps"],
        "prompt_version": result["prompt_version"],
    }
