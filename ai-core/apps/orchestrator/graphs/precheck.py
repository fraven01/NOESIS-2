"""Precheck graph reusing retrieve and composing a score."""

from __future__ import annotations

from typing import Dict

from ..nodes import compose_precheck, retrieve


def run(context: str, meta: Dict) -> Dict:
    retrieved = retrieve.run(context, meta=meta)
    composed = compose_precheck.run(retrieved["chunks"], meta=meta)
    gaps = retrieved["gaps"] + composed["gaps"]
    return {
        "precheck": composed["score"],
        "gaps": gaps,
        "prompt_version": composed["prompt_version"],
    }
