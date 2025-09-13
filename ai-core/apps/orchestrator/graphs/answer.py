"""Answer graph connecting retrieve and compose nodes."""

from __future__ import annotations

from typing import Dict

from ..nodes import compose, retrieve


def run(question: str, meta: Dict) -> Dict:
    """Run the answer graph with stubbed nodes."""
    retrieved = retrieve.run(question, meta=meta)
    composed = compose.run(retrieved["chunks"], meta=meta)
    gaps = retrieved["gaps"] + composed["gaps"]
    return {
        "answer": composed["text"],
        "citations": composed["citations"],
        "gaps": gaps,
        "prompt_version": composed["prompt_version"],
    }
