"""Assessment graph extracting, classifying and assessing."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..nodes import assess as assess_node
from ..nodes import classify, extract


def run(items: Optional[List[str]], meta: Dict) -> Dict:
    text = " ".join(items or [])
    extracted = extract.run(text, meta=meta)
    classified = classify.run(extracted["items"], meta=meta)
    assessed = assess_node.run(classified["labels"], meta=meta)
    gaps = extracted["gaps"] + classified["gaps"] + assessed["gaps"]
    return {
        "assessment": assessed["risk"],
        "gaps": gaps,
        "prompt_version": assessed["prompt_version"],
    }
