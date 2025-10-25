from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Mapping

# Maintain deterministic rounding for monetary calculations.
getcontext().prec = 28


@dataclass(frozen=True)
class EmbeddingPricing:
    """Per-1K token pricing for embedding models."""

    tokens: Decimal


# Central pricing table derived from vendor documentation. The list is not
# exhaustive; unknown models default to ``0.0`` cost so that ingestion remains
# resilient when pricing entries lag behind new deployments.
_EMBEDDING_PRICING_TABLE: Mapping[str, EmbeddingPricing] = {
    "openai/text-embedding-3-small": EmbeddingPricing(tokens=Decimal("0.000020")),
    "openai/text-embedding-3-large": EmbeddingPricing(tokens=Decimal("0.000130")),
    "vertex_ai/textembedding-gecko@001": EmbeddingPricing(tokens=Decimal("0.000100")),
}


def calculate_embedding_cost(model_id: str, tokens: int | float) -> float:
    """Return the USD cost for embedding the provided number of tokens."""

    if tokens <= 0:
        return 0.0

    pricing = _EMBEDDING_PRICING_TABLE.get(model_id)
    if pricing is None:
        return 0.0

    total = (Decimal(tokens) / Decimal(1000)) * pricing.tokens
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


__all__ = ["calculate_embedding_cost"]
