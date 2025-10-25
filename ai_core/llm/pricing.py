"""Pricing helpers for chat completion models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Mapping

# Ensure deterministic rounding for monetary calculations.
getcontext().prec = 28


@dataclass(frozen=True)
class ChatPricing:
    """Per-1K token pricing for prompt and completion tokens."""

    prompt: Decimal
    completion: Decimal


# Central pricing table sourced from vendor rate cards.
_CHAT_PRICING_TABLE: Mapping[str, ChatPricing] = {
    "openai/gpt-5-nano": ChatPricing(
        prompt=Decimal("0.000500"), completion=Decimal("0.001500")
    ),
    "vertex_ai/gemini-2.5-flash": ChatPricing(
        prompt=Decimal("0.000225"), completion=Decimal("0.000675")
    ),
    "vertex_ai/gemini-2.5-pro": ChatPricing(
        prompt=Decimal("0.001250"), completion=Decimal("0.003750")
    ),
}


def calculate_chat_completion_cost(
    model_id: str, prompt_tokens: int | float, completion_tokens: int | float
) -> float:
    """Return the USD cost for a chat completion invocation.

    Pricing is calculated using the configured per-1K token rates. Unknown
    models default to ``0.0`` cost so that callers remain resilient when a
    pricing entry has not yet been defined.
    """

    pricing = _CHAT_PRICING_TABLE.get(model_id)
    if pricing is None:
        return 0.0

    prompt_cost = (Decimal(prompt_tokens) / Decimal(1000)) * pricing.prompt
    completion_cost = (Decimal(completion_tokens) / Decimal(1000)) * pricing.completion
    total = prompt_cost + completion_cost
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


__all__ = ["calculate_chat_completion_cost"]
