"""Standardized model usage reporting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Any


@dataclass(frozen=True)
class Usage:
    """Standard usage statistics for a model generation."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_provider_response(cls, response: Any) -> Usage:
        """Create Usage from a raw provider response (duck-typed)."""
        # OpenAI style
        usage = getattr(response, "usage", None)
        if hasattr(usage, "prompt_tokens"):
             return cls(
                 input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                 output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                 total_tokens=getattr(usage, "total_tokens", 0) or 0,
             )
        
        if isinstance(usage, Mapping):
            return cls(
                input_tokens=usage.get("prompt_tokens", 0) or 0,
                output_tokens=usage.get("completion_tokens", 0) or 0,
                total_tokens=usage.get("total_tokens", 0) or 0,
            )

        if isinstance(response, Mapping):
            return cls(
                input_tokens=response.get("prompt_tokens", 0) or 0,
                output_tokens=response.get("completion_tokens", 0) or 0,
                total_tokens=response.get("total_tokens", 0) or 0,
            )


        return cls()

    def __add__(self, other: Any) -> Usage:
        """Accumulate usage."""
        if not isinstance(other, Usage):
            return NotImplemented
        
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=(self.cost_usd or 0.0) + (other.cost_usd or 0.0),
            details={**self.details, **other.details},
        )
