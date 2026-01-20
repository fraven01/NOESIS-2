"""Service protocols for framework analysis dependencies."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class FrameworkRetrievalService(Protocol):
    """Protocol for retrieval graph integration."""

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Run retrieval graph and return a structured payload."""


class FrameworkLLMService(Protocol):
    """Protocol for LLM prompt execution."""

    def __call__(
        self,
        *,
        prompt_key: str,
        prompt_input: str,
        meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Execute a JSON-capable LLM prompt."""


__all__ = ["FrameworkRetrievalService", "FrameworkLLMService"]
