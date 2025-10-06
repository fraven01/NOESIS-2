"""DEMO/INTERNAL; deprecated for MVP.

This module previously hosted the RAG demo graph. It is intentionally left
empty so the MVP build no longer depends on the legacy demo implementation.
Importers receive a clear runtime error when attempting to access removed
attributes.
"""

from __future__ import annotations

from typing import Any

import warnings

warnings.warn(
    "ai_core.graphs.rag_demo is deprecated and no longer shipped with the MVP build.",
    DeprecationWarning,
    stacklevel=2,
)

__all__: list[str] = []


def __getattr__(name: str) -> Any:
    """Raise an explicit error for any attribute access."""

    raise RuntimeError(
        "The rag_demo graph has been removed. Use the production "
        "retrieval_augmented_generation workflow instead."
    )
