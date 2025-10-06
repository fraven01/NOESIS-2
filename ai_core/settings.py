"""AI Core settings defaults.

This module intentionally keeps the surface minimal and free from side
effects so it can be imported by lightweight helpers such as the hybrid
parameter parser without triggering wider Django configuration.
"""

from __future__ import annotations


class RAG:
    """Namespace for Retrieval-Augmented Generation defaults."""

    TOPK_DEFAULT: int = 5
    TOPK_MAX: int = 10
    TOPK_MIN: int = 1
    HYBRID_ALPHA_DEFAULT: float = 0.7
    MIN_SIM_DEFAULT: float = 0.15


__all__ = ["RAG"]

