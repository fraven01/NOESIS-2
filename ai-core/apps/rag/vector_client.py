"""Simple in-memory vector client for demonstration purposes."""

from __future__ import annotations

from typing import Dict, List

from .filters import strict_filters
from .schemas import Chunk


# In-memory store for chunks. In a real implementation this would be an actual
# vector database such as pgvector or similar.
_STORE: List[Chunk] = []


def upsert_chunks(chunks: List[Chunk]) -> int:
    """Insert chunks into the in-memory store."""

    _STORE.extend(chunks)
    return len(chunks)


def search(query: str, filters: Dict[str, str], top_k: int) -> List[Chunk]:
    """Return chunks matching the tenant and case filters."""

    tenant = filters.get("tenant")
    case = filters.get("case")
    results = [c for c in _STORE if strict_filters(c.meta, tenant, case)]
    return results[:top_k]


__all__ = ["upsert_chunks", "search"]
