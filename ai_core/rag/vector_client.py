from __future__ import annotations

from typing import Dict, Iterable, List

from .filters import strict_match
from .schemas import Chunk


class InMemoryVectorClient:
    """A minimal in-memory vector client for testing."""

    def __init__(self) -> None:
        self._chunks: Dict[str, Chunk] = {}

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            key = chunk.meta.get("hash")
            if key:
                self._chunks[key] = chunk

    def search(
        self, query: str, filters: Dict[str, str], top_k: int = 5
    ) -> List[Chunk]:
        tenant = filters.get("tenant")
        case = filters.get("case")
        query_lower = query.lower()
        results: List[Chunk] = []
        for chunk in self._chunks.values():
            if not strict_match(chunk.meta, tenant, case):
                continue
            if query_lower in chunk.content.lower():
                results.append(chunk)
        return results[:top_k]
