"""Data structures for Retrieval-Augmented Generation."""

from dataclasses import dataclass
from typing import TypedDict


class ChunkMeta(TypedDict):
    """Metadata attached to a chunk."""

    tenant: str
    case: str
    source: str
    hash: str


@dataclass
class Chunk:
    """A piece of content with associated tenant/case metadata."""

    content: str
    meta: ChunkMeta


__all__ = ["Chunk", "ChunkMeta"]
