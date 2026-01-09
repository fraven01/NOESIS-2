"""State-of-the-art chunking strategies for RAG systems.

This package implements hybrid chunking combining:
- Late Chunking: Embed full documents first, then chunk with contextual embeddings
- Agentic Chunking: LLM-driven boundary detection for complex documents

See README.md for architecture and usage details.
"""

from __future__ import annotations

__all__ = [
    # New SOTA chunkers
    "HybridChunker",
    "RoutingAwareChunker",
    "LateChunker",
    "ChunkerMode",
    "ChunkerConfig",
    "get_default_chunker_config",
    "get_chunker_config_from_routing",
    # Legacy exports (backward compatibility)
    "SectionChunkPlan",
    "SemanticChunker",
    "SemanticTextBlock",
]

from .hybrid_chunker import (
    HybridChunker,
    RoutingAwareChunker,
    ChunkerMode,
    ChunkerConfig,
    get_default_chunker_config,
    get_chunker_config_from_routing,
)
from .late_chunker import LateChunker

# Backward compatibility: re-export old chunker classes
from ..semantic_chunker import SectionChunkPlan, SemanticChunker, SemanticTextBlock

# AgenticChunker will be added in Phase 2
