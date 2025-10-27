"""Bootstrap registration for legacy graph runners."""

from __future__ import annotations

from ai_core.graphs import (
    crawler_ingestion_graph,
    info_intake,
    retrieval_augmented_generation,
)

from .adapters import module_runner
from .registry import register


def bootstrap() -> None:
    """Register the current set of graph runners with the registry.

    The bootstrap process is idempotent because the underlying registry overwrites
    existing entries on repeated registrations.
    """

    register("info_intake", module_runner(info_intake))
    rag_graph = retrieval_augmented_generation.build_graph()
    register("retrieval_augmented_generation", rag_graph)
    register("rag.default", rag_graph)
    crawler_graph = crawler_ingestion_graph.build_graph()
    register("crawler.ingestion", crawler_graph)
