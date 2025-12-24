"""Bootstrap registration for legacy graph runners."""

from __future__ import annotations

from .adapters import module_runner
from .registry import LazyGraphFactory, register


def bootstrap() -> None:
    """Register the current set of graph runners with the registry.

    The bootstrap process is idempotent because the underlying registry overwrites
    existing entries on repeated registrations.

    NOTE: We use lazy factories (lambdas/functions) to avoid importing the heavy
    graph definitions at module level. This significantly reduces memory usage
    for processes that don't use all graphs (e.g. basic web workers).
    """

    # 1. Info Intake
    def _make_info_intake():
        from ai_core.graphs.technical import info_intake

        return module_runner(info_intake)

    register("info_intake", LazyGraphFactory(_make_info_intake))

    # 2. RAG Graph
    def _make_rag_graph():
        from ai_core.graphs.technical import retrieval_augmented_generation

        return retrieval_augmented_generation.build_graph()

    rag_factory = LazyGraphFactory(_make_rag_graph)
    register("retrieval_augmented_generation", rag_factory)
    register("rag.default", rag_factory)

    # 3. Crawler / Universal Ingestion
    def _make_universal_ingestion():
        from ai_core.graphs.technical import universal_ingestion_graph

        return universal_ingestion_graph.build_universal_ingestion_graph()

    register("crawler.ingestion", LazyGraphFactory(_make_universal_ingestion))

    # 4. Collection Search
    def _make_collection_search():
        from ai_core.graphs.technical import collection_search

        return collection_search.build_graph()

    register("collection_search", LazyGraphFactory(_make_collection_search))
