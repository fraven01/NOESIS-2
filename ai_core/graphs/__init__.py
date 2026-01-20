from __future__ import annotations

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

"""
ai_core.graphs package.

We keep graph imports lazy to avoid pulling heavy dependency chains during
module import, while still preserving the legacy package-level aliases that
callers rely on.
"""

_LAZY_MODULES: dict[str, str] = {
    "collection_search": "ai_core.graphs.technical.collection_search",
    "cost_tracking": "ai_core.graphs.technical.cost_tracking",
    "document_service": "ai_core.graphs.technical.document_service",
    "framework_analysis_graph": "ai_core.graphs.business.framework_analysis_graph",
    "rag_retrieval": "ai_core.graphs.technical.rag_retrieval",
    "retrieval_augmented_generation": "ai_core.graphs.technical.retrieval_augmented_generation",
    "transition_contracts": "ai_core.graphs.transition_contracts",
    "universal_ingestion_graph": "ai_core.graphs.technical.universal_ingestion_graph",
}

__all__ = sorted(_LAZY_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(_LAZY_MODULES[name])
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + list(_LAZY_MODULES)))


if TYPE_CHECKING:
    from ai_core.graphs.business import framework_analysis_graph  # noqa: F401
    import ai_core.graphs.transition_contracts as transition_contracts  # noqa: F401
    from ai_core.graphs.technical import (
        collection_search,  # noqa: F401
        cost_tracking,  # noqa: F401
        document_service,  # noqa: F401
        rag_retrieval,  # noqa: F401
        retrieval_augmented_generation,  # noqa: F401
        universal_ingestion_graph,  # noqa: F401
    )
