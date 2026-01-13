"""Factory helpers for document processing graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from documents.processing_graph import build_document_processing_graph


@dataclass(frozen=True)
class DocumentProcessingDependencies:
    parser: Any
    repository: Any
    storage: Any
    captioner: Any
    chunker: Any
    embedder: Callable[..., Any]


def resolve_document_processing_dependencies(
    *,
    repository: Any,
    embedder: Callable[..., Any],
    parser: Any | None = None,
    storage: Any | None = None,
    captioner: Any | None = None,
    chunker: Any | None = None,
) -> DocumentProcessingDependencies:
    if parser is None:
        from documents.parsers import create_default_parser_dispatcher

        parser = create_default_parser_dispatcher()
    if storage is None:
        from documents.storage import ObjectStoreStorage

        storage = ObjectStoreStorage()
    if captioner is None:
        from documents.captioning import DeterministicCaptioner

        captioner = DeterministicCaptioner()
    if chunker is None:
        from ai_core.rag.chunking import RoutingAwareChunker

        chunker = RoutingAwareChunker()

    return DocumentProcessingDependencies(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=captioner,
        chunker=chunker,
        embedder=embedder,
    )


def build_document_processing_workflow(
    *,
    repository: Any,
    embedder: Callable[..., Any],
    build_graph: Callable[..., Any] | None = None,
    parser: Any | None = None,
    storage: Any | None = None,
    captioner: Any | None = None,
    chunker: Any | None = None,
) -> Any:
    graph, _ = build_document_processing_bundle(
        repository=repository,
        embedder=embedder,
        build_graph=build_graph,
        parser=parser,
        storage=storage,
        captioner=captioner,
        chunker=chunker,
    )
    return graph


def build_document_processing_bundle(
    *,
    repository: Any,
    embedder: Callable[..., Any],
    build_graph: Callable[..., Any] | None = None,
    parser: Any | None = None,
    storage: Any | None = None,
    captioner: Any | None = None,
    chunker: Any | None = None,
) -> tuple[Any, DocumentProcessingDependencies]:
    deps = resolve_document_processing_dependencies(
        repository=repository,
        embedder=embedder,
        parser=parser,
        storage=storage,
        captioner=captioner,
        chunker=chunker,
    )
    builder = build_graph or build_document_processing_graph
    graph = builder(
        parser=deps.parser,
        repository=deps.repository,
        storage=deps.storage,
        captioner=deps.captioner,
        chunker=deps.chunker,
        embedder=deps.embedder,
    )
    return graph, deps
