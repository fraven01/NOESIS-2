"""LangGraph orchestration primitives for document processing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

try:  # pragma: no cover - exercised via integration tests
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    START = "__start__"
    END = "__end__"

    class _CompiledGraph:
        def __init__(
            self,
            *,
            start: str,
            nodes: Mapping[str, Callable[[Any], Any]],
            transitions: Mapping[str, Mapping[str, Any]],
        ) -> None:
            self._start = start
            self._nodes = nodes
            self._transitions = transitions

        def invoke(self, state: Any) -> Any:
            current = self._start
            while current != END:
                node = self._nodes[current]
                result = node(state)
                if result is not None:
                    state = result
                transition = self._transitions.get(current)
                if not transition:
                    raise RuntimeError(f"transition_missing:{current}")
                kind = transition.get("type")
                if kind == "edge":
                    current = transition["next"]
                elif kind == "conditional":
                    selector = transition["condition"]
                    mapping = transition["mapping"]
                    key = selector(state)
                    try:
                        current = mapping[key]
                    except KeyError as exc:  # pragma: no cover - defensive guard
                        raise RuntimeError(
                            f"conditional_missing:{current}:{key}"
                        ) from exc
                else:  # pragma: no cover - defensive guard
                    raise RuntimeError(f"transition_invalid:{current}:{kind}")
            return state

    class StateGraph:  # pragma: no cover - compatibility shim
        def __init__(self, _state_type: Any) -> None:
            self._nodes: Dict[str, Callable[[Any], Any]] = {}
            self._transitions: Dict[str, MutableMapping[str, Any]] = {}
            self._start: Optional[str] = None

        def add_node(self, name: str, runner: Callable[[Any], Any]) -> None:
            self._nodes[name] = runner

        def add_edge(self, start_node: str, end_node: str) -> None:
            if start_node == START:
                self._start = end_node
            else:
                self._transitions[start_node] = {"type": "edge", "next": end_node}

        def add_conditional_edges(
            self,
            start_node: str,
            condition: Callable[[Any], str],
            edges: Mapping[str, str],
        ) -> None:
            self._transitions[start_node] = {
                "type": "conditional",
                "condition": condition,
                "mapping": dict(edges),
            }

        def compile(self) -> _CompiledGraph:
            if self._start is None:
                raise RuntimeError("graph_start_missing")
            return _CompiledGraph(
                start=self._start, nodes=self._nodes, transitions=self._transitions
            )


class DocumentProcessingPhase(str, Enum):
    """Defines checkpoints that terminate the processing graph."""

    PARSE_ONLY = "parse_complete"
    PARSE_AND_PERSIST = "persist_complete"
    PARSE_PERSIST_AND_CAPTION = "caption_complete"
    FULL = "full"

    @classmethod
    def coerce(
        cls, value: Optional[str | "DocumentProcessingPhase"]
    ) -> "DocumentProcessingPhase":
        if value is None:
            return cls.FULL
        if isinstance(value, cls):
            return value
        candidate = str(value).strip().lower()
        normalised = candidate.replace("-", "_")
        alias = _PHASE_ALIAS_MAP.get(normalised)
        if alias is not None:
            return alias
        for member in cls:
            if member.name.lower() == normalised or member.value.lower() == normalised:
                return member
        raise ValueError(f"invalid_run_until:{value}")


_PHASE_ALIAS_MAP: Dict[str, DocumentProcessingPhase] = {
    "parse": DocumentProcessingPhase.PARSE_ONLY,
    "parse_complete": DocumentProcessingPhase.PARSE_ONLY,
    "parse-only": DocumentProcessingPhase.PARSE_ONLY,
    "parse_only": DocumentProcessingPhase.PARSE_ONLY,
    "preview": DocumentProcessingPhase.PARSE_AND_PERSIST,
    "persist": DocumentProcessingPhase.PARSE_AND_PERSIST,
    "persist_complete": DocumentProcessingPhase.PARSE_AND_PERSIST,
    "parse_and_persist": DocumentProcessingPhase.PARSE_AND_PERSIST,
    "review": DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION,
    "caption": DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION,
    "caption_complete": DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION,
    "parse_persist_and_caption": DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION,
    "full": DocumentProcessingPhase.FULL,
}


_CHECKPOINT_ORDER = {
    "parse_complete": 0,
    "persist_complete": 1,
    "caption_complete": 2,
}

_PHASE_LIMIT_CHECKPOINT = {
    DocumentProcessingPhase.PARSE_ONLY: "parse_complete",
    DocumentProcessingPhase.PARSE_AND_PERSIST: "persist_complete",
    DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION: "caption_complete",
    DocumentProcessingPhase.FULL: None,
}


@dataclass
class DocumentProcessingState:
    """Mutable state propagated through the document processing LangGraph."""

    document: Any
    config: Any
    context: Any
    parsed_result: Optional[Any] = None
    parse_artifact: Optional[Any] = None
    chunk_artifact: Optional[Any] = None
    run_until: DocumentProcessingPhase = DocumentProcessingPhase.FULL
    phase: str = "initial"
    error: Optional[BaseException] = None

    def __post_init__(self) -> None:
        self.run_until = DocumentProcessingPhase.coerce(self.run_until)

    def mark_phase(self, phase: str) -> None:
        self.phase = phase

    def should_stop(self, checkpoint: str) -> bool:
        limit_checkpoint = _PHASE_LIMIT_CHECKPOINT[self.run_until]
        if limit_checkpoint is None:
            return False
        limit_index = _CHECKPOINT_ORDER[limit_checkpoint]
        checkpoint_index = _CHECKPOINT_ORDER.get(checkpoint)
        if checkpoint_index is None:
            return False
        return checkpoint_index >= limit_index


def build_document_processing_graph(
    parser: Any,
    repository: Any,
    storage: Any,
    captioner: Any,
    chunker: Any,
    *,
    propagate_errors: bool = True,
):
    """Return a compiled LangGraph coordinating document processing phases."""

    graph = StateGraph(DocumentProcessingState)

    def _with_error_capture(
        name: str,
        runner: Callable[[DocumentProcessingState], Any],
        *,
        propagate: bool = True,
    ):
        def _wrapped(state: DocumentProcessingState) -> DocumentProcessingState:
            if state.error is not None:
                state.mark_phase(name)
                return state
            state.mark_phase(name)
            try:
                result = runner(state)
            except Exception as exc:  # pragma: no cover - propagated to caller
                state.error = exc
                if propagate:
                    raise
                return state
            return result or state

        return _wrapped

    def _parse_document(state: DocumentProcessingState) -> DocumentProcessingState:
        from . import pipeline as pipeline_module

        context = state.context
        current_state = context.state
        config = state.config
        metadata = context.metadata

        chunk_done = pipeline_module._state_rank(
            current_state
        ) >= pipeline_module._state_rank(pipeline_module.ProcessingState.CHUNKED)
        assets_done = pipeline_module._state_rank(
            current_state
        ) >= pipeline_module._state_rank(
            pipeline_module.ProcessingState.ASSETS_EXTRACTED
        )

        should_parse = not chunk_done or not assets_done

        if should_parse:

            def _parse_action() -> Any:
                pipeline_module.log_extra_entry(phase="parse")
                result = parser.parse(state.document, config)
                pipeline_module.log_extra_exit(
                    parsed_blocks=len(result.text_blocks),
                    parsed_assets=len(result.assets),
                )
                return result

            parsed_result = pipeline_module._run_phase(
                "parse.dispatch",
                "pipeline.parse",
                workflow_id=metadata.workflow_id,
                attributes={
                    "tenant_id": metadata.tenant_id,
                    "document_id": str(metadata.document_id),
                },
                action=_parse_action,
            )
            state.parsed_result = parsed_result
        else:
            state.parsed_result = None
        return state

    def _persist_artifacts(state: DocumentProcessingState) -> DocumentProcessingState:
        from . import pipeline as pipeline_module

        context = state.context
        metadata = context.metadata
        workflow_id = metadata.workflow_id
        parsed_result = state.parsed_result

        if parsed_result is not None:
            parse_artifact = pipeline_module.persist_parsed_document(
                context,
                state.document,
                parsed_result,
                repository=repository,
                storage=storage,
            )
            state.parse_artifact = parse_artifact
            state.context = parse_artifact.asset_context
            stats = parse_artifact.statistics
            ocr_triggers = len(stats.get("ocr.triggered_pages", []) or [])
            pipeline_module._observe_counts(
                workflow_id=workflow_id,
                blocks=int(stats.get("parse.blocks.total", 0)),
                assets=int(stats.get("parse.assets.total", 0)),
                ocr_triggers=ocr_triggers,
            )
        else:
            state.parse_artifact = None

        stored = repository.get(
            metadata.tenant_id,
            metadata.document_id,
            version=metadata.version,
            workflow_id=workflow_id,
        )
        if stored is None:
            if parsed_result is not None:
                raise RuntimeError("document_missing_after_parse")
            stored = state.document
        state.document = stored

        stats_state = pipeline_module._state_from_stats(
            dict(getattr(stored.meta, "parse_stats", {}) or {})
        )
        if stats_state and (
            pipeline_module._state_rank(stats_state)
            > pipeline_module._state_rank(state.context.state)
        ):
            state.context = state.context.transition(stats_state)

        return state

    def _caption_assets(state: DocumentProcessingState) -> DocumentProcessingState:
        from . import pipeline as pipeline_module
        from .captioning import AssetExtractionPipeline
        from .contract_utils import is_image_mediatype

        config = state.config
        context = state.context
        metadata = context.metadata
        workflow_id = metadata.workflow_id

        caption_done = pipeline_module._state_rank(
            context.state
        ) >= pipeline_module._state_rank(pipeline_module.ProcessingState.CAPTIONED)

        if config.enable_asset_captions and not caption_done:

            def _caption_action() -> Any:
                pipeline_module.log_extra_entry(phase="caption")
                pipeline = AssetExtractionPipeline(
                    repository=repository,
                    storage=storage,
                    captioner=captioner,
                    config=config,
                )
                result = pipeline.process_document(state.document)
                pipeline_module.log_extra_exit(asset_count=len(result.assets))
                return result

            current_document = pipeline_module._run_phase(
                "assets.caption.process",
                "pipeline.assets.caption",
                workflow_id=workflow_id,
                attributes={"document_id": str(metadata.document_id)},
                action=_caption_action,
            )

            image_assets = [
                asset
                for asset in current_document.assets
                if is_image_mediatype(getattr(asset, "media_type", ""))
            ]
            attempts = len(image_assets)
            hits = sum(
                1 for asset in image_assets if asset.caption_method == "vlm_caption"
            )
            ocr_fallbacks = sum(
                1 for asset in image_assets if asset.caption_method == "ocr_only"
            )
            hit_rate = hits / attempts if attempts else 0.0
            updates = {
                "caption.state": pipeline_module.ProcessingState.CAPTIONED.value,
                "caption.total_assets": attempts,
                "caption.vlm_hits": hits,
                "caption.ocr_fallbacks": ocr_fallbacks,
                "caption.hit_rate": hit_rate,
            }
            updated_document = pipeline_module._update_document_stats(
                current_document,
                updates,
                repository=repository,
                workflow_id=workflow_id,
            )
            state.document = updated_document
            state.context = state.context.transition(
                pipeline_module.ProcessingState.CAPTIONED
            )
            pipeline_module._observe_caption_metrics(
                workflow_id=workflow_id, hits=hits, attempts=attempts
            )
        elif not config.enable_asset_captions and not caption_done:
            updated_document = pipeline_module._update_document_stats(
                state.document,
                {"caption.state": pipeline_module.ProcessingState.CAPTIONED.value},
                repository=repository,
                workflow_id=workflow_id,
            )
            state.document = updated_document
            state.context = state.context.transition(
                pipeline_module.ProcessingState.CAPTIONED
            )

        return state

    def _chunk_document(state: DocumentProcessingState) -> DocumentProcessingState:
        from . import pipeline as pipeline_module

        context = state.context
        metadata = context.metadata
        workflow_id = metadata.workflow_id

        chunk_done = pipeline_module._state_rank(
            context.state
        ) >= pipeline_module._state_rank(pipeline_module.ProcessingState.CHUNKED)
        if chunk_done:
            return state

        parsed_result = state.parsed_result
        if parsed_result is None:

            def _parse_refresh() -> Any:
                pipeline_module.log_extra_entry(phase="parse")
                result = parser.parse(state.document, state.config)
                pipeline_module.log_extra_exit(
                    parsed_blocks=len(result.text_blocks),
                    parsed_assets=len(result.assets),
                )
                return result

            parsed_result = pipeline_module._run_phase(
                "parse.dispatch",
                "pipeline.parse",
                workflow_id=workflow_id,
                attributes={
                    "tenant_id": metadata.tenant_id,
                    "document_id": str(metadata.document_id),
                    "retry": True,
                },
                action=_parse_refresh,
            )
            state.parsed_result = parsed_result

        def _chunk_action() -> Any:
            pipeline_module.log_extra_entry(phase="chunk")
            result = chunker.chunk(
                state.document,
                parsed_result,
                context=context,
                config=state.config,
            )
            chunks, stats = result
            pipeline_module.log_extra_exit(chunk_count=len(chunks))
            return result

        chunks, chunk_stats = pipeline_module._run_phase(
            "chunk.generate",
            "pipeline.chunk",
            workflow_id=workflow_id,
            attributes={"document_id": str(metadata.document_id)},
            action=_chunk_action,
        )

        chunk_stats = dict(chunk_stats or {})
        chunk_stats["chunk.state"] = pipeline_module.ProcessingState.CHUNKED.value
        chunk_stats.setdefault("chunk.count", len(chunks))

        updated_document = pipeline_module._update_document_stats(
            state.document,
            chunk_stats,
            repository=repository,
            workflow_id=workflow_id,
        )
        state.document = updated_document
        state.context = state.context.transition(
            pipeline_module.ProcessingState.CHUNKED
        )

        from .pipeline import DocumentChunkArtifact

        state.chunk_artifact = DocumentChunkArtifact(
            context=state.context,
            chunks=tuple(chunks),
            statistics=chunk_stats,
        )
        state.mark_phase("chunk_complete")
        return state

    def _checkpoint_router(checkpoint: str) -> Callable[[DocumentProcessingState], str]:
        def _route(state: DocumentProcessingState) -> str:
            if state.error is not None:
                return "stop"
            return "stop" if state.should_stop(checkpoint) else "continue"

        return _route

    def _checkpoint_node(
        name: str,
    ) -> Callable[[DocumentProcessingState], DocumentProcessingState]:
        def _runner(state: DocumentProcessingState) -> DocumentProcessingState:
            state.mark_phase(name)
            return state

        return _runner

    graph.add_node(
        "parse_document",
        _with_error_capture(
            "parse_document", _parse_document, propagate=propagate_errors
        ),
    )
    graph.add_node(
        "parse_complete",
        _with_error_capture(
            "parse_complete", _checkpoint_node("parse_complete"), propagate=False
        ),
    )
    graph.add_node(
        "persist_artifacts",
        _with_error_capture(
            "persist_artifacts", _persist_artifacts, propagate=propagate_errors
        ),
    )
    graph.add_node(
        "persist_complete",
        _with_error_capture(
            "persist_complete", _checkpoint_node("persist_complete"), propagate=False
        ),
    )
    graph.add_node(
        "caption_assets",
        _with_error_capture(
            "caption_assets", _caption_assets, propagate=propagate_errors
        ),
    )
    graph.add_node(
        "caption_complete",
        _with_error_capture(
            "caption_complete", _checkpoint_node("caption_complete"), propagate=False
        ),
    )
    graph.add_node(
        "chunk_document",
        _with_error_capture(
            "chunk_document", _chunk_document, propagate=propagate_errors
        ),
    )

    graph.add_edge(START, "parse_document")
    graph.add_edge("parse_document", "parse_complete")
    graph.add_conditional_edges(
        "parse_complete",
        _checkpoint_router("parse_complete"),
        {"continue": "persist_artifacts", "stop": END},
    )
    graph.add_edge("persist_artifacts", "persist_complete")
    graph.add_conditional_edges(
        "persist_complete",
        _checkpoint_router("persist_complete"),
        {"continue": "caption_assets", "stop": END},
    )
    graph.add_edge("caption_assets", "caption_complete")
    graph.add_conditional_edges(
        "caption_complete",
        _checkpoint_router("caption_complete"),
        {"continue": "chunk_document", "stop": END},
    )
    graph.add_edge("chunk_document", END)

    return graph.compile()


__all__ = [
    "DocumentProcessingPhase",
    "DocumentProcessingState",
    "build_document_processing_graph",
]
