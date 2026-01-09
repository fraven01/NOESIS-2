"""LangGraph orchestration primitives for document processing."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, MutableMapping, Optional
import base64
import hashlib
import uuid
from datetime import datetime, timezone

if TYPE_CHECKING:
    from documents.api import NormalizedDocumentPayload

from documents.contracts import InlineBlob

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
    CHUNK_AND_EMBED = "embed_complete"
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
    "embed": DocumentProcessingPhase.CHUNK_AND_EMBED,
    "embed_complete": DocumentProcessingPhase.CHUNK_AND_EMBED,
    "chunk_and_embed": DocumentProcessingPhase.CHUNK_AND_EMBED,
    "full": DocumentProcessingPhase.FULL,
}


_CHECKPOINT_ORDER = {
    "parse_complete": 0,
    "persist_complete": 1,
    "caption_complete": 2,
    "embed_complete": 3,
}

_PHASE_LIMIT_CHECKPOINT = {
    DocumentProcessingPhase.PARSE_ONLY: "parse_complete",
    DocumentProcessingPhase.PARSE_AND_PERSIST: "persist_complete",
    DocumentProcessingPhase.PARSE_PERSIST_AND_CAPTION: "caption_complete",
    DocumentProcessingPhase.CHUNK_AND_EMBED: "embed_complete",
    DocumentProcessingPhase.FULL: None,
}


@dataclass
class DocumentProcessingState:
    """Mutable state propagated through the document processing LangGraph."""

    document: Any
    config: Any
    context: Any
    storage: Optional[Any] = None  # Storage service for blob payload decoding
    parsed_result: Optional[Any] = None
    parse_artifact: Optional[Any] = None
    chunk_artifact: Optional[Any] = None
    delta_decision: Optional[Any] = None
    guardrail_decision: Optional[Any] = None
    run_until: DocumentProcessingPhase = DocumentProcessingPhase.FULL
    phase: str = "initial"
    error: Optional[BaseException] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_until = DocumentProcessingPhase.coerce(self.run_until)

    def mark_phase(self, phase: str) -> None:
        self.phase = phase

    def should_stop(self, checkpoint: str) -> bool:
        limit_checkpoint = _PHASE_LIMIT_CHECKPOINT[self.run_until]
        if limit_checkpoint is None:
            logger.info(
                f"should_stop({checkpoint}): run_until={self.run_until.value}, limit=None, result=False"
            )
            return False
        limit_index = _CHECKPOINT_ORDER[limit_checkpoint]
        checkpoint_index = _CHECKPOINT_ORDER.get(checkpoint)
        if checkpoint_index is None:
            logger.info(
                f"should_stop({checkpoint}): checkpoint_index=None, result=False"
            )
            return False
        result = checkpoint_index >= limit_index
        logger.info(
            f"should_stop({checkpoint}): run_until={self.run_until.value}, "
            f"limit={limit_checkpoint}, checkpoint_index={checkpoint_index}, "
            f"limit_index={limit_index}, result={result}"
        )
        return result


logger = logging.getLogger(__name__)


def build_document_processing_graph(
    parser: Any,
    repository: Any,
    storage: Any,
    captioner: Any,
    chunker: Any,
    *,
    embedder: Optional[Callable[..., Any]] = None,
    delta_decider: Optional[Callable[..., Any]] = None,
    guardrail_enforcer: Optional[Callable[..., Any]] = None,
    quarantine_scanner: Optional[Callable[[bytes, Any], Any]] = None,
    propagate_errors: bool = True,
):
    """Return a compiled LangGraph coordinating document processing phases."""

    graph = StateGraph(DocumentProcessingState)

    def _build_normalized_payload(
        document: Any, context: Any, storage: Optional[Any] = None
    ) -> NormalizedDocumentPayload:
        from documents.api import NormalizedDocumentPayload
        from documents.normalization import (
            document_payload_bytes,
            normalized_primary_text,
        )

        # Pass storage to handle FileBlob/ExternalBlob
        payload_bytes = document_payload_bytes(document, storage=storage)
        payload_text = payload_bytes.decode("utf-8", errors="replace")
        primary_text = normalized_primary_text(payload_text)

        metadata = {
            "tenant_id": getattr(getattr(document, "ref", None), "tenant_id", None),
            "workflow_id": getattr(getattr(document, "ref", None), "workflow_id", None),
            "source": getattr(document, "source", None),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        context_metadata = getattr(context, "metadata", None)
        if context_metadata is not None:
            case_id = getattr(context_metadata, "case_id", None)
            if case_id:
                metadata["case_id"] = case_id

        return NormalizedDocumentPayload(
            document=document,
            primary_text=primary_text,
            payload_bytes=payload_bytes,
            metadata=metadata,
            content_raw=payload_text,
            content_normalized=primary_text,
        )

    def _accepts_keyword(handler: Callable[..., Any], keyword: str) -> bool:
        try:
            target = inspect.unwrap(handler)
        except Exception:
            target = handler
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return False
        for parameter in signature.parameters.values():
            if (
                parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.VAR_KEYWORD,
                )
                and parameter.name == keyword
            ):
                return True
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return True
        return False

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

    def _check_delta_and_guardrails(
        state: DocumentProcessingState,
    ) -> DocumentProcessingState:
        from . import pipeline as pipeline_module

        context = state.context
        metadata = context.metadata

        # Check Delta
        if delta_decider:
            # We need the previous document state to decide delta.
            # We fetch using 'prefer_latest=True' to compare against the latest active version.
            baseline_doc = None
            if repository:
                try:
                    baseline_doc = repository.get(
                        metadata.tenant_id,
                        metadata.document_id,
                        prefer_latest=True,
                        workflow_id=metadata.workflow_id,
                    )
                except Exception:
                    # Repository access failed - continue without baseline
                    baseline_doc = None
            # Baseline is expected to be a dict or mapping by decide_delta, usually
            # But the contract says Optional[Mapping[str, Any]].
            # If baseline_doc is NormalizedDocument, we should model_dump it?
            # Or define a protocol?
            # decide_delta uses baseline.get("content_hash"), etc.
            # So a dict is preferred.
            baseline = baseline_doc.model_dump() if baseline_doc else None

            pipeline_module.log_extra_entry(phase="delta")
            # Wrap document as payload for delta_decider
            normalized_payload = _build_normalized_payload(
                state.document, context, storage=state.storage
            )
            decision = delta_decider(
                normalized_document=normalized_payload,
                baseline=baseline,
                frontier_state=getattr(
                    context, "frontier", None
                ),  # If frontier exists in context?
            )
            state.delta_decision = decision
            pipeline_module.log_extra_exit(delta_decision=decision.decision)

        # Check Guardrails
        if guardrail_enforcer:
            pipeline_module.log_extra_entry(phase="guardrail")
            # Wrap document as payload for guardrail_enforcer
            normalized_payload = _build_normalized_payload(
                state.document, context, storage=state.storage
            )
            decision = guardrail_enforcer(normalized_document=normalized_payload)
            state.guardrail_decision = decision
            pipeline_module.log_extra_exit(guardrail_decision=decision.decision)

        return state

    def _accept_upload(state: DocumentProcessingState) -> DocumentProcessingState:
        """Validate upload against constraints and run quarantine scan."""
        if not getattr(state.config, "enable_upload_validation", False):
            return state

        from . import pipeline as pipeline_module

        config = state.config
        doc = state.document

        # Check size
        max_bytes = getattr(config, "max_bytes", None)
        if max_bytes is not None:
            # InlineBlob has known size
            blob_size = getattr(doc.blob, "size", 0)
            if blob_size > max_bytes:
                # We could raise specific error or set state.error
                raise ValueError("file_too_large")

        # Check MIME
        allowed_mimes = getattr(config, "mime_allowlist", None)
        if allowed_mimes:
            mime = getattr(doc.blob, "media_type", "")
            logger.debug(
                "upload.mime_validation",
                extra={
                    "tenant_id": getattr(getattr(doc, "ref", None), "tenant_id", None),
                    "document_id": str(
                        getattr(getattr(doc, "ref", None), "document_id", "")
                    )
                    or None,
                    "mime": mime,
                    "allowed_mimes": list(allowed_mimes),
                    "blob_type": type(doc.blob).__name__,
                },
            )
            if mime:
                mime = mime.lower()
                if mime not in allowed_mimes:
                    raise ValueError(f"mime_not_allowed:{mime}")

        # Quarantine Scan
        if quarantine_scanner:
            pipeline_module.log_extra_entry(phase="quarantine")
            # Resolve binary payload
            from documents.normalization import document_payload_bytes

            binary = document_payload_bytes(doc, storage=state.storage)

            # Context for scanner
            scan_context = {
                "tenant_id": doc.ref.tenant_id,
                "declared_mime": getattr(doc.blob, "media_type", None),
            }

            # Scanner expected to return transition or raise error?
            # Existing UploadIngestionGraph scanner returns GraphTransition
            # Here we expect it to raise if malware found, or return safe signal?
            # Adapt to existing signature: scanner(binary, context) -> transition
            # If transition decision is 'block' or 'deny', we should stop.

            result = quarantine_scanner(binary, scan_context)
            # Result could be GraphTransition object. We check its decision.
            decision = getattr(result, "decision", "proceed")
            if decision != "proceed" and decision != "accepted":
                raise ValueError(f"quarantine_failed:{decision}")

            pipeline_module.log_extra_exit(quarantine_decision=decision)

        return state

    def _stage_document(state: DocumentProcessingState) -> DocumentProcessingState:
        """Download remote blobs to local temporary file for parsing."""
        from .staging import FileStager

        try:
            stager = FileStager()
            state.document = stager.stage(state.document, storage=state.storage)
            return state
        except Exception as exc:
            logger.exception("staging_failed", extra={"error": str(exc)})
            state.error = exc
            if propagate_errors:
                raise
            return state

    def _cleanup_after_run(state: DocumentProcessingState) -> DocumentProcessingState:
        """Cleanup temporary staged files."""
        # This node runs at the end regardless of success/failure if possible,
        # but in standard LangGraph it runs as a step.
        # We rely on it being the final step of the graph.
        from .staging import FileStager
        from documents.contracts import LocalFileBlob

        if isinstance(state.document.blob, LocalFileBlob):
            try:
                stager = FileStager()
                stager.cleanup(state.document)
            except Exception as exc:
                logger.warning("cleanup_failed", extra={"error": str(exc)})

        return state

    def _parse_document(state: DocumentProcessingState) -> DocumentProcessingState:
        """Parse document and store result in state.

        CRITICAL: Avoid circular imports by importing at function scope carefully.
        """
        context = state.context
        config = state.config
        metadata = context.metadata
        existing_result = state.parsed_result

        # Always parse if we don't have a result yet
        should_parse = existing_result is None

        if should_parse:
            try:
                # Import pipeline functions lazily to avoid circular import at module load time
                from .logging_utils import log_extra_entry, log_extra_exit
                from . import pipeline as pipeline_module

                def _parse_action() -> Any:

                    try:
                        log_extra_entry(phase="parse")

                        # Parsing Logic relying on staged document (LocalFileBlob) or InlineBlob
                        result = parser.parse(state.document, config)

                        log_extra_exit(
                            parsed_blocks=len(result.text_blocks),
                            parsed_assets=len(result.assets),
                        )
                        return result
                    except Exception as parse_err:
                        # Log specific parser error but allow higher level to catch
                        logger.exception(
                            "parse_action_failed", extra={"error": str(parse_err)}
                        )
                        raise

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
                logger.info(
                    "parse_completed",
                    extra={
                        "document_id": str(metadata.document_id),
                        "blocks": (
                            len(parsed_result.text_blocks) if parsed_result else 0
                        ),
                        "assets": len(parsed_result.assets) if parsed_result else 0,
                    },
                )
            except Exception as exc:
                logger.exception(
                    "parse_document_failed",
                    extra={
                        "document_id": str(metadata.document_id),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                raise
        elif existing_result is not None:
            # Already have a result, just ensure it's set in state (redundant but safe)
            state.parsed_result = existing_result
        else:
            state.parsed_result = None

        return state

    def _persist_artifacts(state: DocumentProcessingState) -> DocumentProcessingState:
        from . import pipeline as pipeline_module

        context = state.context
        metadata = context.metadata
        workflow_id = metadata.workflow_id
        parsed_result = state.parsed_result

        if repository is None:
            state.parse_artifact = None
            if parsed_result is not None:
                parsed_context = state.context.transition(
                    pipeline_module.ProcessingState.PARSED_TEXT
                )
                state.context = parsed_context.transition(
                    pipeline_module.ProcessingState.ASSETS_EXTRACTED
                )
                stats = dict(getattr(parsed_result, "statistics", {}) or {})
                ocr_triggers = len(stats.get("ocr.triggered_pages", []) or [])
                pipeline_module._observe_counts(
                    workflow_id=workflow_id,
                    blocks=int(stats.get("parse.blocks.total", 0)),
                    assets=int(stats.get("parse.assets.total", 0)),
                    ocr_triggers=ocr_triggers,
                )
            # No repository - just use state.document as-is
            return state

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
            metadata = (
                state.context.metadata
            )  # refresh to align with persisted document_id
            stats = parse_artifact.statistics
            ocr_triggers = len(stats.get("ocr.triggered_pages", []) or [])
            pipeline_module._observe_counts(
                workflow_id=workflow_id,
                blocks=int(stats.get("parse.blocks.total", 0)),
                assets=int(stats.get("parse.assets.total", 0)),
                ocr_triggers=ocr_triggers,
            )
            # Update state with the persisted document (which may have normalized_content)
            if getattr(parse_artifact, "document", None):
                state.document = parse_artifact.document
        else:
            state.parse_artifact = None

        # DEBUG: Log repository get attempt with all details
        logger.info(
            "persist_artifacts_repository_get_attempt",
            extra={
                "repository_type": type(repository).__name__,
                "tenant_id": metadata.tenant_id,
                "document_id": str(metadata.document_id),
                "workflow_id": workflow_id,
                "parsed_result_present": parsed_result is not None,
                "parse_artifact_present": state.parse_artifact is not None,
            },
        )

        # Try to get the persisted document from repository
        # Note: Document model doesn't have a version field, so we don't filter by version
        stored = repository.get(
            metadata.tenant_id,
            metadata.document_id,
            version=metadata.version,
            workflow_id=workflow_id,
        )

        # DEBUG: Log result with details
        logger.info(
            "persist_artifacts_repository_get_result",
            extra={
                "repository_type": type(repository).__name__,
                "stored_is_none": stored is None,
                "stored_type": type(stored).__name__ if stored else None,
                "tenant_id": metadata.tenant_id,
                "document_id": str(metadata.document_id),
            },
        )

        if stored is None:
            # If we actually parsed and persisted, this is an error
            if parsed_result is not None and state.parse_artifact is not None:
                logger.error(
                    "document_missing_after_parse",
                    extra={
                        "tenant_id": metadata.tenant_id,
                        "document_id": str(metadata.document_id),
                        "workflow_id": workflow_id,
                        "parsed_result_present": True,
                        "parse_artifact_present": True,
                    },
                )
                raise RuntimeError("document_missing_after_parse")
            # If we didn't parse (no parsed_result), use state document
            logger.info(
                "no_parse_using_state_document",
                extra={
                    "tenant_id": metadata.tenant_id,
                    "document_id": str(metadata.document_id),
                },
            )
            stored = state.document
        else:
            try:
                from documents.contracts import (
                    NormalizedDocument,
                )  # local import to avoid cycles
            except (
                Exception
            ):  # pragma: no cover - defensive guard for test environments
                NormalizedDocument = None  # type: ignore

            if NormalizedDocument is not None and not isinstance(
                stored, NormalizedDocument
            ):
                # Repository should return a normalized document; fall back to the current state
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
        logger.debug(
            "caption_assets.entered",
            extra={
                "tenant_id": metadata.tenant_id,
                "trace_id": metadata.trace_id,
                "workflow_id": workflow_id,
                "document_id": str(metadata.document_id),
            },
        )

        logger.debug(
            "caption_assets.config",
            extra={
                "tenant_id": metadata.tenant_id,
                "trace_id": metadata.trace_id,
                "workflow_id": workflow_id,
                "document_id": str(metadata.document_id),
                "enable_asset_captions": getattr(
                    config, "enable_asset_captions", False
                ),
            },
        )

        caption_done = pipeline_module._state_rank(
            context.state
        ) >= pipeline_module._state_rank(pipeline_module.ProcessingState.CAPTIONED)
        logger.debug(
            "caption_assets.state",
            extra={
                "tenant_id": metadata.tenant_id,
                "trace_id": metadata.trace_id,
                "workflow_id": workflow_id,
                "document_id": str(metadata.document_id),
                "caption_done": caption_done,
            },
        )

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
        """Chunk parsed document and store result in state."""
        # Import at function scope to avoid circular import issues
        from . import pipeline as pipeline_module
        from .logging_utils import log_extra_entry, log_extra_exit

        context = state.context
        metadata = context.metadata
        workflow_id = metadata.workflow_id

        # Early return if embedding disabled (Store-Only mode)
        # Skip chunking entirely to avoid wasting time on quality evaluation
        if not getattr(state.config, "enable_embedding", True):
            logger.info(
                "chunk_document.skip_store_only",
                extra={
                    "tenant_id": metadata.tenant_id,
                    "trace_id": metadata.trace_id,
                    "workflow_id": workflow_id,
                    "document_id": str(metadata.document_id),
                    "reason": "enable_embedding_false",
                },
            )
            state.mark_phase("chunk_skipped")
            return state

        parsed_result = state.parsed_result
        if parsed_result is None:
            logger.warning(
                "chunk_missing_parsed_result",
                extra={"document_id": str(metadata.document_id)},
            )

            def _parse_refresh() -> Any:
                log_extra_entry(phase="parse")
                result = parser.parse(state.document, state.config)
                log_extra_exit(
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
                },
                action=_parse_refresh,
            )
            state.parsed_result = parsed_result

        try:

            def _chunk_action() -> Any:
                log_extra_entry(phase="chunk")
                result = chunker.chunk(
                    state.document,
                    parsed_result,
                    context=context,
                    config=state.config,
                )
                chunks, stats = result
                log_extra_exit(chunk_count=len(chunks))
                return result

            chunks, chunk_stats = pipeline_module._run_phase(
                "chunk.generate",
                "pipeline.chunk",
                workflow_id=workflow_id,
                attributes={"document_id": str(metadata.document_id)},
                action=_chunk_action,
            )

            # Persist chunks as assets for debugging/vis
            for i, chunk in enumerate(chunks):
                # Fallback for LateChunker (uses 'text') vs other chunkers ('content')
                text_content = chunk.get("content") or chunk.get("text")
                if not text_content:
                    continue

                chunk_id = chunk.get("chunk_id") or str(uuid.uuid4())
                meta = dict(chunk.get("metadata") or {})
                meta["asset_kind"] = "chunk"
                meta["chunk_index"] = i
                if "parent_ref" in chunk:
                    meta["parent_ref"] = chunk["parent_ref"]
                
                try:
                    repository.add_asset(
                        document=state.document,
                        asset_id=chunk_id,
                        media_type="text/plain",
                        content=InlineBlob(data=text_content.encode("utf-8")),
                        metadata=meta,
                        workflow_id=workflow_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "chunk_persistence_failed",
                        extra={
                            "chunk_id": chunk_id,
                            "error": str(exc),
                            "document_id": str(metadata.document_id)
                        }
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

            # --- Persistence of Chunks to DocumentAsset (Debug/Explorer Visibility) ---
            if repository is not None:
                try:
                    # Local imports to avoid circular dependency
                    from documents.contracts import Asset, AssetRef

                    for i, chunk in enumerate(chunks):
                        # Handle both dicts and objects (Pydantic models)
                        content = getattr(chunk, "content", None)
                        if content is None and isinstance(chunk, dict):
                            content = chunk.get("content")
                        
                        if not content:
                            continue

                        # Ensure content is string
                        if not isinstance(content, str):
                            content = str(content)

                        # Generate unique asset ID
                        asset_id = uuid.uuid4()
                        now = datetime.now(timezone.utc)

                        # Create Blob (Inline text)
                        blob_content = content.encode("utf-8")
                        blob_b64 = base64.b64encode(blob_content).decode("ascii")
                        blob_sha = hashlib.sha256(blob_content).hexdigest()

                        blob = InlineBlob(
                            type="inline",
                            media_type="text/plain",
                            base64=blob_b64,
                            sha256=blob_sha,
                            size=len(blob_content),
                        )

                        # Create Reference
                        ref = AssetRef(
                            tenant_id=metadata.tenant_id,
                            workflow_id=workflow_id,
                            document_id=metadata.document_id,
                            collection_id=metadata.collection_id,
                            asset_id=asset_id,
                        )

                        # Create Asset
                        # We use ocr_text to store the content up to 8192 chars for DB searchability
                        # 'asset_kind' helps identify these as chunks
                        asset = Asset(
                            ref=ref,
                            media_type="text/plain",
                            blob=blob,
                            text_description=content,  # Truncated to 2048 by validator
                            ocr_text=content,          # Truncated to 8192 by validator
                            created_at=now,
                            checksum=blob_sha,
                            asset_kind="chunk",
                            caption_method="none",
                        )

                        repository.add_asset(asset, workflow_id=workflow_id)
                        
                except Exception as persist_exc:
                    # Non-blocking failure - log and continue
                    logger.warning(
                        "chunk_persistence_failed", 
                        extra={
                            "document_id": str(metadata.document_id),
                            "error": str(persist_exc)
                        }
                    )
            # --------------------------------------------------------------------------

            from .pipeline import DocumentChunkArtifact

            state.chunk_artifact = DocumentChunkArtifact(
                context=state.context,
                chunks=tuple(chunks),
                statistics=chunk_stats,
            )
            state.mark_phase("chunk_complete")

            logger.info(
                "chunk_completed",
                extra={
                    "tenant_id": metadata.tenant_id,
                    "trace_id": metadata.trace_id,
                    "workflow_id": workflow_id,
                    "document_id": str(metadata.document_id),
                    "chunk_count": len(chunks),
                },
            )

        except Exception as exc:
            logger.exception(
                "chunk_document_failed",
                extra={
                    "tenant_id": metadata.tenant_id,
                    "trace_id": metadata.trace_id,
                    "workflow_id": workflow_id,
                    "document_id": str(metadata.document_id),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise

        return state

    def _embed_chunks(state: DocumentProcessingState) -> DocumentProcessingState:
        """Embed document chunks and index in vector store."""
        from . import pipeline as pipeline_module

        logger.debug(
            "embed_chunks.debug",
            extra={
                "tenant_id": state.context.metadata.tenant_id,
                "trace_id": state.context.metadata.trace_id,
                "workflow_id": state.context.metadata.workflow_id,
                "document_id": str(state.context.metadata.document_id),
                "enable_embedding": getattr(state.config, "enable_embedding", True),
                "embedder_present": embedder is not None,
                "chunk_artifact_present": state.chunk_artifact is not None,
                "chunks_count": (
                    len(state.chunk_artifact.chunks) if state.chunk_artifact else 0
                ),
            },
        )

        if not getattr(state.config, "enable_embedding", True):
            logger.debug(
                "embed_chunks.skip",
                extra={
                    "tenant_id": state.context.metadata.tenant_id,
                    "trace_id": state.context.metadata.trace_id,
                    "workflow_id": state.context.metadata.workflow_id,
                    "document_id": str(state.context.metadata.document_id),
                    "reason": "enable_embedding_false",
                },
            )
            return state

        # If embedder is not provided, we can't embed.
        # But if enable_embedding is True, maybe we should error?
        # For now, we assume embedder is optional in builder but required if logic reached.
        # But we only reach here via graph edges.

        if embedder is None:
            logger.debug(
                "embed_chunks.skip",
                extra={
                    "tenant_id": state.context.metadata.tenant_id,
                    "trace_id": state.context.metadata.trace_id,
                    "workflow_id": state.context.metadata.workflow_id,
                    "document_id": str(state.context.metadata.document_id),
                    "reason": "embedder_missing",
                },
            )
            # Decide if we skip or error.
            # If enabled in config but missing component, it's a configuration error.
            # But making it optional in builder means we might not have it.
            return state

        context = state.context
        metadata = context.metadata
        workflow_id = metadata.workflow_id

        # We need chunk artifact
        if state.chunk_artifact is None:
            # If no chunks, nothing to embed
            return state

        def _embed_action() -> Any:
            pipeline_module.log_extra_entry(phase="embed")
            supports_chunks = _accepts_keyword(embedder, "chunks")
            supports_normalized = _accepts_keyword(embedder, "normalized_document")

            embed_kwargs: Dict[str, Any] = {}

            # Build normalized_document if supported (and likely required by trigger_embedding)
            normalized_payload = None
            if supports_normalized:
                try:
                    normalized_payload = _build_normalized_payload(
                        state.document, state.context, storage=state.storage
                    )
                    embed_kwargs["normalized_document"] = normalized_payload
                except ValueError:
                    # If normalized_document fails but we have chunks, log warning but continue
                    if not (
                        supports_chunks
                        and state.chunk_artifact
                        and state.chunk_artifact.chunks
                    ):
                        # If we can't build normalized AND don't have chunks, we must fail
                        raise

            # Add chunks if supported (in addition to normalized_document)
            if supports_chunks and state.chunk_artifact and state.chunk_artifact.chunks:
                embed_kwargs["chunks"] = state.chunk_artifact.chunks

            if _accepts_keyword(embedder, "context"):
                embed_kwargs["context"] = state.context
            if _accepts_keyword(embedder, "config"):
                embed_kwargs["config"] = state.config

            if _accepts_keyword(embedder, "tenant_id") and metadata.tenant_id:
                embed_kwargs["tenant_id"] = metadata.tenant_id
            case_value = getattr(metadata, "case_id", None)
            if _accepts_keyword(embedder, "case_id") and case_value:
                embed_kwargs["case_id"] = case_value

            profile_value = getattr(state.config, "embedding_profile", None)
            if _accepts_keyword(embedder, "embedding_profile") and profile_value:
                embed_kwargs["embedding_profile"] = profile_value

            if not embed_kwargs:
                # Fallback: Try to build normalized_document as last resort
                normalized_payload = _build_normalized_payload(
                    state.document, state.context, storage=state.storage
                )
                embed_kwargs["normalized_document"] = normalized_payload

            result = embedder(**embed_kwargs)
            # Result could be EmbeddingResult or similar
            count = (
                getattr(result, "chunks_inserted", 0)
                if hasattr(result, "chunks_inserted")
                else 0
            )
            pipeline_module.log_extra_exit(embed_count=count)
            return result

        # Execute embedding
        # Note: pipeline.embed metric event doesn't exist yet, we reuse generic pattern
        # Execute embedding
        # Note: pipeline.embed metric event doesn't exist yet, we reuse generic pattern
        embed_result = pipeline_module._run_phase(
            "embed.generate",
            "pipeline.embed",
            workflow_id=workflow_id,
            attributes={"document_id": str(metadata.document_id)},
            action=_embed_action,
        )

        # Update statistics with insertion count
        if embed_result:
            count = (
                getattr(embed_result, "chunks_inserted", 0)
                if hasattr(embed_result, "chunks_inserted")
                else 0
            )
            # state is frozen, use replace
            from dataclasses import replace

            new_stats = dict(state.statistics)
            new_stats["embedding.inserted"] = count
            state = replace(state, statistics=new_stats)

        return state

    def _checkpoint_node(
        name: str,
    ) -> Callable[[DocumentProcessingState], DocumentProcessingState]:
        def _runner(state: DocumentProcessingState) -> DocumentProcessingState:
            state.mark_phase(name)
            return state

        return _runner

    def _ingestion_router(state: DocumentProcessingState) -> str:
        # Check Delta
        if state.delta_decision:
            if getattr(state.delta_decision, "decision", "") == "skip":
                return "end"  # Or stop

        # Check Guardrails
        if state.guardrail_decision:
            decision = state.guardrail_decision
            allowed_flag = getattr(decision, "allowed", None)
            if allowed_flag is not None:
                if not allowed_flag:
                    return "end"
            else:
                guardrail_value = str(getattr(decision, "decision", "")).strip().lower()
                if guardrail_value in {"block", "blocked", "deny", "denied"}:
                    return "end"

        return "continue"

    def _checkpoint_router(checkpoint: str) -> Callable[[DocumentProcessingState], str]:
        def _route(state: DocumentProcessingState) -> str:
            if state.error is not None:
                return "stop"
            return "stop" if state.should_stop(checkpoint) else "continue"

        return _route

    graph.add_node(
        "accept_upload", _with_error_capture("upload_accepted", _accept_upload)
    )
    graph.add_node(
        "check_delta_guardrails",
        _with_error_capture("guardrail_complete", _check_delta_and_guardrails),
    )
    graph.add_node(
        "parse_document", _with_error_capture("parse_complete", _parse_document)
    )
    graph.add_node(
        "persist_document", _with_error_capture("persist_complete", _persist_artifacts)
    )
    graph.add_node(
        "caption_assets", _with_error_capture("caption_complete", _caption_assets)
    )
    graph.add_node(
        "chunk_document", _with_error_capture("chunk_complete", _chunk_document)
    )
    graph.add_node("embed_chunks", _with_error_capture("embed_complete", _embed_chunks))

    # Adding Staging and Cleanup Nodes
    graph.add_node(
        "stage_document", _with_error_capture("staging_complete", _stage_document)
    )
    graph.add_node("cleanup", _cleanup_after_run)

    # WIRED EDGES
    graph.add_edge(START, "accept_upload")

    # After upload check, move to delta/guardrails
    graph.add_edge("accept_upload", "check_delta_guardrails")

    # Router after delta/guardrails
    # If continue -> go to staging (to download blobs)
    graph.add_conditional_edges(
        "check_delta_guardrails",
        _ingestion_router,
        {"continue": "stage_document", "end": "cleanup"},
    )

    # Staging -> Parse
    graph.add_edge("stage_document", "parse_document")

    # Parse -> Persist
    graph.add_edge("parse_document", "persist_document")

    # Persist -> Caption (conditional)
    graph.add_conditional_edges(
        "persist_document",
        lambda state: ("end" if state.should_stop("persist_complete") else "continue"),
        {"continue": "caption_assets", "end": "cleanup"},
    )

    # Caption -> Chunk (conditional: skip if embedding disabled)
    def _should_chunk(state: DocumentProcessingState) -> str:
        """Determine if chunking should run based on lifecycle limits and config."""
        if state.should_stop("caption_complete"):
            return "end"
        # Store-only mode: skip chunking + embedding when disabled.
        if not getattr(state.config, "enable_embedding", True):
            logger.info(
                "graph.skip_chunk_store_only",
                extra={
                    "document_id": str(state.context.metadata.document_id),
                    "trace_id": state.context.metadata.trace_id,
                    "reason": "enable_embedding_false",
                },
            )
            return "end"
        return "continue"

    graph.add_conditional_edges(
        "caption_assets",
        _should_chunk,
        {"continue": "chunk_document", "end": "cleanup"},
    )

    # Chunk -> Embed
    graph.add_conditional_edges(
        "chunk_document",
        lambda state: ("end" if state.should_stop("chunk_complete") else "continue"),
        {"continue": "embed_chunks", "end": "cleanup"},
    )

    # Embed -> Cleanup (ensures staged files are cleaned up on success)
    graph.add_edge("embed_chunks", "cleanup")

    # Cleanup -> End
    graph.add_edge("cleanup", END)

    return graph.compile()


__all__ = [
    "DocumentProcessingPhase",
    "DocumentProcessingState",
    "build_document_processing_graph",
]
