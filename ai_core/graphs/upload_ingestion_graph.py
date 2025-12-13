"""Upload ingestion graph orchestrating document processing for uploads."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Dict

from ai_core import api as ai_core_api
from ai_core.graphs.transition_contracts import (
    GraphTransition,
    PipelineSection,
    StandardTransitionResult,
    build_delta_section,
    build_guardrail_section,
)
from ai_core.infra.observability import observe_span
from documents.contracts import NormalizedDocument
from documents.pipeline import (
    DocumentPipelineConfig,
    DocumentProcessingContext,
    require_document_components,
)
from documents.processing_graph import (
    DocumentProcessingState,
    build_document_processing_graph,
)
from documents.cli import SimpleDocumentChunker
from django.conf import settings


DEFAULT_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_MIME_ALLOWLIST: tuple[str, ...] = (
    "text/plain",
    "text/markdown",
    "text/html",
    "application/octet-stream",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


class UploadIngestionError(RuntimeError):
    """Raised for unexpected internal errors in the upload ingestion graph."""


class UploadIngestionGraph:
    """High level orchestration for processing uploaded documents."""

    def __init__(
        self,
        *,
        document_service: Any = None,
        repository: Any = None,
        document_persistence: Any = None,
        persistence_handler: Any = None,  # Legacy parameter for backward compatibility
        guardrail_enforcer: (
            Callable[..., ai_core_api.GuardrailDecision] | None
        ) = ai_core_api.enforce_guardrails,
        delta_decider: (
            Callable[..., ai_core_api.DeltaDecision] | None
        ) = ai_core_api.decide_delta,
        quarantine_scanner: (
            Callable[[bytes, Mapping[str, Any]], GraphTransition] | None
        ) = None,
        embedding_handler: Callable[..., ai_core_api.EmbeddingResult] | None = None,
        lifecycle_hook: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._document_service = document_service
        self._repository = repository
        # Support legacy persistence_handler parameter
        self._document_persistence = document_persistence or persistence_handler
        self._guardrail_enforcer = guardrail_enforcer
        self._delta_decider = delta_decider
        self._quarantine_scanner = quarantine_scanner
        if embedding_handler is None:
            embedding_handler = ai_core_api.trigger_embedding
        self._embed = embedding_handler
        self._lifecycle_hook = lifecycle_hook

        # Build dependencies
        components = require_document_components()

        # Parsers
        from documents.parsers import create_default_parser_dispatcher

        parser_dispatcher = create_default_parser_dispatcher()

        # Chunker
        chunker = SimpleDocumentChunker()

        # Storage
        storage_candidate = components.storage
        try:
            storage = storage_candidate()
        except Exception:
            storage = storage_candidate

        # Captioner
        captioner_cls = components.captioner
        try:
            captioner = captioner_cls()
        except TypeError:  # if init args needed or already instance?
            captioner = captioner_cls

        self._storage = storage

        self._document_graph = build_document_processing_graph(
            parser=parser_dispatcher,
            repository=self._repository,
            storage=self._storage,
            captioner=captioner,
            chunker=chunker,
            embedder=self._embed,
            delta_decider=self._delta_decider,
            guardrail_enforcer=self._guardrail_enforcer,
            quarantine_scanner=self._quarantine_scanner,
        )

    @observe_span(name="upload.ingestion.run")
    def run(
        self,
        state: Mapping[str, Any],
        meta: Mapping[str, Any] | None = None,
        run_until: str | None = None
    ) -> Mapping[str, Any]:
        """Execute the upload ingestion graph.

        Args:
            state: The input state dictionary containing `normalized_document_input`.
            meta: Optional context metadata (trace_id, case_id etc).
        """
        if meta is None:
            meta = {}

        # 1. Validate Input (Contract from Worker)
        normalized_input = state.get("normalized_document_input")
        if not normalized_input:
             raise UploadIngestionError("input_missing:normalized_document_input")

        # 2. Re-hydrate normalized doc
        try:
            doc_obj = NormalizedDocument.model_validate(normalized_input)
        except Exception as exc:
             raise UploadIngestionError(f"input_invalid:{exc}") from exc

        # 3. Build Config and Context
        config = self._build_config(state)
        context = self._build_context(state, meta, doc_obj)

        graph_state = DocumentProcessingState(
            document=doc_obj,
            config=config,
            context=context,
            run_until=run_until,
            storage=self._storage,
        )

        # 4. Invoke Graph
        try:
            result_state = self._document_graph.invoke(graph_state)
            if isinstance(result_state, dict):
                result_state = DocumentProcessingState(**result_state)
        except Exception as exc:
            raise UploadIngestionError(f"graph_failed:{str(exc)}") from exc

        # 5. Map results back
        return self._map_result(result_state)

    def _build_config(self, state: Mapping[str, Any]) -> DocumentPipelineConfig:
        return DocumentPipelineConfig(
            enable_upload_validation=True,
            max_bytes=int(getattr(settings, "UPLOAD_MAX_BYTES", DEFAULT_MAX_BYTES)),
            mime_allowlist=tuple(
                getattr(settings, "UPLOAD_ALLOWED_MIME_TYPES", DEFAULT_MIME_ALLOWLIST)
            ),
            enable_asset_captions=False,
            enable_embedding=True,
        )

    def _build_context(
        self,
        state: Mapping[str, Any],
        meta: Mapping[str, Any],
        doc: NormalizedDocument,
    ) -> DocumentProcessingContext:
        # Resolve trace_id from meta first, then state
        trace_id = meta.get("trace_id") or state.get("trace_id")
        case_id = meta.get("case_id") or state.get("case_id")
        
        return DocumentProcessingContext.from_document(
            doc,
            case_id=str(case_id) if case_id else None,
            trace_id=str(trace_id) if trace_id else None,
        )

    def _map_result(self, state: DocumentProcessingState) -> Mapping[str, Any]:
        doc = state.document
        normalized_doc = getattr(doc, "document", doc)
        ref = getattr(normalized_doc, "ref", None)
        document_id = getattr(ref, "document_id", None)
        version = getattr(ref, "version", None)

        # Transition helpers -------------------------------------------------
        def _transition(
            *,
            phase: str,
            decision: str,
            reason: str,
            severity: str = "info",
            context: Mapping[str, Any] | None = None,
            pipeline: PipelineSection | None = None,
            delta: Any | None = None,
            guardrail: Any | None = None,
        ) -> Mapping[str, Any]:
            ctx = {k: v for k, v in (context or {}).items() if v is not None}
            result = StandardTransitionResult(
                phase=phase,  # type: ignore[arg-type]
                decision=decision,
                reason=reason,
                severity=severity,
                context=ctx,
                pipeline=pipeline,
                delta=build_delta_section(delta) if delta is not None else None,
                guardrail=(
                    build_guardrail_section(guardrail)
                    if guardrail is not None
                    else None
                ),
            )
            return result.model_dump()

        blob = getattr(normalized_doc, "blob", None)
        mime_type = getattr(blob, "media_type", None)
        # FileBlob does not carry media_type; rely on allowlist or downstream checks if missing
        accept_context = {
            "mime": mime_type,
            "size_bytes": getattr(blob, "size", None),
            "max_bytes": getattr(state.config, "max_bytes", None),
        }
        mime_allowlist = getattr(state.config, "mime_allowlist", None)
        if mime_allowlist:
            accept_context["mime_allowlist"] = tuple(mime_allowlist)

        transitions: Dict[str, Mapping[str, Any]] = {}
        transitions["accept_upload"] = _transition(
            phase="accept_upload",
            decision="accepted",
            reason="upload_validated",
            context=accept_context,
        )

        delta = state.delta_decision
        guardrail = state.guardrail_decision

        if guardrail and not getattr(guardrail, "allowed", False):
            delta_guardrail_decision = guardrail.decision
            delta_guardrail_reason = guardrail.reason
            delta_guardrail_severity = "error"
        elif delta:
            delta_guardrail_decision = delta.decision
            delta_guardrail_reason = delta.reason
            delta_guardrail_severity = "info"
        else:
            delta_guardrail_decision = "unknown"
            delta_guardrail_reason = "guardrail_delta_missing"
            delta_guardrail_severity = "info"

        transitions["delta_and_guardrails"] = _transition(
            phase="delta_and_guardrails",
            decision=delta_guardrail_decision,
            reason=delta_guardrail_reason,
            severity=delta_guardrail_severity,
            context={"document_id": str(document_id) if document_id else None},
            delta=delta,
            guardrail=guardrail,
        )

        pipeline_section = PipelineSection(
            phase=state.phase,
            run_until=state.run_until,
            error=repr(state.error) if state.error else None,
        )
        transitions["document_pipeline"] = _transition(
            phase="document_pipeline",
            decision="processed" if state.error is None else "error",
            reason=(
                "document_pipeline_completed"
                if state.error is None
                else "document_pipeline_failed"
            ),
            severity="error" if state.error else "info",
            context={"phase": state.phase},
            pipeline=pipeline_section,
        )

        # Final decision -----------------------------------------------------
        decision = "completed"
        reason = "ingestion_finished"
        severity = "info"

        if guardrail and not getattr(guardrail, "allowed", False):
            decision = "skip_guardrail"
            reason = guardrail.reason or "guardrail_denied"
            severity = "error"
        elif delta:
            delta_flag = delta.decision.strip().lower()
            if delta_flag in {"skip", "unchanged", "duplicate", "near_duplicate"}:
                decision = "skip_duplicate"
                reason = delta.reason or "delta_skip"
        if state.error is not None:
            decision = "error"
            reason = "document_pipeline_failed"
            severity = "error"

        telemetry = {
            "phase": state.phase,
            "run_until": state.run_until.value if state.run_until else None,
            "delta_decision": getattr(delta, "decision", None) if delta else None,
            "guardrail_decision": (
                getattr(guardrail, "decision", None) if guardrail else None
            ),
        }

        return {
            "decision": decision,
            "reason": reason,
            "severity": severity,
            "document_id": str(document_id) if document_id else None,
            "version": version,
            "telemetry": {k: v for k, v in telemetry.items() if v is not None},
            "transitions": transitions,
        }


__all__ = ["UploadIngestionGraph", "UploadIngestionError"]
