"""Upload ingestion graph orchestrating document processing for uploads."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping
from uuid import uuid4

from common.constants import DEFAULT_WORKFLOW_PLACEHOLDER
from django.conf import settings

from ai_core import api as ai_core_api
from ai_core.graphs.transition_contracts import (
    GraphTransition,
    PipelineSection,
    StandardTransitionResult,
    build_delta_section,
    build_guardrail_section,
)
from ai_core.infra.observability import observe_span
from documents.api import NormalizedDocumentPayload
from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
)
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

        self._document_graph = build_document_processing_graph(
            parser=parser_dispatcher,
            repository=self._repository,
            storage=storage,
            captioner=captioner,
            chunker=chunker,
            embedder=self._embed,
            delta_decider=self._delta_decider,
            guardrail_enforcer=self._guardrail_enforcer,
            quarantine_scanner=self._quarantine_scanner,
        )

    @observe_span(name="upload.ingestion.run")
    def run(
        self, payload: Mapping[str, Any], run_until: str | None = None
    ) -> Mapping[str, Any]:
        """Execute the upload ingestion graph for *payload*."""

        # 1. Normalize payload to DocumentProcessingState inputs
        normalized_input = self._prepare_upload_document(payload)

        # Config
        config = self._build_config(payload)

        # Context
        context = self._build_context(payload, normalized_input)

        state = DocumentProcessingState(
            document=normalized_input.document,
            config=config,
            context=context,
            run_until=run_until,
        )

        # 2. Invoke Graph
        try:
            result_state = self._document_graph.invoke(state)
            if isinstance(result_state, dict):
                result_state = DocumentProcessingState(**result_state)
        except Exception as exc:
            raise UploadIngestionError(f"graph_failed:{str(exc)}") from exc

        # 3. Map results back to legacy dict
        return self._map_result(result_state)

    def _prepare_upload_document(
        self, payload: Mapping[str, Any]
    ) -> NormalizedDocumentPayload:
        """Construct NormalizedDocumentPayload from raw upload payload."""

        # Extract inputs
        file_bytes = payload.get("file_bytes")
        file_uri = payload.get("file_uri")

        if file_bytes is None and file_uri is None:
            raise UploadIngestionError("payload_missing")

        if file_bytes is None and file_uri:
            raise UploadIngestionError("file_uri_not_supported")

        if isinstance(file_bytes, str):
            binary = file_bytes.encode("utf-8")
        else:
            binary = bytes(file_bytes)

        declared_mime = self._normalize_mime(payload.get("declared_mime"))
        filename = (payload.get("filename") or "upload").strip()
        if not declared_mime and filename:
            guess, _ = mimetypes.guess_type(filename)
            declared_mime = self._normalize_mime(guess) or "application/octet-stream"

        tenant_id = self._require_str(payload, "tenant_id")
        content_hash = hashlib.sha256(binary).hexdigest()

        # Build document object structure
        meta = DocumentMeta(
            tenant_id=tenant_id,
            workflow_id=payload.get("workflow_id") or DEFAULT_WORKFLOW_PLACEHOLDER,
            title=filename,
            tags=list(self._normalize_tags(payload.get("tags"))),
            origin_uri=payload.get("origin_uri") or payload.get("file_uri"),
            # External ref logic from old graph
            external_ref={
                "provider": "upload",
                "uploader_id": self._resolve_str(payload.get("uploader_id")),
                "external_id": self._resolve_str(payload.get("source_key")),
            },
        )

        blob = InlineBlob(
            type="inline",
            media_type=declared_mime or "application/octet-stream",
            base64=base64.b64encode(binary).decode("ascii"),
            sha256=content_hash,
            size=len(binary),
        )

        doc_id = uuid4()
        ref = DocumentRef(
            tenant_id=tenant_id,
            workflow_id=meta.workflow_id,
            document_id=doc_id,
        )

        normalized = NormalizedDocument(
            ref=ref,
            meta=meta,
            blob=blob,
            checksum=content_hash,
            created_at=datetime.now(timezone.utc),
            source="upload",
        )

        # Metadata dictionary for legacy compat
        metadata_map = {
            "tenant_id": tenant_id,
            "workflow_id": meta.workflow_id,
            "case_id": payload.get("case_id"),
            "trace_id": payload.get("trace_id"),
            "source": "upload",
            "visibility": self._resolve_visibility(payload.get("visibility")),
            "filename": filename,
        }

        return NormalizedDocumentPayload(
            document=normalized,
            primary_text="",
            payload_bytes=binary,
            metadata=MappingProxyType(
                {k: v for k, v in metadata_map.items() if v is not None}
            ),
            content_raw="",
            content_normalized="",
        )

    def _build_config(self, payload: Mapping[str, Any]) -> DocumentPipelineConfig:
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
        self, payload: Mapping[str, Any], doc: NormalizedDocumentPayload
    ) -> DocumentProcessingContext:
        return DocumentProcessingContext.from_document(
            doc.document,
            case_id=self._resolve_str(payload.get("case_id")),
            trace_id=self._resolve_str(payload.get("trace_id")),
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
        accept_context = {
            "mime": getattr(blob, "media_type", None),
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

    # Helpers
    def _require_str(self, payload: Any, key: str) -> str:
        value = payload.get(key)
        if not value:
            raise UploadIngestionError(f"{key}_missing")
        return str(value)

    def _resolve_str(self, value: Any) -> str | None:
        return str(value).strip() if value else None

    def _normalize_tags(self, value: Any) -> Iterable[str]:
        if not value:
            return []
        return [str(v).strip() for v in value]

    def _resolve_visibility(self, value: Any) -> str:
        return str(value) if value else "private"

    def _normalize_mime(self, value: Any) -> str | None:
        return str(value).lower() if value else None


__all__ = ["UploadIngestionGraph", "UploadIngestionError"]
