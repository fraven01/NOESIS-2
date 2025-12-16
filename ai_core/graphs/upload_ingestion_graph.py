"""Upload ingestion graph orchestrating document processing for uploads via LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from django.conf import settings
from langgraph.graph import END, START, StateGraph

from ai_core.infra.observability import observe_span
from ai_core import api as ai_core_api
from ai_core.graphs.transition_contracts import (
    StandardTransitionResult,
    PipelineSection,
    build_delta_section,
    build_guardrail_section,
)
from documents.contracts import NormalizedDocument
from documents.pipeline import (
    DocumentPipelineConfig,
    DocumentProcessingContext,
    require_document_components,
)
from documents.processing_graph import (
    DocumentProcessingState,
    DocumentProcessingPhase,
    build_document_processing_graph,
)


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- Protocols
class DocumentRepository(Protocol):
    """Protocol for document repository."""

    def get(self, tenant_id: str, document_id: str, **kwargs: Any) -> Any: ...
    def upsert(self, document: Any, **kwargs: Any) -> Any: ...


class EmbeddingHandler(Protocol):
    """Protocol for embedding handler."""

    def __call__(self, *, normalized_document: Any, **kwargs: Any) -> Any: ...


class GuardrailEnforcer(Protocol):
    """Protocol for guardrail enforcement."""

    def __call__(self, *, normalized_document: Any, **kwargs: Any) -> Any: ...


class DeltaDecider(Protocol):
    """Protocol for delta decision."""

    def __call__(
        self, *, normalized_document: Any, baseline: Any, **kwargs: Any
    ) -> Any: ...


# --------------------------------------------------------------------- State
class UploadIngestionState(TypedDict):
    """State for upload ingestion graph."""

    # Input (Required)
    normalized_document_input: dict[str, Any]

    # Input (Optional)
    run_until: NotRequired[str]  # "persist_complete" | "full" | etc.

    # Runtime Context (IDs + Injected Dependencies)
    # REQUIRED: tenant_id, workflow_id enforced (AGENTS.md Root Law)
    # May include: runtime_repository, runtime_storage, runtime_embedder
    context: dict[str, Any]

    # Intermediate State
    document: NotRequired[Any]  # NormalizedDocument (validated)
    config: NotRequired[Any]  # DocumentPipelineConfig
    processing_context: NotRequired[Any]  # DocumentProcessingContext
    processing_result: NotRequired[DocumentProcessingState]

    # Output
    decision: NotRequired[
        str
    ]  # "completed" | "skip_guardrail" | "skip_duplicate" | "error"
    reason: NotRequired[str]
    severity: NotRequired[str]
    document_id: NotRequired[str]
    version: NotRequired[str]
    telemetry: NotRequired[dict[str, Any]]
    transitions: NotRequired[dict[str, Any]]

    # Error
    error: NotRequired[str]


# --------------------------------------------------------------------- Nodes
@observe_span(name="upload.validate_input")
def validate_input_node(state: UploadIngestionState) -> dict[str, Any]:
    """Validate and hydrate normalized_document_input."""
    normalized_input = state.get("normalized_document_input")
    if not normalized_input:
        return {"error": "input_missing:normalized_document_input"}

    # Enforce tenant context (Finding #7 fix)
    runtime_context = state.get("context", {})
    if not runtime_context.get("tenant_id"):
        return {"error": "tenant_id required in context (AGENTS.md Root Law)"}

    try:
        doc = NormalizedDocument.model_validate(normalized_input)
    except Exception as exc:
        return {"error": f"input_invalid:{exc}"}

    return {"document": doc, "error": None}


@observe_span(name="upload.build_config")
def build_config_node(state: UploadIngestionState) -> dict[str, Any]:
    """Build pipeline config and processing context."""
    doc = state.get("document")
    if not doc:
        return {"error": "document_missing"}

    # Build config (upload-specific settings)
    max_bytes = int(getattr(settings, "UPLOAD_MAX_BYTES", 25 * 1024 * 1024))
    allowed_mimes = tuple(
        getattr(
            settings,
            "UPLOAD_ALLOWED_MIME_TYPES",
            (
                "text/plain",
                "text/markdown",
                "text/html",
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        )
    )

    config = DocumentPipelineConfig(
        enable_upload_validation=True,
        max_bytes=max_bytes,
        mime_allowlist=allowed_mimes,
        enable_asset_captions=False,  # Disabled for upload
        enable_embedding=True,
    )

    # Build processing context
    # Resolve trace_id/case_id from context dict (unified with ExternalKnowledgeGraph)
    runtime_context = state.get("context", {})
    proc_context = DocumentProcessingContext.from_document(
        doc,
        case_id=runtime_context.get("case_id"),
        trace_id=runtime_context.get("trace_id"),
    )

    return {
        "config": config,
        "processing_context": proc_context,
        "error": None,
    }


@observe_span(name="upload.run_processing")
def run_processing_node(state: UploadIngestionState) -> dict[str, Any]:
    """Invoke inner document processing graph."""
    doc = state.get("document")
    config = state.get("config")
    proc_context = state.get("processing_context")
    runtime_context = state.get("context", {})

    if not all([doc, config, proc_context]):
        return {"error": "missing_required_state"}

    # Extract runtime dependencies from context
    repository = runtime_context.get("runtime_repository")
    storage = runtime_context.get("runtime_storage")

    # Defaults handled by ai_core_api or defaults if None
    embedder = runtime_context.get("runtime_embedder") or ai_core_api.trigger_embedding
    delta_decider = (
        runtime_context.get("runtime_delta_decider") or ai_core_api.decide_delta
    )
    guardrail_enforcer = (
        runtime_context.get("runtime_guardrail_enforcer")
        or ai_core_api.enforce_guardrails
    )
    quarantine_scanner = runtime_context.get("runtime_quarantine_scanner")  # Optional

    # Build inner graph components
    from documents.parsers import create_default_parser_dispatcher
    from documents.cli import SimpleDocumentChunker

    components = require_document_components()
    parser = create_default_parser_dispatcher()
    chunker = SimpleDocumentChunker()

    # Captioner
    captioner_cls = components.captioner
    try:
        captioner = captioner_cls()
    except Exception:
        captioner = captioner_cls

    # Storage fallback
    if not storage:
        try:
            storage = components.storage()
        except Exception:
            from documents.storage import ObjectStoreStorage

            storage = ObjectStoreStorage()

    inner_graph = build_document_processing_graph(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=captioner,
        chunker=chunker,
        embedder=embedder,
        delta_decider=delta_decider,
        guardrail_enforcer=guardrail_enforcer,
        quarantine_scanner=quarantine_scanner,
    )

    # Determine run_until
    run_until_str = state.get("run_until")
    # Handle None or empty string gracefully if needed, though DocumentProcessingPhase.coerce might handle it
    run_until = DocumentProcessingPhase.coerce(run_until_str) if run_until_str else None

    # Build inner state
    inner_state = DocumentProcessingState(
        document=doc,
        config=config,
        context=proc_context,
        storage=storage,
        run_until=run_until,
    )

    # Invoke inner graph
    try:
        result_state = inner_graph.invoke(inner_state)
        if isinstance(result_state, dict):
            result_state = DocumentProcessingState(**result_state)
    except Exception as exc:
        logger.exception("Document processing graph failed")
        return {"error": f"processing_failed:{exc}"}

    return {
        "processing_result": result_state,
        "error": None,
    }


@observe_span(name="upload.map_results")
def map_results_node(state: UploadIngestionState) -> dict[str, Any]:
    """Map processing results to output transitions."""
    result_state = state.get("processing_result")

    # If we got here due to an error in previous steps (validation/config),
    # processing_result might be missing, but we still need to map the error.
    if not result_state:
        # Check if we have an error from previous steps
        error = state.get("error")
        if error:
            return {
                "decision": "error",
                "reason": error,
                "severity": "error",
                "transitions": {},
                "telemetry": {},
                "error": error,
            }

        return {
            "decision": "error",
            "reason": "processing_result_missing",
            "severity": "error",
            "transitions": {},
            "telemetry": {},
            "error": "processing_result_missing",
        }

    # Extract data
    doc = result_state.document
    normalized_doc = getattr(doc, "document", doc)
    ref = getattr(normalized_doc, "ref", None)
    document_id = getattr(ref, "document_id", None)
    version = getattr(ref, "version", None)

    delta = result_state.delta_decision
    guardrail = result_state.guardrail_decision

    # Fetch configuration for accept_upload context from state logic if possible,
    # but here we might prefer to just access what we have.
    # Re-deriving accept context logic similar to legacy:
    blob = getattr(normalized_doc, "blob", None)
    mime_type = getattr(blob, "media_type", None)

    config = state.get("config")
    max_bytes = getattr(config, "max_bytes", None) if config else None

    transitions: dict[str, Any] = {}

    def _transition(
        *,
        phase: str,
        decision: str,
        reason: str,
        severity: str = "info",
        context: dict[str, Any] | None = None,
        pipeline: PipelineSection | None = None,
        delta: Any = None,
        guardrail: Any = None,
    ) -> dict[str, Any]:
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
                build_guardrail_section(guardrail) if guardrail is not None else None
            ),
        )
        return result.model_dump()

    # accept_upload transition
    transitions["accept_upload"] = _transition(
        phase="accept_upload",
        decision="accepted",
        reason="upload_validated",
        context={
            "mime": mime_type,
            "size_bytes": getattr(blob, "size", None),
            "max_bytes": max_bytes,
        },
    )

    # delta_and_guardrails transition
    if guardrail and not getattr(guardrail, "allowed", False):
        dg_decision = guardrail.decision
        dg_reason = guardrail.reason
        dg_severity = "error"
    elif delta:
        dg_decision = delta.decision
        dg_reason = delta.reason
        dg_severity = "info"
    else:
        dg_decision = "unknown"
        dg_reason = "guardrail_delta_missing"
        dg_severity = "info"

    transitions["delta_and_guardrails"] = _transition(
        phase="delta_and_guardrails",
        decision=dg_decision,
        reason=dg_reason,
        severity=dg_severity,
        context={"document_id": str(document_id) if document_id else None},
        delta=delta,
        guardrail=guardrail,
    )

    # document_pipeline transition
    pipeline_section = PipelineSection(
        phase=result_state.phase,
        run_until=state.get("run_until"),
        error=repr(result_state.error) if result_state.error else None,
    )
    transitions["document_pipeline"] = _transition(
        phase="document_pipeline",
        decision="processed" if result_state.error is None else "error",
        reason=(
            "document_pipeline_completed"
            if result_state.error is None
            else "document_pipeline_failed"
        ),
        severity="error" if result_state.error else "info",
        context={"phase": result_state.phase},
        pipeline=pipeline_section,
    )

    # Determine final decision
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

    if result_state.error is not None:
        decision = "error"
        reason = "document_pipeline_failed"
        severity = "error"

    telemetry = {
        "phase": result_state.phase,
        "run_until": state.get("run_until"),
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
        "error": result_state.error,
    }


# --------------------------------------------------------------------- Graph Definition

workflow = StateGraph(UploadIngestionState)

workflow.add_node("validate_input", validate_input_node)
workflow.add_node("build_config", build_config_node)
workflow.add_node("run_processing", run_processing_node)
workflow.add_node("map_results", map_results_node)

# Edges
workflow.add_edge(START, "validate_input")


# Conditional: Skip processing if validation failed
def _check_validation_error(
    state: UploadIngestionState,
) -> Literal["continue", "error"]:
    return "error" if state.get("error") else "continue"


workflow.add_conditional_edges(
    "validate_input",
    _check_validation_error,
    {"continue": "build_config", "error": "map_results"},
)

workflow.add_edge("build_config", "run_processing")
workflow.add_edge("run_processing", "map_results")
workflow.add_edge("map_results", END)


def build_upload_graph() -> Any:  # CompiledGraph type from langgraph
    """Build and compile the upload ingestion graph.

    Returns a fresh compiled graph instance to prevent state leakage
    across concurrent Celery workers (Finding #2 fix).
    """
    workflow = StateGraph(UploadIngestionState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("build_config", build_config_node)
    workflow.add_node("run_processing", run_processing_node)
    workflow.add_node("map_results", map_results_node)

    workflow.add_edge(START, "validate_input")
    workflow.add_conditional_edges(
        "validate_input",
        _check_validation_error,
        {"continue": "build_config", "error": "map_results"},
    )
    workflow.add_edge("build_config", "run_processing")
    workflow.add_edge("run_processing", "map_results")
    workflow.add_edge("map_results", END)

    return workflow.compile()


class UploadIngestionError(RuntimeError):
    """Raised for unexpected internal errors in the upload ingestion graph."""


__all__ = [
    "UploadIngestionState",
    "UploadIngestionError",
    "build_upload_graph",  # Factory function instead of singleton
    "validate_input_node",
    "build_config_node",
    "run_processing_node",
    "map_results_node",
]
