"""Universal Technical Graph for Ingestion (Upload + Crawler)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ai_core.infra.observability import observe_span
from documents.contracts import (
    NormalizedDocument,
    DocumentRef,
    DocumentMeta,
)
from documents.pipeline import DocumentProcessingContext, DocumentPipelineConfig
from documents.processing_graph import build_document_processing_graph
from ai_core.services import _get_documents_repository
from ai_core.contracts.scope import ScopeContext


class UniversalIngestionError(Exception):
    """Error raised during universal ingestion graph execution."""

    pass


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- State
class UniversalIngestionInput(TypedDict):
    """Input payload for the universal ingestion graph."""

    source: Literal["upload", "crawler"]
    mode: Literal["ingest_only"]
    collection_id: str

    # Payload for 'upload' source
    upload_blob: dict[str, Any] | None
    metadata_obj: dict[str, Any] | None

    # Payload for 'crawler' source
    # In Phase 2, we assume we receive a pre-built NormalizedDocument or equivalent payload for crawler
    # For simplification, we'll support a direct 'normalized_document' input option
    # if the caller has already normalized (e.g. crawler builder)
    normalized_document: dict[str, Any] | None


class UniversalIngestionOutput(TypedDict):
    """Output contract for the universal ingestion graph."""

    decision: Literal["ingested", "skipped", "error"]
    reason: str | None
    telemetry: dict[str, Any]
    ingestion_run_id: str | None
    document_id: str | None

    # Roadmap Phase 2 requirements
    transitions: list[str]
    hitl_required: bool
    hitl_reasons: list[str]
    review_payload: dict[str, Any] | None
    normalized_document_ref: dict[str, Any] | None


class UniversalIngestionState(TypedDict):
    """Runtime state for the universal ingestion graph."""

    # Input (Immutable during run ideally)
    input: UniversalIngestionInput
    context: dict[str, Any]  # IDs: tenant_id, trace_id, case_id

    # Pipeline Artifacts
    normalized_document: NormalizedDocument | None

    # Outcomes
    processing_result: dict[str, Any] | None
    ingestion_result: dict[str, Any] | None

    # Final Output
    output: UniversalIngestionOutput | None
    error: str | None


# --------------------------------------------------------------------- Helpers


def _build_normalized_document(
    input_data: UniversalIngestionInput,
    context: dict[str, Any],
) -> NormalizedDocument:
    """Build a NormalizedDocument from upload input."""

    collection_id = input_data.get("collection_id")
    if not collection_id:
        raise ValueError("collection_id_missing")

    # 1. Resolve IDs
    tenant_id = context.get("tenant_id")
    workflow_id = (
        context.get("workflow_id") or context.get("case_id") or "universal_ingestion"
    )

    if not tenant_id:
        raise ValueError("tenant_id_missing_in_context")

    # Generate document ID if not provided in meta (usually generated here)
    document_id = uuid4()

    # 2. Build Document Ref
    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        collection_id=collection_id,
    )

    # 3. Build Document Meta
    meta_obj = input_data.get("metadata_obj") or {}

    # Map common upload metadata to DocumentMeta fields
    if "title" not in meta_obj and "file_name" in meta_obj:
        meta_obj["title"] = meta_obj["file_name"]

    # Filter only valid fields for DocumentMeta to avoid extra="forbid" error
    valid_meta_fields = set(DocumentMeta.model_fields.keys())
    filtered_meta = {
        k: v
        for k, v in meta_obj.items()
        if k in valid_meta_fields
        and k not in ("tenant_id", "workflow_id", "document_collection_id")
    }

    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_collection_id=collection_id,
        **filtered_meta,
    )

    # 4. Build Blob Locator
    blob_data = input_data.get("upload_blob")
    if not blob_data:
        raise ValueError("upload_blob_missing")

    # Pydantic will discriminate based on 'type' field in blob_data
    # TypedAdapter could be used, or just let NormalizedDocument validate it

    # 5. Construct NormalizedDocument
    # Note: checksum and created_at are required.
    # If checksum not in blob_data, we might fail if strict checks are on.
    # For Inline/LocalFile, standard logic usually calculates it.
    # Here we assume blob_data is fully populated (e.g. by UploadManager/Service)

    blob_checksum = blob_data.get("sha256")
    if not blob_checksum:
        raise ValueError("blob_sha256_missing")

    return NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob_data,  # Pydantic coercion to BlobLocator
        checksum=blob_checksum,
        created_at=datetime.now(timezone.utc),
        source="upload",
    )


# --------------------------------------------------------------------- Nodes


@observe_span(name="node.validate_input")
def validate_input_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Validate input source, mode, and context presence."""
    inp = state.get("input", {})
    context = state.get("context", {})

    # 1. Validate Context IDs (Strict per existing contracts)
    required_ids = ["tenant_id", "trace_id", "case_id"]
    for rid in required_ids:
        if not context.get(rid):
            error_msg = f"Missing required context context: {rid}"
            logger.error(error_msg)
            return {"error": error_msg}

    # 2. Validate Mode
    mode = inp.get("mode")
    if mode != "ingest_only":
        error_msg = f"Unsupported mode for Phase 2: {mode}"
        logger.error(error_msg)
        return {"error": error_msg}

    # 3. Validate Source specific payloads
    source = inp.get("source")
    if source == "upload":
        # Allow normalized_document to supersede upload_blob
        if not inp.get("upload_blob") and not inp.get("normalized_document"):
            return {
                "error": "Missing upload_blob or normalized_document for source=upload"
            }
        if not inp.get("collection_id"):
            return {"error": "Missing collection_id for source=upload"}

    elif source == "crawler":
        if not inp.get("normalized_document"):
            return {"error": "Missing normalized_document for source=crawler"}
    else:
        return {"error": f"Unsupported source: {source}"}

    return {"error": None}


@observe_span(name="node.normalize")
def normalize_document_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Normalize input into a NormalizedDocument object."""
    inp = state.get("input", {})
    source = inp.get("source")
    context = state.get("context", {})

    try:
        norm_doc = None
        # Check explicit normalized input first (preferred for all sources)
        raw_doc = inp.get("normalized_document")
        if raw_doc:
            if isinstance(raw_doc, dict):
                norm_doc = NormalizedDocument.model_validate(raw_doc)
            else:
                norm_doc = raw_doc

        # Fallback to upload_blob for uploads if no normalized doc provided
        if not norm_doc and source == "upload":
            norm_doc = _build_normalized_document(inp, context)

        if not norm_doc:
            return {"error": "Could not verify normalized document"}

        return {"normalized_document": norm_doc}

    except ValidationError as ve:
        logger.error(f"Normalization failed: {ve}")
        return {"error": f"Normalization failed: {ve}"}
    except Exception as exc:
        logger.exception("Unexpected normalization error")
        return {"error": str(exc)}


@observe_span(name="node.persist")
def persist_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Persist the normalized document (upsert)."""
    norm_doc = state.get("normalized_document")
    if not norm_doc:
        return {"error": "No normalized document to persist"}

    context = state.get("context", {})
    service = _get_documents_repository()

    try:
        # Build ScopeContext for traceability
        # Ensure we have required fields (invocation_id might be missing in some paths)
        scope = ScopeContext(
            tenant_id=context["tenant_id"],
            trace_id=context["trace_id"],
            # Fallback to trace_id if invocation_id not explicit
            invocation_id=context.get("invocation_id") or context["trace_id"],
            case_id=context.get("case_id"),
            workflow_id=context.get("workflow_id"),
            ingestion_run_id=context.get("ingestion_run_id"),
            collection_id=state.get("input", {}).get("collection_id"),
        )

        # Upsert with scope to preserve lineage
        saved_doc = service.upsert(norm_doc, scope=scope)

        # If successfully saved, we return the persisted ID
        doc_id = (
            str(saved_doc.ref.document_id)
            if hasattr(saved_doc, "ref")
            else str(saved_doc.id)
        )

        return {"ingestion_result": {"status": "persisted", "id": doc_id}}

    except Exception as exc:
        logger.exception("Persistence failed")
        return {"error": f"Persistence failed: {exc}"}


@observe_span(name="node.process")
def process_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Run document processing (embedding, etc)."""
    norm_doc = state.get("normalized_document")
    context_data = state.get("context", {})

    if not norm_doc:
        return {"error": "No normalized document to process"}

    # We reuse the existing document processing graph
    # This graph usually takes { "document": ... } or similar state

    # We need to adapt the state for the sub-graph
    # Create DocumentProcessingContext from the document and environment
    processing_context = DocumentProcessingContext.from_document(
        norm_doc,
        case_id=context_data.get("case_id"),
        trace_id=context_data.get("trace_id"),
        span_id=context_data.get("span_id"),
    )

    # DocumentProcessingState requires document, config, and context
    # config must be DocumentPipelineConfig object, not dict
    raw_config = norm_doc.meta.pipeline_config or {}
    pipeline_config = DocumentPipelineConfig(**raw_config)

    # Get storage component for the state
    from ai_core.services import get_document_components

    storage = get_document_components().storage()

    sub_graph_input = {
        "document": norm_doc,
        "context": processing_context,
        "config": pipeline_config,
        "storage": storage,
    }

    try:
        # Singleton cached graph
        processing_workflow = _get_cached_processing_graph()

        # We need to construct the config/meta for the sub-graph
        config = {
            "configurable": {
                # pass dependencies if needed
            }
        }

        result = processing_workflow.invoke(sub_graph_input, config=config)

        return {"processing_result": result}

    except Exception as exc:
        logger.exception("Processing failed")
        return {"error": f"Processing failed: {exc}"}


@observe_span(name="node.finalize")
def finalize_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Map results to final output."""
    error = state.get("error")
    context = state.get("context", {})

    telemetry = {
        "trace_id": context.get("trace_id"),
        "tenant_id": context.get("tenant_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if error:
        return {
            "output": {
                "decision": "error",
                "reason": error,
                "telemetry": telemetry,
                "ingestion_run_id": None,
                "document_id": None,
                "hitl_required": False,
                "hitl_reasons": [],
                "transitions": [],
                "review_payload": None,
                "normalized_document_ref": None,
            }
        }

    # If success
    # Extract document_id from normalized_document or processing result
    norm_doc = state.get("normalized_document")
    doc_id = str(norm_doc.ref.document_id) if norm_doc else None

    # Construct transitions (simplified for Phase 2)
    # Ideally we'd extract this from processing_result history if available
    transitions = ["validate_input", "normalize", "persist", "process", "finalize"]

    # Construct review_payload for HITL signaling
    review_payload = norm_doc.model_dump() if norm_doc else None

    return {
        "output": {
            "decision": "ingested",
            "reason": "Success",
            "telemetry": telemetry,
            "ingestion_run_id": context.get("ingestion_run_id"),
            "document_id": doc_id,
            "hitl_required": False,
            "hitl_reasons": [],
            "transitions": transitions,
            "review_payload": review_payload,
            "normalized_document_ref": norm_doc.ref.model_dump() if norm_doc else None,
        }
    }


# --------------------------------------------------------------------- Graph Definition

_CACHED_PROCESSING_GRAPH = None


def _get_cached_processing_graph() -> Any:
    """Return the cached, compiled document processing graph singleton.

    This avoids rebuilding the huge LangGraph structure on every request,
    preventing OOM and initialization overhead.
    """
    global _CACHED_PROCESSING_GRAPH
    if _CACHED_PROCESSING_GRAPH is not None:
        return _CACHED_PROCESSING_GRAPH

    # Lazy imports to avoid circular dependencies
    from ai_core.services import _get_documents_repository, get_document_components
    from documents.parsers import create_default_parser_dispatcher
    from documents.cli import SimpleDocumentChunker
    from ai_core.api import (
        trigger_embedding,
        decide_delta,
        enforce_guardrails,
    )

    # DEPENDENCY INJECTION ROOT
    # We construct the shared, stateless components for the graph here.

    # 1. Repository & Storage
    repository = _get_documents_repository()
    components = get_document_components()
    storage = components.storage()  # storage class to instance
    captioner = components.captioner()  # captioner class to instance

    # 2. Key Processing Logic
    parser_dispatcher = create_default_parser_dispatcher()
    chunker = SimpleDocumentChunker()

    # 3. Build Compiled Graph
    # The graph is stateless regarding the pipeline config (passed in state),
    # but holds references to these service singletons.
    _CACHED_PROCESSING_GRAPH = build_document_processing_graph(
        parser=parser_dispatcher,
        repository=repository,
        storage=storage,
        captioner=captioner,
        chunker=chunker,
        embedder=trigger_embedding,
        delta_decider=decide_delta,
        guardrail_enforcer=enforce_guardrails,
        propagate_errors=True,
    )

    logger.info("Universal ingestion: Document processing graph compiled and cached.")
    return _CACHED_PROCESSING_GRAPH


def build_universal_ingestion_graph() -> Any:
    """Build and compile the universal ingestion graph."""
    workflow = StateGraph(UniversalIngestionState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("normalize", normalize_document_node)
    workflow.add_node("persist", persist_node)

    # Process node will now use the cached sub-graph internally
    workflow.add_node("process", process_node)

    workflow.add_node("finalize", finalize_node)

    # Edges
    workflow.add_edge(START, "validate_input")

    def check_validation(
        state: UniversalIngestionState,
    ) -> Literal["normalize", "finalize"]:
        if state.get("error"):
            return "finalize"
        return "normalize"

    workflow.add_conditional_edges("validate_input", check_validation)

    def check_normalization(
        state: UniversalIngestionState,
    ) -> Literal["persist", "finalize"]:
        if state.get("error"):
            return "finalize"
        return "persist"

    workflow.add_conditional_edges("normalize", check_normalization)

    def check_persistence(
        state: UniversalIngestionState,
    ) -> Literal["process", "finalize"]:
        if state.get("error"):
            return "finalize"
        return "process"

    workflow.add_conditional_edges("persist", check_persistence)

    workflow.add_edge("process", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


__all__ = [
    "build_universal_ingestion_graph",
    "UniversalIngestionState",
    "UniversalIngestionError",
]
