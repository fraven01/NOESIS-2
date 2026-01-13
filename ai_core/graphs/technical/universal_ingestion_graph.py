"""Universal Technical Graph for Ingestion (Pure Ingestion Primitive).

This graph accepts a NormalizedDocument and persists it to the repository,
then optionally triggers RAG processing (chunking/embedding).

Contracts:
- Input: UniversalIngestionInput (strictly normalized_document)
- Output: UniversalIngestionOutput (decision, reason_code)
- Context: ToolContext required (provides scope/business IDs)
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, ValidationError

from ai_core.graph.io import GraphIOSpec, GraphIOVersion
from ai_core.infra.observability import observe_span
from documents.contracts import NormalizedDocument
from documents.pipeline import DocumentProcessingContext, DocumentPipelineConfig
from documents.processing_graph import build_document_processing_graph
from ai_core.services import _get_documents_repository
from ai_core.services.document_processing_factory import (
    build_document_processing_bundle,
)
from ai_core.tool_contracts import ToolContext

# --------------------------------------------------------------------- I/O Contracts

UNIVERSAL_INGESTION_SCHEMA_ID = "noesis.graphs.universal_ingestion"
UNIVERSAL_INGESTION_IO_VERSION = GraphIOVersion(major=1, minor=0, patch=0)
UNIVERSAL_INGESTION_IO_VERSION_STRING = UNIVERSAL_INGESTION_IO_VERSION.as_string()


class UniversalIngestionInput(TypedDict):
    """Input payload for the universal ingestion graph.

    Strictly expects a NormalizedDocument (dict or object).
    Legacy fields (search_query, upload_blob, collection_id) are REMOVED.
    """

    normalized_document: dict[str, Any] | NormalizedDocument


class UniversalIngestionOutput(TypedDict):
    """Output contract for the universal ingestion graph."""

    decision: Literal["processed", "skipped", "failed"]
    reason_code: (
        Literal[
            "DUPLICATE", "VALIDATION_ERROR", "PERSISTENCE_ERROR", "PROCESSING_ERROR"
        ]
        | None
    )
    reason: str | None

    document_id: str | None
    ingestion_run_id: str | None
    telemetry: dict[str, Any]

    # Legacy compatibility fields (can be deprecated later)
    formatted_status: str | None


class UniversalIngestionInputModel(BaseModel):
    """Pydantic input contract for the universal ingestion graph."""

    normalized_document: dict[str, Any] | NormalizedDocument | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class UniversalIngestionGraphInput(BaseModel):
    """Boundary input model for the universal ingestion graph."""

    schema_id: Literal[UNIVERSAL_INGESTION_SCHEMA_ID] = UNIVERSAL_INGESTION_SCHEMA_ID
    schema_version: Literal[UNIVERSAL_INGESTION_IO_VERSION_STRING] = (
        UNIVERSAL_INGESTION_IO_VERSION_STRING
    )
    input: UniversalIngestionInputModel
    context: dict[str, Any]

    model_config = ConfigDict(frozen=True, extra="forbid")


class UniversalIngestionGraphOutput(BaseModel):
    """Boundary output model for the universal ingestion graph."""

    schema_id: Literal[UNIVERSAL_INGESTION_SCHEMA_ID] = UNIVERSAL_INGESTION_SCHEMA_ID
    schema_version: Literal[UNIVERSAL_INGESTION_IO_VERSION_STRING] = (
        UNIVERSAL_INGESTION_IO_VERSION_STRING
    )
    decision: Literal["processed", "skipped", "failed"]
    reason_code: (
        Literal[
            "DUPLICATE", "VALIDATION_ERROR", "PERSISTENCE_ERROR", "PROCESSING_ERROR"
        ]
        | None
    )
    reason: str | None
    document_id: str | None
    ingestion_run_id: str | None
    telemetry: dict[str, Any]
    formatted_status: str | None

    model_config = ConfigDict(frozen=True, extra="forbid")


UNIVERSAL_INGESTION_IO = GraphIOSpec(
    schema_id=UNIVERSAL_INGESTION_SCHEMA_ID,
    version=UNIVERSAL_INGESTION_IO_VERSION,
    input_model=UniversalIngestionGraphInput,
    output_model=UniversalIngestionGraphOutput,
)


class UniversalIngestionState(TypedDict):
    """Runtime state for the universal ingestion graph."""

    # Input
    input: UniversalIngestionInput
    context: dict[str, Any]  # Raw context dict from invocation
    tool_context: ToolContext | None  # Validated ToolContext

    # Artifacts
    normalized_document: NormalizedDocument | None
    dedup_status: Literal["new", "duplicate"] | None
    existing_document_ref: Any | None  # DocumentRef if duplicate found

    # Results
    ingestion_result: dict[str, Any] | None
    processing_result: dict[str, Any] | None

    # Internal Error
    error: str | None
    output: UniversalIngestionOutput | None


logger = logging.getLogger(__name__)


class UniversalIngestionError(Exception):
    """Error raised during universal ingestion graph execution."""

    pass


# --------------------------------------------------------------------- Nodes


@observe_span(name="node.validate_input")
def validate_input_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Validate input contract and context."""
    try:
        graph_input = UniversalIngestionGraphInput.model_validate(state)
    except ValidationError as exc:
        msg = f"Invalid graph input: {exc.errors()}"
        logger.error(msg)
        return {"error": msg, "tool_context": None}

    inp = graph_input.input
    context_dict = graph_input.context

    # 1. Validate ToolContext (Single Source of Truth)
    try:
        tool_context = ToolContext.model_validate(context_dict)
    except ValidationError as exc:
        msg = f"Invalid context structure: {exc.errors()}"
        logger.error(msg)
        return {"error": msg, "tool_context": None}

    # 2. Validate Input (NormalizedDocument)
    raw_doc = inp.normalized_document
    if not raw_doc:
        return {
            "error": "Missing normalized_document in input",
            "tool_context": tool_context,
        }

    # 3. Normalize to Object
    try:
        if isinstance(raw_doc, dict):
            norm_doc = NormalizedDocument.model_validate(raw_doc)
        else:
            norm_doc = raw_doc  # Assume Object
    except ValidationError as exc:
        msg = f"Invalid normalized_document: {exc.errors()}"
        logger.error(msg)
        return {"error": msg, "tool_context": tool_context}

    # 4. Strict Collection ID Check
    # Must use context.business.collection_id
    if not tool_context.business.collection_id:
        return {
            "error": "Missing collection_id in BusinessContext",
            "tool_context": tool_context,
        }

    return {
        "tool_context": tool_context,
        "normalized_document": norm_doc,
        "error": None,
    }


@observe_span(name="node.dedup")
def dedup_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Check deduplication status (Layer 1: Document Level)."""
    if state.get("error"):
        return {}

    # MVP: Always mark as 'new'.
    # P2: Check SHA256 against repository.
    # Future:
    # norm_doc = state["normalized_document"]
    # repo = state["tool_context"].metadata.get("runtime_repository")
    # existing = repo.find_by_hash(norm_doc.checksum) -> dedup_status="duplicate"

    return {"dedup_status": "new", "existing_document_ref": None}


@observe_span(name="node.persist")
def persist_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Persist document to repository (Upsert)."""
    if state.get("error"):
        return {}

    dedup_status = state.get("dedup_status")
    if dedup_status == "duplicate":
        # Skip persistence for duplicates
        return {}

    norm_doc = state.get("normalized_document")
    tool_context = state.get("tool_context")
    if not norm_doc or not tool_context:
        return {"error": "State corrupted before persist"}

    # Resolve Repository (Service Dependency)
    repo = (
        tool_context.metadata.get("runtime_repository") or _get_documents_repository()
    )

    raw_version_id = tool_context.business.document_version_id
    resolved_version_id: UUID | None = None
    if raw_version_id:
        try:
            resolved_version_id = (
                raw_version_id
                if isinstance(raw_version_id, UUID)
                else UUID(str(raw_version_id))
            )
        except (TypeError, ValueError, AttributeError):
            resolved_version_id = None
    if resolved_version_id is None:
        resolved_version_id = uuid4()

    norm_doc = norm_doc.model_copy(
        update={
            "ref": norm_doc.ref.model_copy(
                update={"document_version_id": resolved_version_id},
                deep=True,
            )
        },
        deep=True,
    )

    try:
        # Business ID Propagation happens via Scope/Business contexts
        # The Repository service handles the actual DB writing.
        audit_meta = tool_context.metadata.get("audit_meta")

        saved_doc = repo.upsert(
            norm_doc, scope=tool_context.scope, audit_meta=audit_meta
        )

        # Extract ID
        doc_id = (
            str(saved_doc.ref.document_id)
            if hasattr(saved_doc, "ref")
            else str(saved_doc.id)
        )

        return {
            "ingestion_result": {"status": "persisted", "id": doc_id},
            "normalized_document": saved_doc,
        }

    except Exception as exc:
        logger.exception("Persistence failed")
        return {"error": f"Persistence failed: {exc}"}


@observe_span(name="node.process")
def process_node(
    state: UniversalIngestionState, config: RunnableConfig
) -> dict[str, Any]:
    """Run RAG processing (Chunking/Embedding)."""
    if state.get("error"):
        return {}

    dedup_status = state.get("dedup_status")
    if dedup_status == "duplicate":
        return {}

    norm_doc = state.get("normalized_document")
    tool_context = state.get("tool_context")
    if not norm_doc or not tool_context:
        return {"error": "State corrupted before process"}

    # Check if embedding enabled (Store-Only support)
    pipeline_config_dict = norm_doc.meta.pipeline_config or {}
    enable_embedding = pipeline_config_dict.get("enable_embedding", True)

    if not enable_embedding:
        return {"processing_result": {"status": "skipped_store_only"}}

    try:
        # 1. Build DocumentProcessingContext
        proc_ctx = DocumentProcessingContext.from_document(
            norm_doc,
            case_id=tool_context.business.case_id,
            trace_id=tool_context.scope.trace_id,
            span_id=tool_context.metadata.get("span_id"),
        )

        # 2. Build Pipeline Config
        valid_fields = set(DocumentPipelineConfig.__dataclass_fields__.keys())
        filtered_conf = {
            k: v for k, v in pipeline_config_dict.items() if k in valid_fields
        }
        pipeline_config = DocumentPipelineConfig(**filtered_conf)

        # 3. Resolve Dependencies
        from ai_core.api import trigger_embedding

        repository = _get_documents_repository()

        # 4. Invoke Processing Graph
        # Inject all required dependencies into the factory
        processing_workflow, dependencies = build_document_processing_bundle(
            repository=repository,
            embedder=trigger_embedding,  # Fixed: Pass embedder for RAG indexing
            build_graph=build_document_processing_graph,
        )
        storage = dependencies.storage

        sub_input = {
            "document": norm_doc,
            "context": proc_ctx,
            "config": pipeline_config,
            "storage": storage,
        }

        # Pass through configurable if needed
        result = processing_workflow.invoke(sub_input, config=config)
        return {"processing_result": result}

    except Exception as exc:
        logger.exception("Processing failed")
        return {"error": f"Processing failed: {exc}"}


@observe_span(name="node.finalize")
def finalize_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Build final Unified Output."""
    error = state.get("error")
    tool_context = state.get("tool_context")
    dedup_status = state.get("dedup_status")
    norm_doc = state.get("normalized_document")

    # ID Resolution
    doc_id = None
    if dedup_status == "duplicate":
        existing = state.get("existing_document_ref")
        if existing:
            # Handle both dict and object
            if hasattr(existing, "document_id"):
                doc_id = str(existing.document_id)
            elif isinstance(existing, dict):
                doc_id = existing.get("document_id")
    elif norm_doc:
        doc_id = str(norm_doc.ref.document_id)

    # Telemetry
    telemetry = {}
    ingestion_run_id = None
    if tool_context:
        telemetry = {
            "trace_id": tool_context.scope.trace_id,
            "tenant_id": tool_context.scope.tenant_id,
        }
        ingestion_run_id = tool_context.scope.ingestion_run_id

    # Decision Logic
    if error:
        decision = "failed"
        # P2 Fix: Classify reason_code based on error content
        if any(x in error for x in ("Missing", "Invalid", "context", "collection_id")):
            reason_code = "VALIDATION_ERROR"
        elif "Persistence" in error or "persist" in error.lower():
            reason_code = "PERSISTENCE_ERROR"
        else:
            reason_code = "PROCESSING_ERROR"
        reason = error
    elif dedup_status == "duplicate":
        decision = "skipped"
        reason_code = "DUPLICATE"
        reason = "Document is a duplicate."
    else:
        # Check explicit Step failures
        # e.g. ingest_res might be None if Persist failed but didn't throw (unlikely with this code)
        decision = "processed"
        reason_code = None
        reason = "Ingestion successful."

    output = UniversalIngestionGraphOutput(
        decision=decision,
        reason_code=reason_code,
        reason=reason,
        document_id=doc_id,
        ingestion_run_id=ingestion_run_id,
        telemetry=telemetry,
        formatted_status=decision.upper(),  # Legacy compat
    )
    return {"output": output.model_dump(mode="json")}


# --------------------------------------------------------------------- Graph


def build_universal_ingestion_graph() -> StateGraph:
    """Build the Universal Ingestion Graph (Pure Primitive)."""
    workflow = StateGraph(UniversalIngestionState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("dedup", dedup_node)
    workflow.add_node("persist", persist_node)
    workflow.add_node("process", process_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "validate_input")

    def check_valid(state: UniversalIngestionState) -> str:
        if state.get("error"):
            return "finalize"
        return "dedup"

    workflow.add_conditional_edges("validate_input", check_valid)

    def check_dedup(state: UniversalIngestionState) -> str:
        if state.get("error"):
            return "finalize"
        if state.get("dedup_status") == "duplicate":
            return "finalize"
        return "persist"

    workflow.add_conditional_edges("dedup", check_dedup)

    def check_persist(state: UniversalIngestionState) -> str:
        if state.get("error"):
            return "finalize"
        return "process"

    workflow.add_conditional_edges("persist", check_persist)
    workflow.add_edge("process", "finalize")
    workflow.add_edge("finalize", END)

    graph = workflow.compile()
    setattr(graph, "io_spec", UNIVERSAL_INGESTION_IO)
    return graph
