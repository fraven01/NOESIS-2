"""Universal Technical Graph for Ingestion (Upload + Crawler)."""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timezone
from queue import Empty, Queue
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
from ai_core.tools.web_search import (
    SearchProviderError,
    WebSearchResponse,
)


class UniversalIngestionError(Exception):
    """Error raised during universal ingestion graph execution."""

    pass


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- State
class UniversalIngestionInput(TypedDict):
    """Input payload for the universal ingestion graph."""

    source: Literal["upload", "crawler", "search"]
    mode: Literal["ingest_only", "acquire_only", "acquire_and_ingest"]
    collection_id: str

    # Payload for 'upload' source
    upload_blob: dict[str, Any] | None
    metadata_obj: dict[str, Any] | None

    # Payload for 'crawler' source
    # In Phase 2, we assume we receive a pre-built NormalizedDocument or equivalent payload for crawler
    # For simplification, we'll support a direct 'normalized_document' input option
    # if the caller has already normalized (e.g. crawler builder)
    normalized_document: dict[str, Any] | None

    # Payload for 'search' source
    search_query: str | None
    search_config: dict[str, Any] | None
    # Phase 5: Support direct ingestion of specific search results (bypass search worker)
    preselected_results: list[dict[str, Any]] | None


class UniversalIngestionOutput(TypedDict):
    """Output contract for the universal ingestion graph."""

    decision: Literal["ingested", "skipped", "error", "acquired"]
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
    """Runtime state for the universal ingestion graph.

    BREAKING CHANGE (Option A - Full ToolContext Migration):
    context is now a ToolContext-compatible dict with nested structure:
    {"scope": {...}, "business": {...}, "metadata": {...}}
    """

    # Input (Immutable during run ideally)
    input: UniversalIngestionInput
    context: dict[str, Any]  # ToolContext-compatible dict (scope/business/metadata)

    # Pipeline Artifacts
    normalized_document: NormalizedDocument | None

    # Search Artifacts
    search_results: list[dict[str, Any]] | None
    selected_result: dict[str, Any] | None

    # Outcomes
    processing_result: dict[str, Any] | None
    ingestion_result: dict[str, Any] | None

    # Final Output
    output: UniversalIngestionOutput | None
    error: str | None


# --------------------------------------------------------------------- Helpers


def _build_normalized_document(
    input_data: UniversalIngestionInput,
    context,  # ToolContext
) -> NormalizedDocument:
    """Build a NormalizedDocument from upload input.

    BREAKING CHANGE (Option A):
    context is now ToolContext, access via context.scope.X and context.business.X
    """
    collection_id = input_data.get("collection_id")
    if not collection_id:
        raise ValueError("collection_id_missing")

    # 1. Resolve IDs from ToolContext
    tenant_id = context.scope.tenant_id
    workflow_id = (
        context.business.workflow_id
        or context.business.case_id
        or "universal_ingestion"
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
    """Validate input source, mode, and context presence.

    BREAKING CHANGE (Option A):
    context is now ToolContext-compatible dict, validated here.
    """
    from ai_core.tool_contracts import ToolContext
    from pydantic import ValidationError

    inp = state.get("input", {})
    context_dict = state.get("context", {})

    # 1. Validate ToolContext structure
    try:
        ToolContext.model_validate(context_dict)
    except ValidationError as exc:
        error_msg = f"Invalid context structure: {exc.errors()}"
        logger.error(error_msg)
        return {"error": error_msg}

    # 2. Validate required IDs are present (Pydantic already validates non-None tenant_id, trace_id)
    # Additional checks if needed could go here

    # 2. Validate Mode
    mode = inp.get("mode")
    # Phase 4 Update: mode can be acquire_only or acquire_and_ingest for search
    allowed_modi = ("ingest_only", "acquire_only", "acquire_and_ingest")
    if mode not in allowed_modi:
        error_msg = f"Unsupported mode: {mode}"
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

    elif source == "search":
        # Phase 5: Allowed if query OR preselected results provided
        if not inp.get("search_query") and not inp.get("preselected_results"):
            return {
                "error": "Missing search_query or preselected_results for source=search"
            }

    else:
        return {"error": f"Unsupported source: {source}"}

    return {"error": None}


def _run_search_with_timeout(worker, query: str, context, timeout: int):
    """Run search worker with timeout protection using threading.

    Args:
        worker: WebSearchWorker instance
        query: Search query string
        context: ToolContext instance
        timeout: Timeout in seconds
    """
    result_queue: Queue = Queue()

    def worker_thread():
        try:
            response = worker.run(query=query, context=context)
            result_queue.put(("success", response))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=worker_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"Search worker exceeded {timeout}s timeout")

    try:
        status, result = result_queue.get_nowait()
        if status == "error":
            raise result
        return result
    except Empty:
        raise TimeoutError("Search worker produced no result")


def _check_search_rate_limit(tenant_id: str, query: str) -> bool:
    """Check if tenant has exceeded search rate limit.

    Uses Django cache to track search requests per tenant per hour.
    Returns True if request is allowed, False if rate limit exceeded.
    """
    from django.conf import settings
    from django.core.cache import cache

    cache_key = f"search_rate_limit:{tenant_id}"
    count = cache.get(cache_key, 0)

    max_searches_per_hour = getattr(settings, "MAX_SEARCHES_PER_TENANT_PER_HOUR", 100)

    if count >= max_searches_per_hour:
        logger.warning(
            "Search rate limit exceeded",
            extra={
                "tenant_id": tenant_id,
                "query": query[:100],  # Log first 100 chars only
                "count": count,
                "limit": max_searches_per_hour,
            },
        )
        return False

    # Increment counter with 1 hour expiry
    cache.set(cache_key, count + 1, timeout=3600)
    return True


@observe_span(name="node.search")
def search_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Execute web search using the configured worker.

    BREAKING CHANGE (Option A):
    WebSearchWorker.run() now requires ToolContext instead of dict.
    We build ToolContext from the state context dict.
    """
    from ai_core.tool_contracts import ToolContext
    from pydantic import ValidationError

    inp = state.get("input", {})
    query = inp.get("search_query")
    preselected = inp.get("preselected_results")

    # Phase 5: Optimization - if results passed in, use them directly
    if preselected:
        return {"search_results": preselected}

    context_dict = state.get("context", {})

    # BREAKING CHANGE (Option A): Build ToolContext from dict first
    # The context dict should have nested structure: {"scope": {...}, "business": {...}, "metadata": {...}}
    # OR it can have both nested and flattened for backward compatibility
    try:
        tool_context = ToolContext.model_validate(context_dict)
    except ValidationError as exc:
        logger.error("Failed to build ToolContext", extra={"errors": exc.errors()})
        return {"error": "Invalid context structure", "search_results": []}

    # Check rate limit for tenant
    tenant_id = tool_context.scope.tenant_id
    if tenant_id and query:
        if not _check_search_rate_limit(str(tenant_id), query):
            return {
                "error": "Search rate limit exceeded. Please try again later.",
                "search_results": [],
            }

    # Worker is expected in metadata (Phase 4 requirement)
    worker = tool_context.metadata.get("runtime_worker")
    if not worker:
        # Fallback to simple error if no worker injected
        return {"error": "No search worker configured in context"}

    # Get timeout from settings with fallback to 30 seconds
    from django.conf import settings

    search_timeout = getattr(settings, "SEARCH_WORKER_TIMEOUT_SECONDS", 30)

    try:
        response: WebSearchResponse = _run_search_with_timeout(
            worker, query, tool_context, timeout=search_timeout
        )
    except TimeoutError as e:
        logger.warning(str(e), extra={"query": query, "timeout": search_timeout})
        return {"error": str(e), "search_results": []}
    except SearchProviderError as exc:
        logger.warning(f"Search failed: {exc}")
        return {"error": str(exc), "search_results": []}

    if response.outcome.decision == "error":
        error_msg = (
            response.outcome.meta.get("error", {}).get("message")
            or response.outcome.rationale
        )
        return {"error": error_msg, "search_results": []}

    results = [r.model_dump(mode="json") for r in response.results]
    return {"search_results": results}


def _blocked_domain(url: str, blocked_domains: list[str]) -> bool:
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if not hostname or not blocked_domains:
        return False
    for domain in blocked_domains:
        blocked = domain.lower()
        if hostname == blocked or hostname.endswith(f".{blocked}"):
            return True
    return False


@observe_span(name="node.select")
def selection_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Filter and select the best candidate from search results."""
    results = state.get("search_results", [])
    inp = state.get("input", {})
    config = inp.get("search_config") or {}

    min_len = config.get("min_snippet_length", 40)
    blocked = config.get("blocked_domains", [])
    top_n = config.get("top_n", 5)
    prefer_pdf = config.get("prefer_pdf", True)

    # Phase 5 Fix: Allow preselected results (bare URLs) to bypass snippet check
    preselected_urls = {
        item["url"]
        for item in (inp.get("preselected_results") or [])
        if item.get("url")
    }

    validated = []
    for raw in results:
        url = raw.get("url")
        snippet = raw.get("snippet", "")

        # Bypass length check if preselected
        if not url:
            continue

        if url not in preselected_urls and len(snippet) < min_len:
            continue

        if _blocked_domain(url, blocked):
            continue

        lowered = snippet.lower()
        if "noindex" in lowered and "robot" in lowered:
            continue

        validated.append(raw)

    shortlisted = validated[:top_n]
    selected = None

    if shortlisted:
        if prefer_pdf:
            for cand in shortlisted:
                if cand.get("is_pdf"):
                    selected = cand
                    break

        if not selected:
            selected = shortlisted[0]

    return {"selected_result": selected, "search_results": shortlisted}

    return {"error": None}


def _normalize_from_crawler(
    input_data: dict[str, Any],
) -> NormalizedDocument:
    """Build NormalizedDocument from crawler source (already normalized)."""
    raw_doc = input_data.get("normalized_document")
    if isinstance(raw_doc, dict):
        return NormalizedDocument.model_validate(raw_doc)
    return raw_doc


def _normalize_from_search(
    selected_result: dict[str, Any],
    collection_id: str,
    context,  # ToolContext
) -> NormalizedDocument:
    """Build NormalizedDocument from search result.

    BREAKING CHANGE (Option A):
    context is now ToolContext, access via context.scope.X and context.business.X
    """
    tenant_id = context.scope.tenant_id
    workflow_id = (
        context.business.workflow_id
        or context.business.case_id
        or "universal_ingestion"
    )

    if not tenant_id:
        raise ValueError("tenant_id_missing_in_context")

    # Generate ID
    document_id = uuid4()

    ref = DocumentRef(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        document_id=document_id,
        collection_id=collection_id,
    )

    # Use provided snippet as title/desc if title missing, or use result title
    meta = DocumentMeta(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        title=selected_result.get("title") or "Search Result",
        origin_uri=selected_result.get("url"),
        external_ref={
            "provider": "web_search",
            "external_id": selected_result.get("url", "")[:128],
        },
    )

    # Use ExternalBlob for the URL
    blob = {
        "type": "external",
        "kind": "https",
        "uri": selected_result.get("url"),
    }

    # Calculate deterministic checksum from URL
    url = selected_result.get("url", "")
    url_checksum = hashlib.sha256(url.encode("utf-8")).hexdigest()

    norm_doc = NormalizedDocument(
        ref=ref,
        meta=meta,
        blob=blob,
        checksum=url_checksum,
        source="other",
        created_at=datetime.now(timezone.utc),
    )

    return norm_doc


def _ensure_embedding_enabled(
    norm_doc: NormalizedDocument,
    source: str,
) -> NormalizedDocument:
    """Force enable_embedding=True for search/crawler/upload sources."""
    if source in ("search", "crawler", "upload") or source == "other":
        raw_app_config = norm_doc.meta.pipeline_config or {}
        # Check if explicitly disabled, otherwise enable
        if raw_app_config.get("enable_embedding") is not False:
            # We need to update the immutable Pydantic model
            new_config = dict(raw_app_config)
            new_config["enable_embedding"] = True

            # Careful update of nested Pydantic model
            updated_meta = norm_doc.meta.model_copy(
                update={"pipeline_config": new_config}
            )
            norm_doc = norm_doc.model_copy(update={"meta": updated_meta})

    return norm_doc


@observe_span(name="node.normalize")
def normalize_document_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Normalize input into a NormalizedDocument object.

    BREAKING CHANGE (Option A):
    Builds ToolContext from context dict to pass to helper functions.
    """
    from ai_core.tool_contracts import ToolContext

    inp = state.get("input", {})
    source = inp.get("source")
    context_dict = state.get("context", {})

    # Build ToolContext for helper functions
    tool_context = ToolContext.model_validate(context_dict)

    try:
        # Check explicit normalized input first
        raw_doc = inp.get("normalized_document")
        if raw_doc:
            norm_doc = _normalize_from_crawler(inp)
        elif source == "upload":
            norm_doc = _build_normalized_document(inp, tool_context)
        elif source == "search":
            selected = state.get("selected_result")
            if not selected:
                raise ValueError("Missing selected_result for search source")
            collection_id = inp.get("collection_id")
            if not collection_id:
                raise ValueError("Missing collection_id for search ingestion")
            norm_doc = _normalize_from_search(selected, collection_id, tool_context)
        else:
            return {"error": "Could not verify normalized document"}

        # Ensure embedding enabled for all sources
        norm_doc = _ensure_embedding_enabled(norm_doc, source)

        return {"normalized_document": norm_doc}

    except ValidationError as ve:
        logger.error(f"Normalization failed: {ve}")
        return {"error": f"Normalization failed: {ve}"}
    except Exception as exc:
        logger.exception("Unexpected normalization error")
        return {"error": str(exc)}


@observe_span(name="node.persist")
def persist_node(state: UniversalIngestionState) -> dict[str, Any]:
    """Persist the normalized document (upsert).

    BREAKING CHANGE (Option A):
    Extracts ScopeContext from ToolContext (no more business IDs in ScopeContext).
    """
    from ai_core.tool_contracts import ToolContext

    norm_doc = state.get("normalized_document")
    if not norm_doc:
        return {"error": "No normalized document to persist"}

    context_dict = state.get("context", {})
    service = _get_documents_repository()

    try:
        # Build ToolContext first
        tool_context = ToolContext.model_validate(context_dict)

        # Extract ScopeContext (already built, no business IDs)
        # BREAKING CHANGE: ScopeContext no longer has case_id, workflow_id, collection_id
        scope = tool_context.scope

        # Upsert with scope to preserve lineage
        audit_meta = tool_context.metadata.get("audit_meta")
        saved_doc = service.upsert(norm_doc, scope=scope, audit_meta=audit_meta)

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
    """Run document processing (embedding, etc).

    BREAKING CHANGE (Option A):
    Extracts IDs from ToolContext instead of dict.
    """
    from ai_core.tool_contracts import ToolContext

    norm_doc = state.get("normalized_document")
    context_dict = state.get("context", {})

    if not norm_doc:
        return {"error": "No normalized document to process"}

    # Build ToolContext
    tool_context = ToolContext.model_validate(context_dict)

    # Create DocumentProcessingContext from the document and environment
    processing_context = DocumentProcessingContext.from_document(
        norm_doc,
        case_id=tool_context.business.case_id,
        trace_id=tool_context.scope.trace_id,
        span_id=tool_context.metadata.get("span_id"),
    )

    # DocumentProcessingState requires document, config, and context
    # config must be DocumentPipelineConfig object, not dict
    raw_config = norm_doc.meta.pipeline_config or {}

    # Filter config to only valid fields for DocumentPipelineConfig
    valid_config_fields = set(DocumentPipelineConfig.__dataclass_fields__.keys())
    filtered_config = {k: v for k, v in raw_config.items() if k in valid_config_fields}
    pipeline_config = DocumentPipelineConfig(**filtered_config)

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
    """Map results to final output.

    BREAKING CHANGE (Option A):
    Extracts IDs from ToolContext instead of dict.
    """
    from ai_core.tool_contracts import ToolContext

    error = state.get("error")
    context_dict = state.get("context", {})

    tool_context = None
    try:
        tool_context = ToolContext.model_validate(context_dict)
    except ValidationError:
        if not error:
            raise

    if tool_context is None:
        telemetry = {
            "trace_id": None,
            "tenant_id": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service_id": None,
            "invocation_id": None,
        }
    else:
        telemetry = {
            "trace_id": tool_context.scope.trace_id,
            "tenant_id": tool_context.scope.tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # Pre-MVP ID Contract: identity tracking
            "service_id": tool_context.scope.service_id,
            "invocation_id": tool_context.scope.invocation_id,
        }

    if error:
        return {
            "output": {
                "decision": "error",
                "reason": error,
                "telemetry": telemetry,
                "ingestion_run_id": (
                    tool_context.scope.ingestion_run_id if tool_context else None
                ),
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
    # Construct transitions dynamically based on execution path
    inp_source = state.get("input", {}).get("source")

    if inp_source == "search":
        if doc_id:
            # acquire_and_ingest
            transitions = [
                "validate_input",
                "search",
                "select",
                "normalize",
                "persist",
                "process",
                "finalize",
            ]
        else:
            # acquire_only
            transitions = ["validate_input", "search", "select", "finalize"]
    else:
        # standard ingestion (upload/crawler)
        transitions = ["validate_input", "normalize", "persist", "process", "finalize"]

    # Construct review_payload for HITL signaling
    review_payload = norm_doc.model_dump() if norm_doc else None

    # Determine decision with strict invariants
    if doc_id:
        decision = "ingested"
    elif state.get("selected_result"):
        # If we didn't persist but had a selection (and no error), it's an acquisition
        decision = "acquired"
    else:
        # Fallback if neither persisted nor acquired (e.g. skipped or empty)
        decision = "skipped"

    return {
        "output": {
            "decision": decision,
            "reason": "Success",
            "telemetry": telemetry,
            "ingestion_run_id": tool_context.scope.ingestion_run_id,
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


def _clear_cached_processing_graph():
    """Clear the cached processing graph (for testing/cleanup).

    This function is useful for:
    - Test isolation (prevent cache pollution between tests)
    - Memory management (force garbage collection of large graph)
    - Development (reload graph after code changes)
    """
    global _CACHED_PROCESSING_GRAPH
    _CACHED_PROCESSING_GRAPH = None
    logger.info("Processing graph cache cleared")


def build_universal_ingestion_graph() -> Any:
    """Build and compile the universal ingestion graph."""
    workflow = StateGraph(UniversalIngestionState)

    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("search", search_node)
    workflow.add_node("select", selection_node)
    workflow.add_node("normalize", normalize_document_node)
    workflow.add_node("persist", persist_node)

    # Process node will now use the cached sub-graph internally
    workflow.add_node("process", process_node)

    workflow.add_node("finalize", finalize_node)

    # Edges
    workflow.add_edge(START, "validate_input")

    def check_validation(
        state: UniversalIngestionState,
    ) -> Literal["search", "normalize", "finalize"]:
        if state.get("error"):
            return "finalize"

        inp = state.get("input", {})
        if inp.get("source") == "search":
            return "search"

        return "normalize"

    workflow.add_conditional_edges("validate_input", check_validation)

    # Search flow
    workflow.add_edge("search", "select")

    def check_selection(
        state: UniversalIngestionState,
    ) -> Literal["normalize", "finalize"]:
        if state.get("error"):
            return "finalize"

        inp = state.get("input", {})
        mode = inp.get("mode")

        # If acquire_only, we stop after selection (staged results returned)
        if mode == "acquire_only":
            return "finalize"

        # For acquire_and_ingest, we proceed to normalize
        return "normalize"

    workflow.add_conditional_edges("select", check_selection)

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
