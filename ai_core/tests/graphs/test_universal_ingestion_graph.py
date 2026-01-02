import pytest
import sys
from unittest.mock import MagicMock, patch

# Remove top-level import to allow pre-import patching
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext


def _context_payload(
    *,
    tenant_id: str,
    trace_id: str,
    invocation_id: str,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    case_id: str | None = None,
    workflow_id: str | None = None,
    collection_id: str | None = None,
    metadata: dict[str, object] | None = None,
):
    if not run_id and not ingestion_run_id:
        run_id = "run-default"
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
        service_id="test-worker",
    )
    business = BusinessContext(
        case_id=case_id, workflow_id=workflow_id, collection_id=collection_id
    )
    # Return as raw dict (payload style) or ToolContext object
    # The graph accepts raw dict in 'context' and validates it
    tc = scope.to_tool_context(business=business, metadata=metadata or {})
    return tc.model_dump(mode="json")


@pytest.fixture
def utg_module():
    """Import the module with observe_span patched out."""
    with patch(
        "ai_core.infra.observability.observe_span",
        side_effect=lambda name=None, **kwargs: lambda func: func,
    ):
        if "ai_core.graphs.technical.universal_ingestion_graph" in sys.modules:
            del sys.modules["ai_core.graphs.technical.universal_ingestion_graph"]

        import ai_core.graphs.technical.universal_ingestion_graph as m

        yield m


@pytest.fixture
def mock_processing_graph(utg_module):
    # Patch build_document_processing_graph in the module
    with patch.object(utg_module, "build_document_processing_graph") as mock:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"status": "processed"}
        mock.return_value = mock_graph
        yield mock


@pytest.fixture
def mock_document_service(utg_module):
    # Patch _get_documents_repository
    with patch.object(utg_module, "_get_documents_repository") as get_repo_mock:
        repo_mock = MagicMock()

        # Pass-through the document to preserve pipeline_config and other metadata
        # which allows process_node to see the correct flags (e.g. enable_embedding)
        def _upsert_side_effect(doc, scope=None, audit_meta=None):
            # Simulate ID assignment if needed, but for unit test, input doc is fine
            # If doc.ref.document_id is None, validation might fail, but test inputs usually have it.
            return doc

        repo_mock.upsert.side_effect = _upsert_side_effect

        get_repo_mock.return_value = repo_mock
        yield repo_mock


def test_uig_success_path(utg_module, mock_processing_graph, mock_document_service):
    """Test successful ingestion of a normalized document."""
    graph = utg_module.build_universal_ingestion_graph()

    # Mock Input
    norm_doc_payload = {
        "ref": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "document_id": "00000000-0000-0000-0000-000000000123",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
        },
        "meta": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
            "pipeline_config": {"enable_embedding": True},
        },
        "blob": {
            "type": "inline",
            "media_type": "text/plain",
            "base64": "SGVsbG8=",
            "sha256": "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969",
            "size": 5,
        },
        "checksum": "185f8db32271fe25f561a6fc938b2e264306ec304eda518007d1764826381969",
        "created_at": "2024-01-01T00:00:00Z",
        "source": "upload",
    }

    context = _context_payload(
        tenant_id="00000000-0000-0000-0000-000000000001",
        trace_id="trace-1",
        invocation_id="inv-1",
        collection_id="00000000-0000-0000-0000-000000000001",
    )

    state = {
        "input": {"normalized_document": norm_doc_payload},
        "context": context,
    }

    # Verify Upsert

    # Verify Processing
    # Wraps dependency patching since process_node imports them
    with patch("documents.parsers.create_default_parser_dispatcher"), patch(
        "ai_core.rag.chunking.RoutingAwareChunker"
    ):
        result = graph.invoke(state)
        output = result["output"]

        assert (
            output["decision"] == "processed"
        ), f"Failed with error: {output.get('error')} | Reason: {output.get('reason')}"
        assert output["document_id"] == "00000000-0000-0000-0000-000000000123"

        # Verify Upsert
        mock_document_service.upsert.assert_called_once()

        # Verify Processing Graph was called
        mock_processing_graph.return_value.invoke.assert_called_once()


def test_uig_missing_collection_id_in_context(utg_module):
    """Test validation failure when collection_id is missing from BusinessContext."""
    graph = utg_module.build_universal_ingestion_graph()

    # Even if doc is valid, missing collection_id in context triggers failure
    norm_doc = {
        "ref": {
            "document_id": "00000000-0000-0000-0000-000000000000",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
            "collection_id": "00000000-0000-0000-0000-000000000001",
        },
        "meta": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
        },
        "blob": {
            "type": "inline",
            "media_type": "text/plain",
            "base64": "AA==",
            "sha256": "0" * 64,
            "size": 1,
        },
        "checksum": "0" * 64,
        "created_at": "2024-01-01T00:00:00Z",
        "source": "upload",
    }

    context = _context_payload(
        tenant_id="t1",
        trace_id="tr1",
        invocation_id="inv1",
        collection_id=None,  # Missing!
    )

    state = {
        "input": {"normalized_document": norm_doc},
        "context": context,
    }

    result = graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "failed"
    assert output["reason_code"] == "VALIDATION_ERROR"
    assert "Missing collection_id" in output["reason"]


def test_uig_duplicate_logic(utg_module, mock_processing_graph, mock_document_service):
    """Test behavior when dedup node identifies a duplicate (Mocked via state injection if possible or modifying node logic).

    Since we can't easily inject 'dedup_status' into the graph internal flow without mocking the node,
    we will stick to checking that MVP logic returns 'new'.
    """
    graph = utg_module.build_universal_ingestion_graph()

    # ... setup valid input ...
    norm_doc_payload = {
        "ref": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "document_id": "00000000-0000-0000-0000-000000000123",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
        },
        "meta": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
        },
        "blob": {
            "type": "inline",
            "media_type": "text/plain",
            "base64": "AA==",
            "sha256": "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            "size": 1,
        },
        "checksum": "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
        "created_at": "2024-01-01T00:00:00Z",
        "source": "upload",
    }
    context = _context_payload(
        tenant_id="t1",
        trace_id="tr1",
        invocation_id="inv1",
        collection_id="00000000-0000-0000-0000-000000000001",
    )

    state = {
        "input": {"normalized_document": norm_doc_payload},
        "context": context,
    }

    # In MVP, dedup logic always returns "new", so it should proceed to persist
    with patch("documents.parsers.create_default_parser_dispatcher"), patch(
        "ai_core.rag.chunking.RoutingAwareChunker"
    ):

        result = graph.invoke(state)
        output = result["output"]
        assert (
            output["decision"] == "processed"
        ), f"Failed with error: {output.get('error')} | Reason: {output.get('reason')}"

    # Ensure intermediate state had dedup_status="new" (if we could check it, but we only get output)


def test_uig_store_only_mode(utg_module, mock_processing_graph, mock_document_service):
    """Test that enable_embedding=False skips processing."""
    graph = utg_module.build_universal_ingestion_graph()

    norm_doc_payload = {
        "ref": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "document_id": "00000000-0000-0000-0000-000000000123",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
        },
        "meta": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "workflow_id": "00000000-0000-0000-0000-000000000001",
            "pipeline_config": {"enable_embedding": False},
        },
        "blob": {
            "type": "inline",
            "media_type": "text/plain",
            "base64": "AA==",
            "sha256": "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            "size": 1,
        },
        "checksum": "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
        "created_at": "2024-01-01T00:00:00Z",
        "source": "upload",
    }
    context = _context_payload(
        tenant_id="00000000-0000-0000-0000-000000000001",
        trace_id="tr1",
        invocation_id="inv1",
        collection_id="00000000-0000-0000-0000-000000000001",
    )

    state = {"input": {"normalized_document": norm_doc_payload}, "context": context}

    # Patch dependencies even for skipped processing to prevent import/instantiation errors
    with patch("documents.parsers.create_default_parser_dispatcher"), patch(
        "ai_core.rag.chunking.RoutingAwareChunker"
    ):
        result = graph.invoke(state)
        output = result["output"]

        assert (
            output["decision"] == "processed"
        ), f"Failed with error: {output.get('error')} | Reason: {output.get('reason')}"

        # Persist called
        mock_document_service.upsert.assert_called_once()

        # Processing SKIPPED
        mock_processing_graph.return_value.invoke.assert_not_called()
