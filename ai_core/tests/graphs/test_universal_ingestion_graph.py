import pytest
import sys
from unittest.mock import MagicMock, patch
from documents.contracts import FileBlob

# Remove top-level import to allow pre-import patching
# from ai_core.graphs.technical.universal_ingestion_graph import ...
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext


def _tool_context(
    *,
    tenant_id: str,
    trace_id: str,
    invocation_id: str,
    run_id: str | None = None,
    ingestion_run_id: str | None = None,
    case_id: str | None = None,
    workflow_id: str | None = None,
    metadata: dict[str, object] | None = None,
):
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id=trace_id,
        invocation_id=invocation_id,
        run_id=run_id,
        ingestion_run_id=ingestion_run_id,
        service_id="test-worker",
    )
    business = BusinessContext(case_id=case_id, workflow_id=workflow_id)
    return scope.to_tool_context(business=business, metadata=metadata or {})


@pytest.fixture
def utg_module():
    """Import the module with observe_span patched out."""
    with patch(
        "ai_core.infra.observability.observe_span",
        side_effect=lambda name=None, **kwargs: lambda func: func,
    ):
        # We need to reload or import strictly here.
        # Since it might be already imported by other tests (unlikely in this single run but possible),
        # we strictly want to ensure we get a version where decorators ran against our mock.
        if "ai_core.graphs.technical.universal_ingestion_graph" in sys.modules:
            del sys.modules["ai_core.graphs.technical.universal_ingestion_graph"]

        import ai_core.graphs.technical.universal_ingestion_graph as m

        # Reset the cached graph to ensure building logic runs (and picks up patches)
        m._CACHED_PROCESSING_GRAPH = None
        yield m


@pytest.fixture
def mock_processing_graph(utg_module):
    # We patch the object in the imported module returned by our fixture
    # But since _get_cached_processing_graph uses local import, we must patch the source
    # We must patch the reference HELD by the imported module, because utg_module
    # performs the import before this fixture runs.
    with patch.object(utg_module, "build_document_processing_graph") as mock:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"status": "processed"}
        mock.return_value = mock_graph
        yield mock


@pytest.fixture
def mock_document_service(utg_module):
    # Patch the function that returns the repository
    # Note: _get_documents_repository is imported in universal_ingestion_graph
    with patch.object(utg_module, "_get_documents_repository") as get_repo_mock:
        repo_mock = MagicMock()
        repo_mock.upsert.return_value = MagicMock(id="doc-123")
        # Ensure the attribute-style access used in fix also works if returned object is specialized
        repo_mock.upsert.return_value.ref = MagicMock(
            document_id="00000000-0000-0000-0000-000000000123"
        )

        get_repo_mock.return_value = repo_mock
        yield repo_mock


def test_universal_ingestion_graph_validation_error(
    utg_module, mock_processing_graph, mock_document_service
):
    """Test that missing inputs trigger a validation error."""
    graph = utg_module.build_universal_ingestion_graph()

    # Missing optional input for upload
    state = {
        "input": {
            "source": "upload",
            "mode": "ingest_only",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "upload_blob": None,  # Invalid
            "metadata_obj": {},
            "normalized_document": None,
        },
        "context": _tool_context(
            tenant_id="tenant-1",
            trace_id="trace-1",
            invocation_id="inv-1",
            run_id="run-1",
            case_id="case-1",
        ),
    }

    result = graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "error"
    assert "Missing upload_blob" in output["reason"]
    assert "Missing upload_blob" in output["reason"]
    assert output["document_id"] is None
    # Roadmap compliance checks
    assert output["transitions"] == []
    assert output["review_payload"] is None
    assert output["hitl_required"] is False


def test_universal_ingestion_graph_success_crawler(
    utg_module, mock_processing_graph, mock_document_service
):
    """Test successful run with crawler source and pre-normalized document."""
    graph = utg_module.build_universal_ingestion_graph()

    # Mock a valid normalized document payload
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
        "blob": FileBlob(
            type="file",
            uri="s3://bucket/key",
            media_type="text/html",
            size=100,
            sha256="0" * 64,
        ).model_dump(exclude_none=True),
        "checksum": "0" * 64,
        "created_at": "2024-01-01T00:00:00Z",
        "lifecycle_state": "active",
        "source": "crawler",
    }

    state = {
        "input": {
            "source": "crawler",
            "mode": "ingest_only",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "upload_blob": None,
            "metadata_obj": None,
            "normalized_document": norm_doc_payload,
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000001",
            invocation_id="inv-crawler-test",
            ingestion_run_id="00000000-0000-0000-0000-000000000001",
            case_id="00000000-0000-0000-0000-000000000001",
        ),
    }

    result = graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "ingested"
    assert output["document_id"] == "00000000-0000-0000-0000-000000000123"

    # Roadmap compliance checks
    assert "transitions" in output
    assert output["transitions"] == [
        "validate_input",
        "normalize",
        "dedup",  # P2 Fix: Include dedup node
        "persist",
        "process",
        "finalize",
    ]
    assert output["review_payload"] is not None
    assert (
        str(output["review_payload"]["ref"]["document_id"])
        == "00000000-0000-0000-0000-000000000123"
    )
    assert (
        str(output["normalized_document_ref"]["document_id"])
        == "00000000-0000-0000-0000-000000000123"
    )
    assert output["hitl_required"] is False

    # Verify persistence called
    mock_document_service.upsert.assert_called_once()

    # Verify processing called
    mock_processing_graph.return_value.invoke.assert_called_once()


def test_universal_ingestion_graph_missing_context(
    utg_module, mock_processing_graph, mock_document_service
):
    """Test that missing context ID triggers error."""
    graph = utg_module.build_universal_ingestion_graph()

    state = {
        "input": {
            "source": "crawler",
            "mode": "ingest_only",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "normalized_document": {},
        },
        "context": {},
    }

    result = graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "error"
    assert "Invalid context structure" in output["reason"]


def test_universal_ingestion_graph_success_upload(
    utg_module, mock_processing_graph, mock_document_service
):
    """Test successful run with upload source and blob payload."""
    graph = utg_module.build_universal_ingestion_graph()

    # Mock a valid upload blob
    upload_blob = {
        "type": "file",
        "uri": "objectstore://bucket/key",  # Standardized URI
        "media_type": "application/pdf",
        "size": 1024,
        "sha256": "a" * 64,
    }

    metadata = {"file_name": "test.pdf", "mime_type": "application/pdf"}

    state = {
        "input": {
            "source": "upload",
            "mode": "ingest_only",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "upload_blob": upload_blob,
            "metadata_obj": metadata,
            "normalized_document": None,
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000001",
            invocation_id="inv-upload-test",
            ingestion_run_id="00000000-0000-0000-0000-000000000001",
            case_id="00000000-0000-0000-0000-000000000001",
        ),
    }

    result = graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "ingested", f"FAILED REASON: {output.get('reason')}"
    assert output["document_id"] is not None  # Generated in _build_normalized_document

    # Roadmap compliance checks
    assert output["transitions"] == [
        "validate_input",
        "normalize",
        "dedup",  # P2 Fix: Include dedup node
        "persist",
        "process",
        "finalize",
    ]
    assert output["review_payload"] is not None
    assert output["normalized_document_ref"] is not None
    assert output["hitl_required"] is False

    # Verify persistence called with correct structure
    mock_document_service.upsert.assert_called_once()
    saved_doc = mock_document_service.upsert.call_args[0][0]

    assert str(saved_doc.ref.collection_id) == "00000000-0000-0000-0000-000000000001"
    assert saved_doc.source == "upload"
    assert saved_doc.blob.uri == "objectstore://bucket/key"

    # Verify processing called
    mock_processing_graph.return_value.invoke.assert_called_once()


# ===== Search Source Tests =====


@pytest.fixture
def mock_search_worker():
    """Mock search worker that returns test results."""

    class MockSearchWorker:
        def run(self, query: str, context: dict):
            from ai_core.tools.web_search import (
                SearchResult,
                WebSearchResponse,
                ToolOutcome,
            )

            results = [
                SearchResult(
                    url="https://example.com/result1.pdf",
                    title="Test Result 1",
                    snippet="This is a test snippet with enough content to pass validation minimum length",
                    source="test_provider",
                    is_pdf=True,
                ),
                SearchResult(
                    url="https://example.com/result2.html",
                    title="Test Result 2",
                    snippet="Another test snippet with sufficient length for validation checks to pass",
                    source="test_provider",
                    is_pdf=False,
                ),
            ]

            # Return proper WebSearchResponse object
            outcome = ToolOutcome(
                decision="ok",
                rationale="Search completed successfully",
                meta={},
            )
            return WebSearchResponse(results=results, outcome=outcome)

    return MockSearchWorker()


def test_search_source_acquire_and_ingest(
    utg_module, mock_processing_graph, mock_document_service, mock_search_worker
):
    """Test search with query → acquisition → ingestion."""
    graph = utg_module.build_universal_ingestion_graph()

    input_payload = {
        "source": "search",
        "mode": "acquire_and_ingest",
        "collection_id": "00000000-0000-0000-0000-000000000001",
        "upload_blob": None,
        "metadata_obj": None,
        "normalized_document": None,
        "search_query": "test query for ingestion",
        "search_config": {
            "min_snippet_length": 40,
            "blocked_domains": [],
            "top_n": 5,
            "prefer_pdf": True,
        },
        "preselected_results": None,
    }

    context = _tool_context(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        ingestion_run_id="run-1",
        case_id="case-1",
        metadata={"runtime_worker": mock_search_worker},
    )

    result = graph.invoke({"input": input_payload, "context": context})

    output = result["output"]
    assert output["decision"] == "ingested"
    assert output["document_id"] is not None
    assert "search" in output["transitions"]
    assert "select" in output["transitions"]
    assert "normalize" in output["transitions"]
    assert "persist" in output["transitions"]
    assert "process" in output["transitions"]


def test_search_source_acquire_only(utg_module, mock_search_worker):
    """Test search with acquire_only mode (no ingestion)."""
    graph = utg_module.build_universal_ingestion_graph()

    input_payload = {
        "source": "search",
        "mode": "acquire_only",
        "collection_id": "00000000-0000-0000-0000-000000000001",
        "search_query": "test query acquisition only",
        "search_config": {
            "min_snippet_length": 40,
            "blocked_domains": [],
            "top_n": 5,
            "prefer_pdf": False,
        },
        "upload_blob": None,
        "metadata_obj": None,
        "normalized_document": None,
        "preselected_results": None,
    }

    context = _tool_context(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        ingestion_run_id="run-1",
        case_id="case-1",
        metadata={"runtime_worker": mock_search_worker},
    )

    result = graph.invoke({"input": input_payload, "context": context})

    output = result["output"]
    assert output["decision"] == "acquired"
    assert output["document_id"] is None  # Not persisted in acquire_only mode
    assert "search" in output["transitions"]
    assert "select" in output["transitions"]
    assert "finalize" in output["transitions"]
    # Should NOT have normalize, persist, or process
    assert "normalize" not in output["transitions"]
    assert "persist" not in output["transitions"]
    assert "process" not in output["transitions"]


def test_search_source_with_preselected_results(
    utg_module, mock_processing_graph, mock_document_service
):
    """Test search with preselected_results (bypass search worker)."""
    graph = utg_module.build_universal_ingestion_graph()

    input_payload = {
        "source": "search",
        "mode": "acquire_and_ingest",
        "collection_id": "00000000-0000-0000-0000-000000000001",
        "search_query": None,  # Not required when preselected provided
        "search_config": None,
        "preselected_results": [
            {
                "url": "https://example.com/doc1.pdf",
                "title": "Preselected Doc 1",
                "snippet": "Test snippet content here with sufficient length",
                "is_pdf": True,
            },
        ],
        "upload_blob": None,
        "metadata_obj": None,
        "normalized_document": None,
    }

    context = _tool_context(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        ingestion_run_id="run-1",
        case_id="case-1",
    )

    result = graph.invoke({"input": input_payload, "context": context})

    output = result["output"]
    assert output["decision"] == "ingested"
    assert output["document_id"] is not None


def test_search_source_missing_query_and_preselected(utg_module):
    """Test search fails when both query and preselected_results are missing."""
    graph = utg_module.build_universal_ingestion_graph()

    input_payload = {
        "source": "search",
        "mode": "acquire_and_ingest",
        "collection_id": "00000000-0000-0000-0000-000000000001",
        "search_query": None,  # Missing
        "preselected_results": None,  # Also missing
        "search_config": None,
        "upload_blob": None,
        "metadata_obj": None,
        "normalized_document": None,
    }

    context = _tool_context(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        ingestion_run_id="run-1",
        case_id="case-1",
    )

    result = graph.invoke({"input": input_payload, "context": context})

    output = result["output"]
    assert output["decision"] == "error"
    assert (
        "search_query" in output["reason"].lower()
        or "preselected" in output["reason"].lower()
    )


def test_search_source_checksum_is_url_hash(
    utg_module, mock_processing_graph, mock_document_service
):
    """Test that search results use SHA256(URL) as checksum, not magic string."""
    import hashlib

    graph = utg_module.build_universal_ingestion_graph()

    test_url = "https://example.com/test-document.pdf"
    expected_checksum = hashlib.sha256(test_url.encode("utf-8")).hexdigest()

    input_payload = {
        "source": "search",
        "mode": "acquire_and_ingest",
        "collection_id": "00000000-0000-0000-0000-000000000001",
        "search_query": None,
        "search_config": None,
        "preselected_results": [
            {
                "url": test_url,
                "title": "Test Document",
                "snippet": "Test snippet content with sufficient length for validation",
                "is_pdf": True,
            },
        ],
        "upload_blob": None,
        "metadata_obj": None,
        "normalized_document": None,
    }

    context = _tool_context(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-checksum-test",
        ingestion_run_id="run-1",
        case_id="case-1",
    )

    result = graph.invoke({"input": input_payload, "context": context})

    # Verify the normalized document has correct checksum
    norm_doc = result.get("normalized_document")
    assert norm_doc is not None
    assert (
        norm_doc.checksum == expected_checksum
    ), f"Expected checksum to be SHA256 of URL, got {norm_doc.checksum}"

    # Verify it's NOT the old magic string
    assert norm_doc.checksum != "0" * 64, "Checksum should not be magic string"


# ===== Additional Error Handling Tests =====


def test_persist_node_missing_invocation_id(utg_module):
    """Test that persist_node fails when invocation_id is missing (no fallback)."""
    from documents.contracts import NormalizedDocument, DocumentRef, DocumentMeta
    from uuid import uuid4

    doc_id = str(uuid4())
    collection_id = str(uuid4())
    norm_doc = NormalizedDocument(
        ref=DocumentRef(
            tenant_id="tenant-1",
            workflow_id="wf-1",
            document_id=doc_id,
            collection_id=collection_id,
        ),
        meta=DocumentMeta(
            tenant_id="tenant-1",
            workflow_id="wf-1",
            title="Test Doc",
        ),
        blob=FileBlob(
            type="file",
            uri="s3://test/key",
            media_type="text/html",
            size=100,
            sha256="0" * 64,
        ),
        checksum="0" * 64,
        created_at="2024-01-01T00:00:00Z",
        lifecycle_state="active",
        source="other",
    )

    state = {
        "normalized_document": norm_doc,
        "context": {},
        "input": {"collection_id": collection_id},
    }

    result = utg_module.persist_node(state)

    # Should return error because invocation_id is mandatory (KeyError or validation error)
    assert "error" in result
    assert result["error"] is not None


def test_unsupported_mode(utg_module, mock_processing_graph, mock_document_service):
    """Test error when mode is invalid."""
    graph = utg_module.build_universal_ingestion_graph()

    upload_blob = {
        "type": "file",
        "uri": "objectstore://bucket/key",
        "media_type": "application/pdf",
        "size": 1024,
        "sha256": "a" * 64,
    }

    input_payload = {
        "source": "upload",
        "mode": "invalid_mode",  # Not in allowed modes!
        "collection_id": "00000000-0000-0000-0000-000000000001",
        "upload_blob": upload_blob,
        "metadata_obj": {"file_name": "test.pdf"},
        "normalized_document": None,
        "search_query": None,
        "search_config": None,
        "preselected_results": None,
    }

    context = _tool_context(
        tenant_id="tenant-1",
        trace_id="trace-1",
        invocation_id="inv-1",
        ingestion_run_id="run-1",
        case_id="case-1",
    )

    result = graph.invoke({"input": input_payload, "context": context})

    output = result["output"]
    assert output["decision"] == "error"
    assert "mode" in output["reason"].lower()
