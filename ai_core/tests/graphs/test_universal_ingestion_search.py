import pytest
import sys
from unittest.mock import MagicMock, patch

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tools.web_search import (
    WebSearchResponse,
    WebSearchWorker,
    ToolOutcome,
    SearchResult,
)


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


@pytest.fixture(autouse=True)
def _prevent_db_access():
    """Ensure no DB access during these tests."""
    with patch(
        "django.db.backends.base.base.BaseDatabaseWrapper.connect",
        side_effect=RuntimeError("DB_ACCESS_FORBIDDEN"),
    ):
        yield


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

        # Reset the cached graph
        m._CACHED_PROCESSING_GRAPH = None
        yield m


@pytest.fixture
def mock_search_worker():
    worker = MagicMock(spec=WebSearchWorker)

    # Setup default response
    results = [
        SearchResult(
            url="https://example.com/1",
            title="Example 1",
            snippet="This is a test snippet for the first result that is also long enough.",
            is_pdf=False,
            source="mock_provider",
        ),
        SearchResult(
            url="https://example.com/2",
            title="Example 2",
            snippet="Another snippet here that is definitely long enough to pass the minimum length filter for valid search results.",
            is_pdf=True,
            source="mock_provider",
        ),
    ]
    response = WebSearchResponse(
        # WebSearchResponse no longer takes query in constructor, just results and outcome
        results=results,
        outcome=ToolOutcome(decision="ok", rationale="test", meta={}),
    )
    worker.run.return_value = response
    return worker


@pytest.fixture
def mock_document_service():
    repo_mock = MagicMock()
    # Mock upsert return
    saver = MagicMock()
    saver.id = "doc-123"
    saver.ref.document_id = "doc-123"
    repo_mock.upsert.return_value = saver
    return repo_mock


@pytest.fixture
def mock_dependencies(utg_module, mock_document_service):
    """Mock external dependencies for all tests."""
    # Use patch.object on the reloaded module to ensure we patch the right objects
    with patch.object(
        utg_module, "_get_documents_repository", return_value=mock_document_service
    ), patch.object(
        utg_module, "_get_cached_processing_graph"
    ) as mock_processing, patch.object(
        utg_module, "uuid4", return_value="00000000-0000-0000-0000-000000000123"
    ):
        yield {"processing_graph": mock_processing}


def test_search_acquire_only(utg_module, mock_search_worker, mock_dependencies):
    """Test source=search with mode=acquire_only."""
    graph = utg_module.build_universal_ingestion_graph()

    state = {
        "input": {
            "source": "search",
            "mode": "acquire_only",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "search_query": "test query",
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000002",
            invocation_id="inv-acquire-only",
            run_id="run-acquire-only",
            case_id="00000000-0000-0000-0000-000000000003",
            metadata={"runtime_worker": mock_search_worker},
        ),
    }

    result = graph.invoke(state)
    output = result["output"]

    # If filtered results are mapped to search_results state
    search_results = result.get("search_results")
    assert search_results is not None
    assert len(search_results) == 2

    # In acquire_only, we expect decision to be empty or skipped or just output generated?
    # Our finalize_node handles output mapping.
    # If we stop at finalize, output should be generated based on successful run.
    # Current finalize_node focuses on 'ingested' decision if normalized_doc exists.
    # If we don't normalize (acquire_only), we might need to adjust finalize_node to handle 'acquired' status?
    # OR, we check if finalize_node correctly handles missing normalized_doc for search?

    # Let's inspect what finalize_node does:
    # `norm_doc = state.get("normalized_document")` -> None
    # `review_payload = ...` -> None
    # returns {"output": { "decision": "ingested", ... "normalized_document_ref": None }}
    # It defaults to "ingested" / "Success".
    # This might be misleading for "acquire_only".
    # But for MVP smoke test, let's verify we got here without error.

    assert result.get("error") is None
    assert output["decision"] == "acquired"  # Was "ingested", now correct status
    assert output["document_id"] is None
    assert output["telemetry"]["trace_id"] == "00000000-0000-0000-0000-000000000002"
    assert output["transitions"] == ["validate_input", "search", "select", "finalize"]


def test_search_acquire_and_ingest(
    utg_module, mock_search_worker, mock_document_service, mock_dependencies
):
    """Test source=search with mode=acquire_and_ingest."""
    graph = utg_module.build_universal_ingestion_graph()

    state = {
        "input": {
            "source": "search",
            "mode": "acquire_and_ingest",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "search_query": "test query",
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000002",
            invocation_id="inv-search-test",
            ingestion_run_id="00000000-0000-0000-0000-000000000004",
            case_id="00000000-0000-0000-0000-000000000003",
            workflow_id="workflow-123",
            metadata={"runtime_worker": mock_search_worker},
        ),
    }

    result = graph.invoke(state)

    # Check for immediate graph errors before verifying side-effects
    if result.get("error"):
        pytest.fail(f"Graph execution failed with error: {result['error']}")

    # Check selection
    selected = result.get("selected_result")
    assert selected is not None
    assert selected["url"] == "https://example.com/2"

    # Check Normalization -> Persist
    mock_document_service.upsert.assert_called_once()
    saved_doc = mock_document_service.upsert.call_args[0][0]
    assert saved_doc.meta.origin_uri == "https://example.com/2"

    # Check Processing Invocation
    mock_dependencies["processing_graph"].return_value.invoke.assert_called_once()

    output = result["output"]
    assert output["decision"] == "ingested"
    assert str(output["document_id"]) == "00000000-0000-0000-0000-000000000123"
    assert output["transitions"] == [
        "validate_input",
        "search",
        "select",
        "normalize",
        "dedup",  # P2 Fix: Include dedup node
        "persist",
        "process",
        "finalize",
    ]


def test_search_preselected_results_bypass(
    utg_module, mock_search_worker, mock_dependencies
):
    """Test that preselected_results bypass the search worker."""
    graph = utg_module.build_universal_ingestion_graph()

    preselected = [
        {
            "url": "https://example.com/preselected",
            "title": "Preselected Title",
            "snippet": "Preselected snippet must be long enough to pass the default filter of 40 characters.",
        }
    ]

    state = {
        "input": {
            "source": "search",
            "mode": "acquire_only",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "preselected_results": preselected,
            # No search_query provided ensuring validation allows it
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000002",
            invocation_id="inv-preselected",
            run_id="run-preselected",
            case_id="00000000-0000-0000-0000-000000000003",
            workflow_id="workflow-123",
            metadata={"runtime_worker": mock_search_worker},
        ),
    }

    result = graph.invoke(state)

    # 1. Verify worker was NOT called
    mock_search_worker.run.assert_not_called()

    # 2. Verify results are in state
    assert result.get("search_results") == preselected

    # 3. Verify output
    output = result.get("output")
    assert output is not None
    # acquire_only with valid selection = acquired
    assert result.get("selected_result") is not None
    assert result["selected_result"]["url"] == "https://example.com/preselected"
    assert output["decision"] == "acquired"


def test_dedup_workflow_new_document(utg_module, mock_dependencies):
    """Test that dedup_node is called and sets dedup_status='new' for new documents."""
    graph = utg_module.build_universal_ingestion_graph()

    # Use new upload_blob format instead of deprecated file_path/content_type
    upload_blob = {
        "type": "file",
        "uri": "objectstore://bucket/new-document.pdf",
        "media_type": "application/pdf",
        "size": 2048,
        "sha256": "a" * 64,
    }

    state = {
        "input": {
            "source": "upload",
            "mode": "acquire_and_ingest",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "upload_blob": upload_blob,
            "metadata_obj": {"file_name": "new-document.pdf"},
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000002",
            invocation_id="inv-dedup-new",
            ingestion_run_id="00000000-0000-0000-0000-000000000004",
        ),
    }

    result = graph.invoke(state)

    # 1. Verify dedup_status was set
    assert result.get("dedup_status") is not None
    # For now, dedup_node always returns "new" (MVP implementation)
    assert result["dedup_status"] == "new"

    # 2. Verify dedup_node was in the execution path
    output = result.get("output")
    assert output is not None
    assert "dedup" in output["transitions"]

    # 3. Verify workflow proceeded to persist and process (not skipped)
    assert "persist" in output["transitions"]
    assert "process" in output["transitions"]

    # 4. Verify decision is "ingested" (not "duplicate")
    assert output["decision"] == "ingested"


def test_dedup_workflow_duplicate_document(utg_module, mock_dependencies):
    """Test that dedup_node detects duplicates and skips persist/process."""
    # NOTE: This test requires mocking the dedup_node to return dedup_status="duplicate"
    # since the MVP implementation always returns "new"
    from unittest.mock import patch

    # P2 Fix: Upload source requires upload_blob, not file_path/content_type
    upload_blob = {
        "type": "file",
        "uri": "objectstore://bucket/fake-document.pdf",
        "media_type": "application/pdf",
        "size": 1024,
        "sha256": "b" * 64,
    }

    state = {
        "input": {
            "source": "upload",
            "mode": "acquire_and_ingest",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "upload_blob": upload_blob,
            "metadata_obj": {"file_name": "fake-document.pdf"},
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000002",
            invocation_id="inv-dedup-duplicate",
            ingestion_run_id="00000000-0000-0000-0000-000000000004",
        ),
    }

    # Mock dedup_node to return duplicate status with existing document ref
    from ai_core.ids.contracts import DocumentRef

    existing_doc_ref = DocumentRef(
        tenant_id="00000000-0000-0000-0000-000000000001",
        workflow_id="workflow-existing",
        document_id="existing-doc-123",
        collection_id="00000000-0000-0000-0000-000000000001",
    )

    def mock_dedup_node(state):
        """Mock dedup that returns duplicate status."""
        return {
            "dedup_status": "duplicate",
            "existing_document_ref": existing_doc_ref,
        }

    # P2 Fix: Patch dedup_node BEFORE building graph so compiled LangGraph uses mock
    with patch.object(utg_module, "dedup_node", side_effect=mock_dedup_node):
        graph = utg_module.build_universal_ingestion_graph()
        result = graph.invoke(state)

    # 1. Verify dedup_status was set to duplicate
    assert result.get("dedup_status") == "duplicate"

    # 2. Verify dedup was in the execution path
    output = result.get("output")
    assert output is not None
    assert "dedup" in output["transitions"]

    # 3. Verify persist and process were SKIPPED (duplicate path)
    assert "persist" not in output["transitions"]
    assert "process" not in output["transitions"]

    # 4. Verify transitions show duplicate path
    expected_transitions = ["validate_input", "normalize", "dedup", "finalize"]
    assert output["transitions"] == expected_transitions

    # 5. Verify decision is "duplicate"
    assert output["decision"] == "duplicate"

    # 6. P1 Fix: Verify document_id is the existing document's ID, not phantom new ID
    assert output["document_id"] == "existing-doc-123"


def test_dedup_workflow_duplicate_without_existing_ref(utg_module, mock_dependencies):
    """Test that duplicate without existing_document_ref returns None (no phantom ID)."""
    from unittest.mock import patch

    upload_blob = {
        "type": "file",
        "uri": "objectstore://bucket/fake-document.pdf",
        "media_type": "application/pdf",
        "size": 1024,
        "sha256": "b" * 64,
    }

    state = {
        "input": {
            "source": "upload",
            "mode": "acquire_and_ingest",
            "collection_id": "00000000-0000-0000-0000-000000000001",
            "upload_blob": upload_blob,
            "metadata_obj": {"file_name": "fake-document.pdf"},
        },
        "context": _tool_context(
            tenant_id="00000000-0000-0000-0000-000000000001",
            trace_id="00000000-0000-0000-0000-000000000002",
            invocation_id="inv-dedup-no-ref",
            ingestion_run_id="00000000-0000-0000-0000-000000000004",
        ),
    }

    # Mock dedup_node to return duplicate status WITHOUT existing_document_ref
    def mock_dedup_node(state):
        """Mock dedup that returns duplicate status but no existing ref."""
        return {
            "dedup_status": "duplicate",
            "existing_document_ref": None,  # MVP implementation
        }

    with patch.object(utg_module, "dedup_node", side_effect=mock_dedup_node):
        graph = utg_module.build_universal_ingestion_graph()
        result = graph.invoke(state)

    # Verify dedup_status was set to duplicate
    assert result.get("dedup_status") == "duplicate"

    output = result.get("output")
    assert output is not None

    # Verify decision is "duplicate"
    assert output["decision"] == "duplicate"

    # P1 Fix: Verify document_id is None (not a phantom new ID from normalized_document)
    assert output["document_id"] is None
