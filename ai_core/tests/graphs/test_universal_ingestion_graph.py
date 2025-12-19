import pytest
import sys
from unittest.mock import MagicMock, patch
from documents.contracts import FileBlob

# Remove top-level import to allow pre-import patching
# from ai_core.graphs.technical.universal_ingestion_graph import ...


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
    with patch("documents.processing_graph.build_document_processing_graph") as mock:
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
            "collection_id": "col-1",
            "upload_blob": None,  # Invalid
            "metadata_obj": {},
            "normalized_document": None,
        },
        "context": {
            "tenant_id": "tenant-1",
            "trace_id": "trace-1",
            "case_id": "case-1",
        },
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
        "context": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "00000000-0000-0000-0000-000000000001",
            "case_id": "00000000-0000-0000-0000-000000000001",
            "ingestion_run_id": "00000000-0000-0000-0000-000000000001",
        },
    }

    result = graph.invoke(state)
    output = result["output"]

    if output["decision"] == "error":
        print(f"DEBUG FAILURE REASON: {output['reason']}")

    assert output["decision"] == "ingested"
    assert output["decision"] == "ingested"
    assert output["document_id"] == "00000000-0000-0000-0000-000000000123"

    # Roadmap compliance checks
    assert "transitions" in output
    assert output["transitions"] == [
        "validate_input",
        "normalize",
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
            "collection_id": "col-1",
            "normalized_document": {},
        },
        "context": {
            "tenant_id": "tenant-1",
            # Missing trace_id and case_id
        },
    }

    result = graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "error"
    assert "Missing required context" in output["reason"]


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
        "context": {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "trace_id": "00000000-0000-0000-0000-000000000001",
            "case_id": "00000000-0000-0000-0000-000000000001",
            "ingestion_run_id": "00000000-0000-0000-0000-000000000001",
        },
    }

    result = graph.invoke(state)
    output = result["output"]

    assert output["decision"] == "ingested"
    assert output["decision"] == "ingested"
    assert output["document_id"] is not None  # Generated in _build_normalized_document

    # Roadmap compliance checks
    assert output["transitions"] == [
        "validate_input",
        "normalize",
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
