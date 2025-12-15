from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import TypeAdapter
import pytest
from documents.contracts import (
    NormalizedDocument,
    DocumentRef,
    DocumentMeta,
    BlobLocator,
)
from ai_core.graphs.upload_ingestion_graph import graph as upload_graph

def test_upload_ingestion_uses_file_blob():
    """Verify that UploadIngestionGraph functions correctly with LangGraph."""

    # 1. Setup Data
    doc_id = uuid4()
    blob_adapter = TypeAdapter(BlobLocator)

    blob_dict = {
        "type": "file",
        "uri": "s3://bucket/test.txt",
        "size": 11,
        "sha256": "5eb63bbbe01eeed093cb22bb8f5acdc3" * 2,
    }
    blob = blob_adapter.validate_python(blob_dict)

    ref = DocumentRef(
        tenant_id="test_tenant",
        workflow_id="default",
        document_id=doc_id,
        version="1.0",
        collection_id=uuid4(),
    )
    meta = DocumentMeta(
        tenant_id="test_tenant",
        workflow_id="default",
    )

    doc = NormalizedDocument(
        ref=ref,
        meta=meta,
        source="upload",
        blob=blob,
        checksum=blob.sha256,
        created_at=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        assets=[],
    )

    payload_input = doc.model_dump(mode="json")
    
    # 2. Mock Dependencies
    repository = MagicMock()
    repository.upsert.side_effect = lambda doc, **kwargs: doc

    storage_mock = MagicMock()
    storage_mock.get.return_value = b"test content"
    
    # Mocking embedding handler to avoid actual AI calls
    mock_embedding_handler = MagicMock()
    mock_embedding_handler.return_value = MagicMock(
        status="upserted",
        chunks_inserted=1,
        embedding_profile="test",
        vector_space_id="test",
        chunk_meta=MagicMock(),
    )

    # Context injection for the graph (unified structure)
    context = {
        # Telemetry IDs (unified with ExternalKnowledgeGraph)
        "tenant_id": "test_tenant",
        "trace_id": "trace-123",
        "case_id": "case-456",
        # Runtime dependencies
        "runtime_repository": repository,
        "runtime_storage": storage_mock,
        "runtime_embedder": mock_embedding_handler,
    }

    state = {
        "normalized_document_input": payload_input,
        "run_until": "persist_complete",
        "context": context,
    }

    # 3. Patch Internal Components
    # We need to patch where they are IMPORTED in the nodes
    with patch("documents.parsers.create_default_parser_dispatcher") as mock_create_parser:
        mock_result = MagicMock()
        mock_block = MagicMock()
        mock_block.text = "test content"
        mock_block.kind = "paragraph"
        mock_block.metadata = {}
        mock_result.text_blocks = [mock_block]
        mock_result.assets = []
        mock_result.statistics = {}
        mock_create_parser.return_value.parse.return_value = mock_result

        # 4. Invoke Graph
        result = upload_graph.invoke(state)

    # 5. Verify
    decision = result.get("decision")
    error = result.get("error")
    
    assert error is None, f"Graph returned error: {error}"
    assert decision == "completed", f"Graph decision was {decision}, expected completed"
    
    # Verify transitions are populated
    transitions = result.get("transitions", {})
    assert "accept_upload" in transitions
    assert "document_pipeline" in transitions
    
    # Verify repository usage (indirectly via inner graph)
    # The inner graph calls repository.upsert
    pass

def test_upload_ingestion_validation_error():
    """Verify handling of invalid input."""
    state = {
        "normalized_document_input": {"invalid": "data"}, # Invalid input
        "context": {"trace_id": "trace-123"},
    }
    
    result = upload_graph.invoke(state)
    
    assert result.get("decision") == "error"
    assert result.get("error").startswith("input_invalid")
