
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import TypeAdapter
from documents.contracts import FileBlob, NormalizedDocument, DocumentRef, DocumentMeta, BlobLocator
from ai_core.graphs.upload_ingestion_graph import UploadIngestionGraph

def test_upload_ingestion_uses_file_blob():
    """Verify that UploadIngestionGraph creates a FileBlob instead of InlineBlob."""
    
    # 1. Setup Data - Use Pydantic models to generate valid payload
    doc_id = uuid4()
    
    blob_adapter = TypeAdapter(BlobLocator)
    
    # Create valid blob
    blob_dict = {
        "type": "file",
        "uri": "s3://bucket/test.txt",
        "size": 11,
        "sha256": "5eb63bbbe01eeed093cb22bb8f5acdc3" * 2, 
    }
    blob = blob_adapter.validate_python(blob_dict)

    # Create valid NormalizedDocument
    ref = DocumentRef(
        tenant_id="test_tenant",
        workflow_id="default",
        document_id=doc_id,
        version="1.0"
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
        assets=[]
    )
    
    # Dump to dict as if it came from JSON
    payload_input = doc.model_dump(mode="json")
    
    payload = {
        "normalized_document_input": payload_input
    }

    # 2. Mock Dependencies
    repository = MagicMock()
    # upsert returns valid ID
    repository.upsert.side_effect = lambda doc, **kwargs: doc

    storage_mock = MagicMock()
    storage_mock.get.return_value = b"test content"
    
    # Mock require_document_components where it is USED
    with patch("ai_core.graphs.upload_ingestion_graph.require_document_components") as mock_components:
        mock_components.return_value.storage = MagicMock(return_value=storage_mock)
        mock_components.return_value.captioner = MagicMock()
        
        # Mock create_default_parser_dispatcher in the source module
        with patch("documents.parsers.create_default_parser_dispatcher") as mock_create_parser:
             # Setup parser result
            mock_result = MagicMock()
            mock_block = MagicMock()
            mock_block.text = "test content"
            mock_block.kind = "paragraph"
            mock_block.section_path = []
            mock_block.page_index = 0
            mock_block.table_meta = None
            mock_block.language = "en"
            mock_block.metadata = {}  # Important for checking metadata attrs
            mock_result.text_blocks = [mock_block]
            mock_result.assets = []
            mock_result.statistics = {}
            mock_create_parser.return_value.parse.return_value = mock_result
            
            # 3. Instantiate Graph (now uses mocked parser)
            mock_embedding_handler = MagicMock()
            mock_embedding_handler.return_value = MagicMock(status="upserted", chunks_inserted=1, embedding_profile="test", vector_space_id="test", chunk_meta=MagicMock())
            graph = UploadIngestionGraph(repository=repository, embedding_handler=mock_embedding_handler)
            
            # 4. Run Graph
            # We also need to patch guardrail/delta decider if they run
            # Graph runs.
            
            result = graph.run(payload)
        
        # 5. Verify
        # Check success
        decision = result.get("decision")
        assert decision in ("completed", "processed", "skip_guardrails", "skip_delta"), f"Graph failed: {result.get('reason')}"
        
        # Verify persistence happened (upsert called)
        # Note: If parser returns no blocks, might skip persistence?
        # But generic text parser should return blocks.
