"""Integration tests for chunk persistence."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from documents.contracts import (
    DocumentMeta,
    DocumentRef,
    InlineBlob,
    NormalizedDocument,
    Asset,
)
from documents.pipeline import (
    DocumentPipelineConfig,
    DocumentProcessingContext,
    DocumentProcessingMetadata,
    ParsedResult,
    ParsedTextBlock,
)
from documents.processing_graph import (
    DocumentProcessingPhase,
    DocumentProcessingState,
    build_document_processing_graph,
)
from documents.repository import DocumentsRepository

def test_chunk_persistence():
    """Test that chunks are persisted to the repository as assets."""
    # Mocks
    parser = MagicMock()
    repository = MagicMock(spec=DocumentsRepository)
    repository.upsert.side_effect = lambda doc, **kwargs: doc
    storage = MagicMock()
    captioner = MagicMock()
    chunker = MagicMock()
    embedder = MagicMock()

    # Stub embedder to avoid error
    embedder.return_value = {"written": 0}

    # Setup parser mock
    parsed_result = ParsedResult(
        text_blocks=[
            ParsedTextBlock(
                kind="paragraph",
                text="Chunk Content 1",
                page_index=0,
                section_path=[],
            )
        ],
        assets=[],
        statistics={},
    )
    parser.parse.return_value = parsed_result

    # Setup mock captioner to avoid issues in caption phase
    captioner.caption.return_value = ([], {})

    # Setup chunker mock to return 2 chunks
    chunk1_content = "Chunk Content 1"
    chunk2_content = "Chunk Content 2"
    chunker.chunk.return_value = (
        [
            {"content": chunk1_content, "meta": {"index": 0}},
            {"content": chunk2_content, "meta": {"index": 1}},
        ],
        {"chunk_count": 2},
    )

    # Build Graph
    graph = build_document_processing_graph(
        parser=parser,
        repository=repository,
        storage=storage,
        captioner=captioner,
        chunker=chunker,
        embedder=embedder,
        propagate_errors=True,
    )

    # Setup Test Document
    doc_ref = DocumentRef(
        document_id="00000000-0000-0000-0000-000000000099",
        tenant_id="dev",
        workflow_id="test_persistence",
        collection_id=None,
        version="1.0",
    )
    doc_meta = DocumentMeta(tenant_id="dev", workflow_id="test_persistence")
    blob = InlineBlob(
        type="inline",
        media_type="text/plain",
        base64="dGVzdA==", # test
        sha256="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        size=4,
    )
    document = NormalizedDocument(
        ref=doc_ref,
        meta=doc_meta,
        blob=blob,
        checksum=blob.sha256,
        created_at=datetime.now(timezone.utc),
        source="upload",
    )

    context_meta = DocumentProcessingMetadata.from_document(document)
    context = DocumentProcessingContext(metadata=context_meta)
    config = DocumentPipelineConfig(enable_embedding=True)

    # Run Graph
    state = DocumentProcessingState(
        document=document,
        config=config,
        context=context,
        run_until=DocumentProcessingPhase.FULL,
    )

    graph.invoke(state)

    # Assertions
    # repository.add_asset should be called twice
    assert repository.add_asset.call_count == 2, f"Expected 2 calls, got {repository.add_asset.call_count}"

    # Verify content of calls
    calls = repository.add_asset.call_args_list
    
    # Check first chunk
    asset1 = calls[0][0][0] # first arg of first call
    # Handle auto-unwrapping if mock stores args weirdly, but usually [0][0] is correct
    assert isinstance(asset1, Asset)
    assert asset1.asset_kind == "chunk"
    assert asset1.text_description == chunk1_content
    # Blob check (optional)
    
    # Check second chunk
    asset2 = calls[1][0][0]
    assert asset2.text_description == chunk2_content
