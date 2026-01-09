"""Integration tests for HybridChunker with universal_ingestion_graph.

Tests the integration of HybridChunker into the document processing pipeline,
verifying that chunks are created correctly with quality metrics.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from documents.pipeline import (
    ParsedResult,
    ParsedTextBlock,
    DocumentProcessingContext,
    DocumentProcessingMetadata,
)
from ai_core.rag.chunking import HybridChunker, ChunkerConfig, ChunkerMode
from datetime import datetime, timezone


@pytest.fixture
def hybrid_chunker_config():
    """HybridChunker configuration for testing."""
    return ChunkerConfig(
        mode=ChunkerMode.LATE,
        late_chunk_model="oai-embed-large",
        late_chunk_max_tokens=8000,
        enable_quality_metrics=False,  # Disable for fast tests
        max_chunk_tokens=450,
        overlap_tokens=80,
        similarity_threshold=0.7,
    )


@pytest.fixture
def hybrid_chunker_with_quality_config():
    """HybridChunker configuration with quality metrics enabled."""
    return ChunkerConfig(
        mode=ChunkerMode.LATE,
        late_chunk_model="oai-embed-large",
        late_chunk_max_tokens=8000,
        enable_quality_metrics=True,
        quality_model="gpt-5-nano",
        max_chunk_tokens=450,
        overlap_tokens=80,
        similarity_threshold=0.7,
    )


@pytest.fixture
def sample_parsed_result():
    """Sample ParsedResult for integration testing."""
    blocks = [
        ParsedTextBlock(
            kind="heading",
            text="Introduction to AI",
            section_path=("Introduction",),
            page_index=None,
        ),
        ParsedTextBlock(
            kind="paragraph",
            text="Artificial intelligence is transforming industries. " * 20,
            section_path=("Introduction",),
            page_index=None,
        ),
        ParsedTextBlock(
            kind="heading",
            text="Machine Learning",
            section_path=("Machine Learning",),
            page_index=None,
        ),
        ParsedTextBlock(
            kind="paragraph",
            text="Machine learning algorithms enable computers to learn from data. "
            * 15,
            section_path=("Machine Learning",),
            page_index=None,
        ),
    ]

    return ParsedResult(
        text_blocks=blocks,
        assets=[],
        statistics={"block.count": len(blocks)},
    )


@pytest.fixture
def sample_processing_context():
    """Sample DocumentProcessingContext for integration testing."""
    metadata = DocumentProcessingMetadata(
        tenant_id="integration-test-tenant",
        document_id=uuid4(),
        workflow_id="integration-test-workflow",
        created_at=datetime.now(timezone.utc),
        trace_id="integration-test-trace-001",
        span_id="integration-test-span-001",
    )

    return DocumentProcessingContext(
        metadata=metadata,
        trace_id="integration-test-trace-001",
        span_id="integration-test-span-001",
    )


@pytest.fixture
def stub_embedding_client():
    """Stub embedding client for integration tests."""

    class StubEmbeddingClient:
        def __init__(self, dimension: int = 1536):
            self.dimension = dimension

        def embed(self, texts: list[str], model: str = None) -> MagicMock:
            """Return fake embeddings."""
            _ = model
            embeddings = []
            for text in texts:
                magnitude = hash(text) % 100 / 100.0
                embedding = [magnitude] + [0.0] * (self.dimension - 1)
                embeddings.append(embedding)

            response = MagicMock()
            response.vectors = embeddings
            return response

    return StubEmbeddingClient()


class TestHybridChunkerIntegration:
    """Integration tests for HybridChunker in document processing pipeline."""

    def test_hybrid_chunker_basic_integration(
        self,
        hybrid_chunker_config,
        sample_parsed_result,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test basic integration: HybridChunker produces chunks from parsed document."""
        chunker = HybridChunker(hybrid_chunker_config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            # Simulate document processing pipeline calling chunker
            chunks, stats = chunker.chunk(
                document=None,  # Not used by HybridChunker
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=None,  # Pipeline config (unused for now)
            )

        # Verify chunks were created
        assert len(chunks) > 0, "HybridChunker should produce at least one chunk"

        # Verify chunk structure (DocumentChunker protocol)
        chunk = chunks[0]
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "parent_ref" in chunk
        assert "metadata" in chunk

        # Verify metadata contains chunker info
        assert chunk["metadata"]["chunker"] == "late"
        assert chunk["metadata"]["model"] == "oai-embed-large"

        # Verify statistics
        assert stats["chunk.count"] == len(chunks)
        assert stats["chunker.mode"] == "late"
        assert stats["chunker.version"] == "hybrid-v1"

    def test_hybrid_chunker_chunk_id_stability(
        self,
        hybrid_chunker_config,
        sample_parsed_result,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test that chunk IDs are stable (deterministic) across runs."""
        chunker = HybridChunker(hybrid_chunker_config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            # Run chunking twice
            chunks1, _ = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=None,
            )

            chunks2, _ = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=None,
            )

        # Chunk IDs should be identical
        chunk_ids1 = [c["chunk_id"] for c in chunks1]
        chunk_ids2 = [c["chunk_id"] for c in chunks2]

        assert chunk_ids1 == chunk_ids2, "Chunk IDs should be deterministic"

    def test_hybrid_chunker_with_quality_metrics(
        self,
        hybrid_chunker_with_quality_config,
        sample_parsed_result,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test integration with quality metrics enabled."""
        chunker = HybridChunker(hybrid_chunker_with_quality_config)

        # Mock both embedding client and LLM for quality evaluation
        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_embed:
            with patch("litellm.completion") as mock_llm:
                mock_embed.return_value = stub_embedding_client

                # Mock LLM quality evaluation response
                mock_llm.return_value = MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(
                                content='{"coherence": 85, "completeness": 90, "reference_resolution": 80, "redundancy": 95, "reasoning": "Test evaluation"}'
                            )
                        )
                    ]
                )

                chunks, stats = chunker.chunk(
                    document=None,
                    parsed=sample_parsed_result,
                    context=sample_processing_context,
                    config=None,
                )

        # Verify chunks have quality metadata
        assert len(chunks) > 0
        chunk = chunks[0]
        assert "quality" in chunk["metadata"]
        assert "coherence" in chunk["metadata"]["quality"]
        assert "completeness" in chunk["metadata"]["quality"]
        assert "overall" in chunk["metadata"]["quality"]

        # Verify statistics include quality metrics
        assert "quality.mean_coherence" in stats
        assert "quality.mean_completeness" in stats
        assert "quality.mean_overall" in stats

    def test_hybrid_chunker_agentic_fallback(
        self,
        sample_parsed_result,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test that Agentic mode falls back to Late chunking."""
        config = ChunkerConfig(
            mode=ChunkerMode.AGENTIC,  # Request agentic mode
            late_chunk_model="oai-embed-large",
            late_chunk_max_tokens=8000,
            enable_quality_metrics=False,
            max_chunk_tokens=450,
        )

        chunker = HybridChunker(config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks, stats = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=None,
            )

        # Should use agentic mode (with internal fallback to Late)
        assert len(chunks) > 0
        assert stats["chunker.mode"] == "agentic"
        # Note: AgenticChunker handles fallback internally (MVP implementation)

    def test_hybrid_chunker_large_document_fallback(
        self,
        hybrid_chunker_config,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test fallback behavior when document exceeds max_tokens."""
        # Create large document that exceeds max_tokens
        large_blocks = [
            ParsedTextBlock(
                kind="paragraph",
                text="This is a very long paragraph that will exceed token limits. "
                * 500,
                section_path=("Section 1",),
                page_index=None,
            ),
            ParsedTextBlock(
                kind="paragraph",
                text="Another very long paragraph to ensure we exceed limits. " * 500,
                section_path=("Section 2",),
                page_index=None,
            ),
        ]

        large_parsed = ParsedResult(
            text_blocks=large_blocks,
            assets=[],
            statistics={"block.count": len(large_blocks)},
        )

        # Configure with very small max_tokens to trigger fallback
        small_config = ChunkerConfig(
            mode=ChunkerMode.LATE,
            late_chunk_model="oai-embed-large",
            late_chunk_max_tokens=100,  # Very small to force fallback
            enable_quality_metrics=False,
            max_chunk_tokens=450,
        )

        chunker = HybridChunker(small_config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks, stats = chunker.chunk(
                document=None,
                parsed=large_parsed,
                context=sample_processing_context,
                config=None,
            )

        # Should use fallback chunking
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["metadata"]["chunker"] == "late-fallback"

    def test_hybrid_chunker_multiple_chunks_with_overlap(
        self,
        hybrid_chunker_config,
        sample_parsed_result,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test that small target_tokens produces multiple chunks with overlap."""
        # Configure with small target to force multiple chunks
        small_target_config = ChunkerConfig(
            mode=ChunkerMode.LATE,
            late_chunk_model="oai-embed-large",
            late_chunk_max_tokens=8000,
            enable_quality_metrics=False,
            max_chunk_tokens=100,  # Small target
            overlap_tokens=20,
        )

        chunker = HybridChunker(small_target_config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks, stats = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=None,
            )

        # Should produce multiple chunks
        assert len(chunks) > 1, "Small target_tokens should produce multiple chunks"

        # Verify sentence ranges show overlap or adjacency
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                curr_range = chunks[i]["metadata"]["sentence_range"]
                next_range = chunks[i + 1]["metadata"]["sentence_range"]

                # Next chunk should start before or at current chunk end (overlap or adjacent)
                assert next_range[0] <= curr_range[1]

    def test_hybrid_chunker_empty_document_handling(
        self,
        hybrid_chunker_config,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test that HybridChunker handles empty/minimal documents gracefully."""
        minimal_blocks = [
            ParsedTextBlock(
                kind="heading",
                text="Title",
                section_path=("Title",),
                page_index=None,
            ),
        ]

        minimal_parsed = ParsedResult(
            text_blocks=minimal_blocks,
            assets=[],
            statistics={"block.count": 1},
        )

        chunker = HybridChunker(hybrid_chunker_config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks, stats = chunker.chunk(
                document=None,
                parsed=minimal_parsed,
                context=sample_processing_context,
                config=None,
            )

        # Should handle minimal document without crashing
        assert isinstance(chunks, list)
        assert isinstance(stats, dict)

    def test_hybrid_chunker_protocol_compliance(
        self,
        hybrid_chunker_config,
        sample_parsed_result,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test that HybridChunker complies with DocumentChunker protocol."""
        chunker = HybridChunker(hybrid_chunker_config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            # Call with protocol signature (document, parsed, context, config)
            result = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=None,
            )

        # Verify protocol compliance: returns (chunks, statistics) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        chunks, stats = result
        assert isinstance(chunks, (list, tuple))
        assert isinstance(stats, dict)

        # Verify chunk structure matches protocol
        if len(chunks) > 0:
            chunk = chunks[0]
            required_fields = ["chunk_id", "text", "parent_ref", "metadata"]
            for field in required_fields:
                assert field in chunk, f"Chunk missing required field: {field}"

    def test_hybrid_chunker_error_handling(
        self,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test HybridChunker error handling when embedding client fails (Phase 2)."""
        # Phase 2 config with embedding-based similarity enabled
        config_with_embeddings = ChunkerConfig(
            mode=ChunkerMode.LATE,
            late_chunk_model="oai-embed-large",
            late_chunk_max_tokens=8000,
            enable_quality_metrics=False,
            max_chunk_tokens=450,
            use_embedding_similarity=True,  # Enable Phase 2
        )
        chunker = HybridChunker(config_with_embeddings)

        # Mock embedding client to raise exception
        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embed.side_effect = Exception("Embedding service unavailable")
            mock_get_client.return_value = mock_client

            # Phase 2 should fallback to Jaccard on embedding failure (no exception)
            chunks, stats = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=None,
            )

            # Should succeed with fallback chunking
            assert len(chunks) > 0
            assert stats["chunker.mode"] == "late"

    def test_hybrid_chunker_quality_evaluation_error_handling(
        self,
        hybrid_chunker_with_quality_config,
        sample_parsed_result,
        sample_processing_context,
        stub_embedding_client,
    ):
        """Test that quality evaluation errors don't break chunking."""
        chunker = HybridChunker(hybrid_chunker_with_quality_config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_embed:
            with patch("litellm.completion") as mock_llm:
                mock_embed.return_value = stub_embedding_client

                # Mock LLM to raise exception
                mock_llm.side_effect = Exception("LLM service unavailable")

                # Should still return chunks (quality evaluation is optional)
                chunks, stats = chunker.chunk(
                    document=None,
                    parsed=sample_parsed_result,
                    context=sample_processing_context,
                    config=None,
                )

        # Chunks should be created despite quality eval failure
        assert len(chunks) > 0

        # Quality metadata should be present but with default scores (all 0.0)
        chunk = chunks[0]
        assert "quality" in chunk["metadata"]
        assert chunk["metadata"]["quality"]["overall"] == 0.0
        assert chunk["metadata"]["quality"]["coherence"] == 0.0
        assert (
            chunk["metadata"]["quality"]["reasoning"] == "Evaluation skipped or failed"
        )

        # Quality stats should be present but with 0.0 values
        assert "quality.mean_coherence" in stats
        assert stats["quality.mean_overall"] == 0.0
