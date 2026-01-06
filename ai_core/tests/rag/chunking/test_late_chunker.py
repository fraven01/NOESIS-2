"""Unit tests for LateChunker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_core.rag.chunking.late_chunker import LateChunker, LateBoundary, SentenceWindow


class TestLateChunker:
    """Test LateChunker implementation."""

    def test_late_chunker_basic(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test basic Late Chunking produces chunks."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=450,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        # Should produce at least one chunk
        assert len(chunks) > 0

        # Check chunk structure
        chunk = chunks[0]
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "parent_ref" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["chunker"] == "late"

    def test_late_chunker_respects_target_tokens(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test that Late Chunker respects target_tokens."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=100,  # Small target
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        # With small target_tokens, should produce multiple chunks
        assert len(chunks) > 1

    def test_late_chunker_preserves_metadata(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test that Late Chunker preserves metadata in chunks."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=450,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        # Check metadata preservation
        for chunk in chunks:
            assert "metadata" in chunk
            assert "chunker" in chunk["metadata"]
            assert chunk["metadata"]["chunker"] == "late"
            assert "sentence_range" in chunk["metadata"]
            assert "model" in chunk["metadata"]
            assert chunk["metadata"]["model"] == "oai-embed-large"

    def test_late_chunker_stable_chunk_ids(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test that chunk IDs are stable (deterministic)."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=450,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            # Run chunking twice
            chunks1 = chunker.chunk(sample_parsed_result, sample_processing_context)
            chunks2 = chunker.chunk(sample_parsed_result, sample_processing_context)

        # Chunk IDs should be identical (deterministic)
        chunk_ids1 = [c["chunk_id"] for c in chunks1]
        chunk_ids2 = [c["chunk_id"] for c in chunks2]

        assert chunk_ids1 == chunk_ids2

    def test_late_chunker_fallback_for_long_documents(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test fallback behavior when document exceeds max_tokens."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=100,  # Very small limit to trigger fallback
            target_tokens=450,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        # Should still produce chunks (via fallback)
        assert len(chunks) > 0

        # Fallback chunks should have "late-fallback" chunker
        for chunk in chunks:
            assert chunk["metadata"]["chunker"] == "late-fallback"

    def test_late_chunker_handles_empty_blocks(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Test that Late Chunker handles empty text blocks gracefully."""
        from documents.pipeline import ParsedResult, ParsedTextBlock

        parsed = ParsedResult(
            text_blocks=[
                ParsedTextBlock(
                    kind="heading", text="A", section_path=(), page_index=None
                ),
                ParsedTextBlock(
                    kind="paragraph", text="B", section_path=(), page_index=None
                ),
            ],
            assets=[],
            statistics={"block.count": 2},
        )

        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=450,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(parsed, sample_processing_context)

        # Should handle empty blocks without crashing
        # Might produce 0 chunks if all blocks are empty
        assert isinstance(chunks, list)

    def test_late_chunker_overlap(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test that Late Chunker creates overlap between chunks."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=100,  # Small target to create multiple chunks
            overlap_tokens=20,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        # With overlap, adjacent chunks should share some content
        if len(chunks) > 1:
            # Check that sentence ranges overlap
            for i in range(len(chunks) - 1):
                curr_range = chunks[i]["metadata"]["sentence_range"]
                next_range = chunks[i + 1]["metadata"]["sentence_range"]

                # Next chunk should start before current chunk ends (overlap)
                # OR immediately after (no overlap if not enough sentences)
                assert next_range[0] <= curr_range[1]

    def test_late_chunker_sets_kind_metadata(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Adaptive chunker sets metadata.kind for text chunks."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=450,
            adaptive_enabled=True,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        assert len(chunks) > 0
        assert all(chunk["metadata"]["kind"] == "text" for chunk in chunks)

    def test_late_chunker_asset_chunks_enabled(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Asset chunks are emitted when enabled."""
        from documents.parsers import ParsedAssetWithMeta, ParsedResult, ParsedTextBlock

        parsed = ParsedResult(
            text_blocks=[
                ParsedTextBlock(
                    kind="paragraph",
                    text="Asset-aware text block.",
                    section_path=("Assets",),
                    page_index=None,
                )
            ],
            assets=[
                ParsedAssetWithMeta(
                    media_type="image/png",
                    file_uri="file:///tmp/asset.png",
                    page_index=2,
                    metadata={
                        "locator": "asset-1",
                        "caption_candidates": [("alt_text", "Asset caption")],
                    },
                )
            ],
            statistics={"block.count": 1},
        )

        chunker = LateChunker(
            model="oai-embed-large",
            adaptive_enabled=True,
            asset_chunks_enabled=True,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(parsed, sample_processing_context)

        asset_chunks = [
            chunk for chunk in chunks if chunk["metadata"].get("kind") == "asset"
        ]
        assert asset_chunks, "Expected asset chunk when assets are present."
        asset_chunk = asset_chunks[0]
        assert asset_chunk["parent_ref"]
        assert asset_chunk["page_index"] == 2

    def test_late_chunker_asset_chunks_disabled(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Asset chunks are skipped when disabled."""
        from documents.parsers import ParsedAssetWithMeta, ParsedResult, ParsedTextBlock

        parsed = ParsedResult(
            text_blocks=[
                ParsedTextBlock(
                    kind="paragraph",
                    text="Asset-aware text block.",
                    section_path=("Assets",),
                    page_index=None,
                )
            ],
            assets=[
                ParsedAssetWithMeta(
                    media_type="image/png",
                    file_uri="file:///tmp/asset.png",
                    metadata={"locator": "asset-1"},
                )
            ],
            statistics={"block.count": 1},
        )

        chunker = LateChunker(
            model="oai-embed-large",
            adaptive_enabled=True,
            asset_chunks_enabled=False,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(parsed, sample_processing_context)

        assert not any(chunk["metadata"].get("kind") == "asset" for chunk in chunks)

    def test_late_chunker_chunk_ids_include_parent_ref(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Identical text in different sections yields distinct chunk IDs."""
        from documents.parsers import ParsedResult, ParsedTextBlock

        parsed = ParsedResult(
            text_blocks=[
                ParsedTextBlock(
                    kind="paragraph",
                    text="Same text.",
                    section_path=("Section A",),
                    page_index=None,
                ),
                ParsedTextBlock(
                    kind="paragraph",
                    text="Same text.",
                    section_path=("Section B",),
                    page_index=None,
                ),
            ],
            assets=[],
            statistics={"block.count": 2},
        )

        chunker = LateChunker(
            model="oai-embed-large",
            target_tokens=1000,
            adaptive_enabled=True,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(parsed, sample_processing_context)

        assert len(chunks) == 2
        assert chunks[0]["text"] == chunks[1]["text"]
        assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]

    def test_late_chunker_legacy_path_unchanged(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Legacy mode keeps metadata shape stable."""
        chunker = LateChunker(
            model="oai-embed-large",
            adaptive_enabled=False,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        assert len(chunks) > 0
        assert "kind" not in chunks[0]["metadata"]


class TestLateChunkerPhase2:
    """Test Phase 2 SOTA embedding-based boundary detection features."""

    def test_create_sentence_windows(self):
        """Test sliding window creation for sentences."""
        chunker = LateChunker(
            model="oai-embed-large",
            window_size=3,
        )

        sentences = ["A", "B", "C", "D", "E"]
        windows = chunker._create_sentence_windows(sentences)

        # Should create windows for each sentence position
        assert len(windows) == 5

        # Check first window (A B C)
        assert windows[0].start_idx == 0
        assert windows[0].end_idx == 3
        assert windows[0].text == "A B C"
        assert windows[0].embedding is None

        # Check middle window (C D E)
        assert windows[2].start_idx == 2
        assert windows[2].end_idx == 5
        assert windows[2].text == "C D E"

        # Check last window (E)
        assert windows[4].start_idx == 4
        assert windows[4].end_idx == 5
        assert windows[4].text == "E"

    def test_create_sentence_windows_empty(self):
        """Test window creation with empty sentence list."""
        chunker = LateChunker(model="oai-embed-large", window_size=3)

        windows = chunker._create_sentence_windows([])

        assert windows == []

    def test_create_sentence_windows_single_sentence(self):
        """Test window creation with single sentence."""
        chunker = LateChunker(model="oai-embed-large", window_size=3)

        windows = chunker._create_sentence_windows(["Only sentence"])

        assert len(windows) == 1
        assert windows[0].start_idx == 0
        assert windows[0].end_idx == 1
        assert windows[0].text == "Only sentence"

    def test_embed_windows_batch(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Test batch embedding of sentence windows."""
        chunker = LateChunker(
            model="oai-embed-large",
            batch_size=2,  # Small batch for testing
        )

        # Create windows without embeddings
        windows = [
            SentenceWindow(0, 3, "A B C", embedding=None),
            SentenceWindow(1, 4, "B C D", embedding=None),
            SentenceWindow(2, 5, "C D E", embedding=None),
        ]

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            embedded_windows = chunker._embed_windows_batch(
                windows, sample_processing_context
            )

        # Should return same number of windows
        assert len(embedded_windows) == 3

        # All windows should have embeddings
        for window in embedded_windows:
            assert window.embedding is not None
            assert isinstance(window.embedding, list)
            assert len(window.embedding) > 0  # Should have embedding dimension

        # Text and indices should be preserved
        assert embedded_windows[0].text == "A B C"
        assert embedded_windows[1].start_idx == 1
        assert embedded_windows[2].end_idx == 5

    def test_embed_windows_batch_empty(
        self,
        sample_processing_context,
    ):
        """Test batch embedding with empty window list."""
        chunker = LateChunker(model="oai-embed-large")

        embedded_windows = chunker._embed_windows_batch([], sample_processing_context)

        assert embedded_windows == []

    def test_embed_windows_batch_failure_raises(
        self,
        sample_processing_context,
    ):
        """Test that batch embedding failures raise exceptions for fallback."""
        chunker = LateChunker(model="oai-embed-large")

        windows = [
            SentenceWindow(0, 1, "Test", embedding=None),
        ]

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embed.side_effect = Exception("Embedding service down")
            mock_get_client.return_value = mock_client

            # Should raise exception (for graceful fallback in caller)
            with pytest.raises(Exception, match="Embedding service down"):
                chunker._embed_windows_batch(windows, sample_processing_context)

    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        chunker = LateChunker(model="oai-embed-large")

        # Identical vectors -> similarity = 1.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = chunker._compute_cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

        # Orthogonal vectors -> similarity = 0.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = chunker._compute_cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

        # Opposite vectors -> similarity = -1.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = chunker._compute_cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001

        # Similar vectors -> similarity > 0.5
        vec1 = [1.0, 1.0, 0.0]
        vec2 = [1.0, 0.5, 0.0]
        similarity = chunker._compute_cosine_similarity(vec1, vec2)
        assert similarity > 0.5

    def test_compute_cosine_similarity_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        chunker = LateChunker(model="oai-embed-large")

        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        # Should return 0.0 for zero vectors (not crash)
        similarity = chunker._compute_cosine_similarity(vec1, vec2)
        assert similarity == 0.0

    def test_compute_cosine_similarity_dimension_mismatch(self):
        """Test cosine similarity with mismatched dimensions."""
        chunker = LateChunker(model="oai-embed-large")

        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        # Should raise ValueError
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            chunker._compute_cosine_similarity(vec1, vec2)

    def test_detect_boundaries_embedding(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Test embedding-based boundary detection."""
        chunker = LateChunker(
            model="oai-embed-large",
            target_tokens=50,  # Small target for multiple boundaries
            similarity_threshold=0.7,
            use_embedding_similarity=True,
        )

        # Create sentences that would trigger boundaries
        sentences = [
            "This is about AI.",  # Topic 1
            "Machine learning is powerful.",
            "Neural networks are complex.",
            "Now let's talk about cooking.",  # Topic 2 (should be boundary)
            "Recipes are important.",
            "Baking requires precision.",
        ]

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            boundaries = chunker._detect_boundaries_embedding(
                sentences, sample_processing_context
            )

        # Should detect at least one boundary
        assert len(boundaries) > 0

        # Boundaries should have valid structure
        for boundary in boundaries:
            assert isinstance(boundary, LateBoundary)
            assert boundary.start_idx >= 0
            assert boundary.end_idx > boundary.start_idx
            assert 0.0 <= boundary.similarity_score <= 1.0

    def test_detect_boundaries_embedding_fallback_on_error(
        self,
        sample_processing_context,
    ):
        """Test that embedding boundary detection falls back to Jaccard on error."""
        chunker = LateChunker(
            model="oai-embed-large",
            target_tokens=50,
            use_embedding_similarity=True,
        )

        sentences = ["A", "B", "C", "D", "E"]

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.embed.side_effect = Exception("Embedding failed")
            mock_get_client.return_value = mock_client

            # Should fallback to Jaccard (not crash)
            boundaries = chunker._detect_boundaries_embedding(
                sentences, sample_processing_context
            )

        # Should still produce boundaries (via Jaccard fallback)
        assert isinstance(boundaries, list)

    def test_phase2_end_to_end_with_embedding_similarity(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test end-to-end Phase 2 chunking with embedding-based similarity."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=100,
            use_embedding_similarity=True,  # Enable Phase 2
            window_size=3,
            batch_size=16,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks = chunker.chunk(sample_parsed_result, sample_processing_context)

        # Should produce chunks
        assert len(chunks) > 0

        # Chunks should have correct metadata
        for chunk in chunks:
            assert chunk["metadata"]["chunker"] == "late"
            assert chunk["metadata"]["model"] == "oai-embed-large"
            assert "similarity_score" in chunk["metadata"]

    def test_content_based_chunk_ids(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
    ):
        """Test Phase 2 content-based chunk IDs (SHA256)."""
        chunker = LateChunker(
            model="oai-embed-large",
            max_tokens=8000,
            target_tokens=450,
            use_content_based_ids=True,  # Enable SHA256 IDs
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            # Run chunking twice
            chunks1 = chunker.chunk(sample_parsed_result, sample_processing_context)
            chunks2 = chunker.chunk(sample_parsed_result, sample_processing_context)

        # Content-based IDs should be identical (deterministic)
        chunk_ids1 = [c["chunk_id"] for c in chunks1]
        chunk_ids2 = [c["chunk_id"] for c in chunks2]

        assert chunk_ids1 == chunk_ids2

        # IDs should start with "sha256-"
        for chunk in chunks1:
            assert chunk["chunk_id"].startswith("sha256-")

    def test_content_based_ids_prevent_collisions_across_documents(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Test that SHA256 chunk IDs are namespaced by document_id to prevent collisions."""
        from documents.pipeline import ParsedResult, ParsedTextBlock
        from uuid import uuid4
        from dataclasses import replace

        # Same content, different document IDs
        parsed = ParsedResult(
            text_blocks=[
                ParsedTextBlock(
                    kind="paragraph",
                    text="This is identical content for testing determinism.",
                    section_path=("Test",),
                    page_index=None,
                ),
            ],
            assets=[],
            statistics={"block.count": 1},
        )

        chunker = LateChunker(
            model="oai-embed-large",
            use_content_based_ids=True,
        )

        # Create two contexts with different document_ids
        context1 = sample_processing_context

        # Create new context with different document_id
        # Use model_copy for Pydantic model, then replace for dataclass
        metadata2 = sample_processing_context.metadata.model_copy(
            update={"document_id": uuid4()}  # Different document ID
        )
        context2 = replace(sample_processing_context, metadata=metadata2)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks1 = chunker.chunk(parsed, context1)
            chunks2 = chunker.chunk(parsed, context2)

        # Same content, different document_ids -> DIFFERENT chunk IDs (collision prevention)
        assert chunks1[0]["chunk_id"] != chunks2[0]["chunk_id"]

        # Both should still be content-based (start with "sha256-")
        assert chunks1[0]["chunk_id"].startswith("sha256-")
        assert chunks2[0]["chunk_id"].startswith("sha256-")

    def test_content_based_ids_deterministic_within_document(
        self,
        stub_embedding_client,
        sample_processing_context,
    ):
        """Test that chunk IDs are deterministic when same document is processed multiple times."""
        from documents.pipeline import ParsedResult, ParsedTextBlock

        # Same content
        parsed = ParsedResult(
            text_blocks=[
                ParsedTextBlock(
                    kind="paragraph",
                    text="This content should produce the same chunk ID every time.",
                    section_path=("Test",),
                    page_index=None,
                ),
            ],
            assets=[],
            statistics={"block.count": 1},
        )

        chunker = LateChunker(
            model="oai-embed-large",
            use_content_based_ids=True,
        )

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            # Process same document twice with same context
            chunks1 = chunker.chunk(parsed, sample_processing_context)
            chunks2 = chunker.chunk(parsed, sample_processing_context)

        # Same content, same document_id -> SAME chunk ID (deterministic)
        assert chunks1[0]["chunk_id"] == chunks2[0]["chunk_id"]
        assert chunks1[0]["chunk_id"].startswith("sha256-")


class TestSentenceWindow:
    """Test SentenceWindow dataclass."""

    def test_sentence_window_creation(self):
        """Test creating SentenceWindow."""
        window = SentenceWindow(
            start_idx=0,
            end_idx=3,
            text="A B C",
            embedding=None,
        )

        assert window.start_idx == 0
        assert window.end_idx == 3
        assert window.text == "A B C"
        assert window.embedding is None

    def test_sentence_window_with_embedding(self):
        """Test SentenceWindow with embedding vector."""
        embedding = [0.1, 0.2, 0.3]
        window = SentenceWindow(
            start_idx=1,
            end_idx=4,
            text="B C D",
            embedding=embedding,
        )

        assert window.embedding == embedding
        assert len(window.embedding) == 3

    def test_sentence_window_frozen(self):
        """Test that SentenceWindow is frozen (immutable)."""
        window = SentenceWindow(
            start_idx=0,
            end_idx=3,
            text="A B C",
            embedding=None,
        )

        with pytest.raises(AttributeError):
            window.start_idx = 5  # Should fail (frozen)


class TestLateBoundary:
    """Test LateBoundary dataclass."""

    def test_late_boundary_creation(self):
        """Test creating LateBoundary."""
        boundary = LateBoundary(
            start_idx=0,
            end_idx=10,
            similarity_score=0.5,
        )

        assert boundary.start_idx == 0
        assert boundary.end_idx == 10
        assert boundary.similarity_score == 0.5

    def test_late_boundary_frozen(self):
        """Test that LateBoundary is frozen (immutable)."""
        boundary = LateBoundary(
            start_idx=0,
            end_idx=10,
            similarity_score=0.5,
        )

        with pytest.raises(AttributeError):
            boundary.start_idx = 5  # Should fail (frozen)
