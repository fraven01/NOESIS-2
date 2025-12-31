"""Unit tests for HybridChunker."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_core.rag.chunking import (
    HybridChunker,
    ChunkerMode,
    ChunkerConfig,
)


class TestHybridChunker:
    """Test HybridChunker implementation."""

    def test_hybrid_chunker_basic(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
        sample_pipeline_config,
    ):
        """Test basic HybridChunker operation."""
        config = ChunkerConfig(
            mode=ChunkerMode.LATE,
            enable_quality_metrics=False,  # Disable for basic test
        )
        chunker = HybridChunker(config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks, stats = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=sample_pipeline_config,
            )

        # Should produce chunks and stats
        assert len(chunks) > 0
        assert isinstance(stats, dict)

        # Stats should include chunk count and mode
        assert "chunk.count" in stats
        assert stats["chunk.count"] == len(chunks)
        assert "chunker.mode" in stats
        assert stats["chunker.mode"] == "late"

    def test_hybrid_chunker_with_quality_metrics(
        self,
        stub_embedding_client,
        mock_llm_quality_evaluation,
        sample_parsed_result,
        sample_processing_context,
        sample_pipeline_config,
    ):
        """Test HybridChunker with quality metrics enabled."""
        config = ChunkerConfig(
            mode=ChunkerMode.LATE,
            enable_quality_metrics=True,
        )
        chunker = HybridChunker(config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            with patch("litellm.completion", side_effect=mock_llm_quality_evaluation):
                chunks, stats = chunker.chunk(
                    document=None,
                    parsed=sample_parsed_result,
                    context=sample_processing_context,
                    config=sample_pipeline_config,
                )

        # Chunks should have quality metadata
        for chunk in chunks:
            assert "metadata" in chunk
            assert "quality" in chunk["metadata"]
            assert "overall" in chunk["metadata"]["quality"]

        # Stats should include quality metrics
        assert "quality.mean_overall" in stats
        assert "quality.mean_coherence" in stats

    def test_hybrid_chunker_late_mode(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
        sample_pipeline_config,
    ):
        """Test HybridChunker in LATE mode."""
        config = ChunkerConfig(
            mode=ChunkerMode.LATE,
            enable_quality_metrics=False,
        )
        chunker = HybridChunker(config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks, stats = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=sample_pipeline_config,
            )

        # Should use late mode
        assert stats["chunker.mode"] == "late"

        # Chunks should have late chunker metadata
        for chunk in chunks:
            assert chunk["metadata"]["chunker"] in ["late", "late-fallback"]

    def test_hybrid_chunker_agentic_fallback(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
        sample_pipeline_config,
    ):
        """Test that AGENTIC mode with internal fallback to LATE (MVP)."""
        config = ChunkerConfig(
            mode=ChunkerMode.AGENTIC,  # MVP: AgenticChunker internally falls back to Late
            enable_quality_metrics=False,
        )
        chunker = HybridChunker(config)

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            chunks, stats = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=sample_pipeline_config,
            )

        # Should use agentic mode (with internal fallback to Late handled by AgenticChunker)
        assert stats["chunker.mode"] == "agentic"
        assert len(chunks) > 0
        # Note: AgenticChunker handles fallback internally (MVP implementation)

    def test_hybrid_chunker_implements_protocol(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
        sample_pipeline_config,
    ):
        """Test that HybridChunker implements DocumentChunker protocol."""
        config = ChunkerConfig(
            mode=ChunkerMode.LATE,
            enable_quality_metrics=False,
        )
        chunker = HybridChunker(config)

        # Should have chunk method with correct signature
        assert hasattr(chunker, "chunk")

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            result = chunker.chunk(
                document=None,
                parsed=sample_parsed_result,
                context=sample_processing_context,
                config=sample_pipeline_config,
            )

        # Should return (chunks, stats) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        chunks, stats = result
        assert isinstance(chunks, (list, tuple))
        assert isinstance(stats, dict)

    def test_hybrid_chunker_quality_error_handling(
        self,
        stub_embedding_client,
        sample_parsed_result,
        sample_processing_context,
        sample_pipeline_config,
    ):
        """Test that quality evaluation errors don't crash chunking."""
        config = ChunkerConfig(
            mode=ChunkerMode.LATE,
            enable_quality_metrics=True,
        )
        chunker = HybridChunker(config)

        def mock_llm_error(*args, **kwargs):
            raise Exception("LLM API error")

        with patch("ai_core.rag.embeddings.get_embedding_client") as mock_get_client:
            mock_get_client.return_value = stub_embedding_client

            with patch("litellm.completion", side_effect=mock_llm_error):
                # Should not crash, but skip quality evaluation
                chunks, stats = chunker.chunk(
                    document=None,
                    parsed=sample_parsed_result,
                    context=sample_processing_context,
                    config=sample_pipeline_config,
                )

        # Should still produce chunks
        assert len(chunks) > 0
        assert stats["chunk.count"] == len(chunks)


class TestChunkerConfig:
    """Test ChunkerConfig dataclass."""

    def test_chunker_config_defaults(self):
        """Test ChunkerConfig default values (MODEL_ROUTING.yaml labels)."""
        config = ChunkerConfig()

        assert config.mode == ChunkerMode.LATE
        assert config.late_chunk_model == "embedding"  # MODEL_ROUTING.yaml label
        assert config.enable_quality_metrics is True
        assert config.max_chunk_tokens == 450

    def test_chunker_config_custom(self):
        """Test ChunkerConfig with custom values."""
        config = ChunkerConfig(
            mode=ChunkerMode.AGENTIC,
            late_chunk_model="custom-model",
            enable_quality_metrics=False,
            max_chunk_tokens=600,
        )

        assert config.mode == ChunkerMode.AGENTIC
        assert config.late_chunk_model == "custom-model"
        assert config.enable_quality_metrics is False
        assert config.max_chunk_tokens == 600

    def test_chunker_config_frozen(self):
        """Test that ChunkerConfig is frozen (immutable)."""
        config = ChunkerConfig()

        with pytest.raises(AttributeError):
            config.mode = ChunkerMode.AGENTIC  # Should fail (frozen)


class TestChunkerMode:
    """Test ChunkerMode enum."""

    def test_chunker_mode_values(self):
        """Test ChunkerMode enum values."""
        assert ChunkerMode.LATE.value == "late"
        assert ChunkerMode.AGENTIC.value == "agentic"
        assert ChunkerMode.HYBRID.value == "hybrid"

    def test_chunker_mode_from_string(self):
        """Test creating ChunkerMode from string."""
        assert ChunkerMode("late") == ChunkerMode.LATE
        assert ChunkerMode("agentic") == ChunkerMode.AGENTIC
        assert ChunkerMode("hybrid") == ChunkerMode.HYBRID
