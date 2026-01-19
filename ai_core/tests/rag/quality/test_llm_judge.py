"""Unit tests for ChunkQualityEvaluator (LLM-as-Judge)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_core.rag.quality.llm_judge import (
    ChunkQualityEvaluator,
    ChunkQualityScore,
    compute_quality_statistics,
)


class TestChunkQualityEvaluator:
    """Test ChunkQualityEvaluator implementation."""

    def test_quality_evaluator_basic(
        self,
        mock_llm_judge,
        sample_chunks,
    ):
        """Test basic quality evaluation."""
        evaluator = ChunkQualityEvaluator(model="quality-eval")

        def fake_call(*_args, **_kwargs):
            response = mock_llm_judge("quality-eval", [{"content": "test"}])
            return {"text": response.choices[0].message.content}

        with patch("ai_core.llm.client.call", side_effect=fake_call):
            scores = evaluator.evaluate(sample_chunks)

        # Should produce one score per chunk
        assert len(scores) == len(sample_chunks)

        # Check score structure
        score = scores[0]
        assert isinstance(score, ChunkQualityScore)
        assert 0 <= score.coherence <= 100
        assert 0 <= score.completeness <= 100
        assert 0 <= score.reference_resolution <= 100
        assert 0 <= score.redundancy <= 100
        assert 0 <= score.overall <= 100

    def test_quality_evaluator_sample_rate(
        self,
        mock_llm_judge,
        sample_chunks,
    ):
        """Test that sample_rate controls evaluation frequency."""
        evaluator = ChunkQualityEvaluator(
            model="quality-eval",
            sample_rate=0.0,  # Sample 0% (skip all)
        )

        def fake_call(*_args, **_kwargs):
            response = mock_llm_judge("quality-eval", [{"content": "test"}])
            return {"text": response.choices[0].message.content}

        with patch("ai_core.llm.client.call", side_effect=fake_call):
            scores = evaluator.evaluate(sample_chunks)

        # Should still return scores (default scores)
        assert len(scores) == len(sample_chunks)

        # All scores should be 0.0 (default/skipped)
        for score in scores:
            assert score.overall == 0.0

    def test_quality_evaluator_handles_errors(
        self,
        sample_chunks,
    ):
        """Test that evaluator handles LLM errors gracefully."""
        evaluator = ChunkQualityEvaluator(model="quality-eval")

        def mock_llm_error(*_args, **_kwargs):
            raise Exception("LLM API error")

        with patch("ai_core.llm.client.call", side_effect=mock_llm_error):
            scores = evaluator.evaluate(sample_chunks)

        # Should return default scores (not crash)
        assert len(scores) == len(sample_chunks)

        for score in scores:
            assert score.overall == 0.0  # Default score

    def test_add_quality_to_chunks(
        self,
        mock_llm_judge,
        sample_chunks,
    ):
        """Test adding quality scores to chunk metadata."""
        evaluator = ChunkQualityEvaluator(model="quality-eval")

        def fake_call(*_args, **_kwargs):
            response = mock_llm_judge("quality-eval", [{"content": "test"}])
            return {"text": response.choices[0].message.content}

        with patch("ai_core.llm.client.call", side_effect=fake_call):
            scores = evaluator.evaluate(sample_chunks)
            updated_chunks = evaluator.add_quality_to_chunks(sample_chunks, scores)

        # Should add quality metadata to each chunk
        for chunk in updated_chunks:
            assert "metadata" in chunk
            assert "quality" in chunk["metadata"]

            quality = chunk["metadata"]["quality"]
            assert "coherence" in quality
            assert "completeness" in quality
            assert "overall" in quality
            assert "evaluated_by" in quality
            assert quality["evaluated_by"] == "llm-judge-v1"

    def test_add_quality_mismatch_warning(
        self,
        sample_chunks,
        sample_quality_scores,
    ):
        """Test that mismatched chunk/score counts emit warning."""
        evaluator = ChunkQualityEvaluator(model="quality-eval")

        # Intentionally mismatch: 3 chunks, 2 scores
        mismatched_scores = sample_quality_scores[:2]

        # Should return original chunks (not crash)
        updated_chunks = evaluator.add_quality_to_chunks(
            sample_chunks, mismatched_scores
        )

        assert len(updated_chunks) == len(sample_chunks)


class TestChunkQualityScore:
    """Test ChunkQualityScore dataclass."""

    def test_quality_score_creation(self):
        """Test creating ChunkQualityScore."""
        score = ChunkQualityScore(
            chunk_id="test-123",
            coherence=85.0,
            completeness=90.0,
            reference_resolution=75.0,
            redundancy=95.0,
            overall=86.25,
            reasoning="Test score",
        )

        assert score.chunk_id == "test-123"
        assert score.coherence == 85.0
        assert score.overall == 86.25

    def test_quality_score_to_dict(self):
        """Test converting ChunkQualityScore to dict."""
        score = ChunkQualityScore(
            chunk_id="test-123",
            coherence=85.0,
            completeness=90.0,
            reference_resolution=75.0,
            redundancy=95.0,
            overall=86.25,
        )

        score_dict = score.to_dict()

        assert score_dict["chunk_id"] == "test-123"
        assert score_dict["coherence"] == 85.0
        assert score_dict["overall"] == 86.25

    def test_quality_score_frozen(self):
        """Test that ChunkQualityScore is frozen (immutable)."""
        score = ChunkQualityScore(
            chunk_id="test-123",
            coherence=85.0,
            completeness=90.0,
            reference_resolution=75.0,
            redundancy=95.0,
            overall=86.25,
        )

        with pytest.raises(AttributeError):
            score.coherence = 100.0  # Should fail (frozen)


class TestComputeQualityStatistics:
    """Test compute_quality_statistics helper function."""

    def test_compute_quality_statistics(
        self,
        sample_quality_scores,
    ):
        """Test computing statistics from quality scores."""
        stats = compute_quality_statistics(sample_quality_scores)

        assert "count" in stats
        assert "mean_coherence" in stats
        assert "mean_overall" in stats
        assert "min_overall" in stats
        assert "max_overall" in stats

        assert stats["count"] == len(sample_quality_scores)
        assert 0 <= stats["mean_overall"] <= 100
        assert stats["min_overall"] <= stats["mean_overall"] <= stats["max_overall"]

    def test_compute_quality_statistics_empty(self):
        """Test computing statistics from empty score list."""
        stats = compute_quality_statistics([])

        assert stats["count"] == 0
        assert stats["mean_overall"] == 0.0
