from __future__ import annotations

from typing import Any, Mapping

from ai_core.rag.chunking import RoutingAwareChunker
from ai_core.rag.quality import llm_judge


def test_chunker_quality_baseline(
    sample_parsed_result,
    sample_processing_context,
    sample_pipeline_config,
    monkeypatch,
) -> None:
    def _fake_evaluate(
        _self, chunks: list[Mapping[str, Any]], _context: Any
    ) -> list[llm_judge.ChunkQualityScore]:
        scores: list[llm_judge.ChunkQualityScore] = []
        for index, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id") or f"chunk-{index}"
            scores.append(
                llm_judge.ChunkQualityScore(
                    chunk_id=str(chunk_id),
                    coherence=75.0,
                    completeness=70.0,
                    reference_resolution=72.0,
                    redundancy=80.0,
                    overall=74.0,
                    reasoning="mocked baseline",
                )
            )
        return scores

    monkeypatch.setattr(
        llm_judge.ChunkQualityEvaluator, "evaluate", _fake_evaluate, raising=True
    )

    chunker = RoutingAwareChunker()
    chunks, stats = chunker.chunk(
        None,
        sample_parsed_result,
        context=sample_processing_context,
        config=sample_pipeline_config,
    )

    assert stats["chunk.count"] > 0
    assert stats["quality.mean_coherence"] > 60.0
    assert stats["quality.mean_completeness"] > 60.0
