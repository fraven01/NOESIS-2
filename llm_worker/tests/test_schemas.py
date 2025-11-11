from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from llm_worker.schemas import (
    CoverageDimension,
    LLMScoredItem,
    RAGCoverageSummary,
    ScoringContext,
    SearchCandidate,
)


def test_rag_coverage_summary_roundtrip_preserves_facets() -> None:
    summary = RAGCoverageSummary(
        document_id=UUID("64f8220b-0ea9-4ccf-9c38-83005c765d13"),
        title="Data residency policy",
        key_points=[
            "Defines residency zones",
            "Documents transfer controls",
            "Lists auditing duties",
        ],
        coverage_facets={CoverageDimension.LEGAL: 0.8},
        custom_facets={"monitoring": 0.5},
        last_ingested_at=datetime(2024, 5, 7, 12, 30, tzinfo=timezone.utc),
    )

    serialised = summary.model_dump_json()
    restored = RAGCoverageSummary.model_validate_json(serialised)

    assert restored == summary


def test_llm_scored_item_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LLMScoredItem(
            candidate_id="doc-1",
            score=75,
            reason="Covers the requested procedure",
            facet_coverage={CoverageDimension.PROCEDURAL: 1.0},
            unexpected_field="not allowed",
        )


def test_search_candidate_detected_date_normalises_to_utc() -> None:
    candidate = SearchCandidate(
        id="abc-123",
        title="Policy",
        snippet="",
        detected_date=datetime(2024, 6, 1, 9, tzinfo=timezone(timedelta(hours=2))),
    )

    assert candidate.detected_date == datetime(2024, 6, 1, 7, tzinfo=timezone.utc)


def test_search_candidate_detected_date_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        SearchCandidate(
            id="abc-123",
            title="Policy",
            snippet="",
            detected_date=datetime(2024, 6, 1, 9, 0, 0),
        )


def test_llm_scored_item_serialises_enum_keys() -> None:
    item = LLMScoredItem(
        candidate_id="doc-1",
        score=80,
        reason="Coverage",
        facet_coverage={CoverageDimension.PROCEDURAL: 0.7},
    )

    payload = json.loads(item.model_dump_json())

    assert payload["facet_coverage"] == {"PROCEDURAL": 0.7}


def test_llm_scored_item_allows_custom_facets() -> None:
    item = LLMScoredItem(
        candidate_id="doc-2",
        score=72,
        reason="Adds analytics",
        facet_coverage={CoverageDimension.ANALYTICS_REPORTING: 0.6},
        custom_facets={"custom_reporting": 0.45, "CUSTOM_EXTRA": 1.2},
    )

    assert item.custom_facets == {"CUSTOM_REPORTING": 0.45, "CUSTOM_EXTRA": 1.0}

    with pytest.raises(ValidationError):
        LLMScoredItem(
            candidate_id="doc-3",
            score=60,
            reason="Invalid",
            facet_coverage={},
            custom_facets={"custom_reporting": "high"},
        )


def test_scoring_context_min_diversity_validation() -> None:
    context = ScoringContext(
        question="What applies?",
        purpose="audit",
        jurisdiction="DE",
        output_target="summary",
        preferred_sources=[],
        disallowed_sources=[],
        collection_scope="compliance",
        min_diversity_buckets=4,
    )

    assert context.min_diversity_buckets == 4

    with pytest.raises(ValidationError):
        ScoringContext(
            question="What applies?",
            purpose="audit",
            jurisdiction="DE",
            output_target="summary",
            preferred_sources=[],
            disallowed_sources=[],
            collection_scope="compliance",
            min_diversity_buckets=0,
        )
