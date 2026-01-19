"""Unit tests for auto_ingest URL selection logic."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ai_core.services.collection_search.auto_ingest import select_auto_ingest_urls


class FakeScoredItem(BaseModel):
    """Mimics LLMScoredItem structure - has candidate_id and score but NO url."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    score: float


class FakeCandidate(BaseModel):
    """Mimics SearchCandidate structure - has id and url."""

    model_config = ConfigDict(frozen=True)

    id: str
    url: str


class TestSelectAutoIngestUrls:
    """Tests for select_auto_ingest_urls function."""

    def test_resolves_urls_via_candidate_lookup(self) -> None:
        """Core bug fix: URLs must be resolved from candidate_id lookup."""
        ranked = [
            FakeScoredItem(candidate_id="c1", score=85.0),
            FakeScoredItem(candidate_id="c2", score=75.0),
            FakeScoredItem(candidate_id="c3", score=65.0),
        ]
        candidate_by_id = {
            "c1": FakeCandidate(id="c1", url="https://example.com/doc1"),
            "c2": FakeCandidate(id="c2", url="https://example.com/doc2"),
            "c3": FakeCandidate(id="c3", url="https://example.com/doc3"),
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert result == [
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3",
        ]

    def test_filters_by_min_score(self) -> None:
        """Only items meeting min_score threshold are selected."""
        ranked = [
            FakeScoredItem(candidate_id="c1", score=85.0),
            FakeScoredItem(candidate_id="c2", score=55.0),  # Below threshold
            FakeScoredItem(candidate_id="c3", score=70.0),
        ]
        candidate_by_id = {
            "c1": FakeCandidate(id="c1", url="https://example.com/doc1"),
            "c2": FakeCandidate(id="c2", url="https://example.com/doc2"),
            "c3": FakeCandidate(id="c3", url="https://example.com/doc3"),
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert result == [
            "https://example.com/doc1",
            "https://example.com/doc3",
        ]
        assert "https://example.com/doc2" not in result

    def test_respects_top_k_limit(self) -> None:
        """Only top_k URLs are returned even if more meet threshold."""
        ranked = [
            FakeScoredItem(candidate_id="c1", score=90.0),
            FakeScoredItem(candidate_id="c2", score=85.0),
            FakeScoredItem(candidate_id="c3", score=80.0),
            FakeScoredItem(candidate_id="c4", score=75.0),
            FakeScoredItem(candidate_id="c5", score=70.0),
        ]
        candidate_by_id = {
            f"c{i}": FakeCandidate(id=f"c{i}", url=f"https://example.com/doc{i}")
            for i in range(1, 6)
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=3,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert len(result) == 3
        assert result == [
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3",
        ]

    def test_returns_empty_when_no_results_meet_threshold(self) -> None:
        """Returns empty list when all scores below threshold."""
        ranked = [
            FakeScoredItem(candidate_id="c1", score=50.0),
            FakeScoredItem(candidate_id="c2", score=45.0),
        ]
        candidate_by_id = {
            "c1": FakeCandidate(id="c1", url="https://example.com/doc1"),
            "c2": FakeCandidate(id="c2", url="https://example.com/doc2"),
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert result == []

    def test_returns_empty_when_ranked_is_empty(self) -> None:
        """Returns empty list when no ranked items provided."""
        result = select_auto_ingest_urls(
            [],
            top_k=10,
            min_score=60.0,
            candidate_by_id={},
        )

        assert result == []

    def test_skips_items_without_url_in_lookup(self) -> None:
        """Items whose candidate_id is not in lookup are skipped."""
        ranked = [
            FakeScoredItem(candidate_id="c1", score=85.0),
            FakeScoredItem(candidate_id="c2", score=75.0),  # Not in lookup
            FakeScoredItem(candidate_id="c3", score=70.0),
        ]
        candidate_by_id = {
            "c1": FakeCandidate(id="c1", url="https://example.com/doc1"),
            # c2 missing from lookup
            "c3": FakeCandidate(id="c3", url="https://example.com/doc3"),
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert result == [
            "https://example.com/doc1",
            "https://example.com/doc3",
        ]

    def test_prefers_direct_url_over_lookup(self) -> None:
        """If item has url directly, use it instead of lookup."""

        class ItemWithUrl(BaseModel):
            model_config = ConfigDict(frozen=True)
            candidate_id: str
            score: float
            url: str

        ranked = [
            ItemWithUrl(candidate_id="c1", score=85.0, url="https://direct.com/doc1"),
        ]
        candidate_by_id = {
            "c1": FakeCandidate(id="c1", url="https://lookup.com/doc1"),
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        # Direct URL takes precedence
        assert result == ["https://direct.com/doc1"]

    def test_works_with_dict_items(self) -> None:
        """Function works with dict items (not just Pydantic models)."""
        ranked = [
            {"candidate_id": "c1", "score": 85.0},
            {"candidate_id": "c2", "score": 75.0},
        ]
        candidate_by_id = {
            "c1": {"id": "c1", "url": "https://example.com/doc1"},
            "c2": {"id": "c2", "url": "https://example.com/doc2"},
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert result == [
            "https://example.com/doc1",
            "https://example.com/doc2",
        ]

    def test_works_without_candidate_lookup(self) -> None:
        """Backwards compatible: works if items have url directly."""

        class ItemWithUrl(BaseModel):
            model_config = ConfigDict(frozen=True)
            score: float
            url: str

        ranked = [
            ItemWithUrl(score=85.0, url="https://example.com/doc1"),
            ItemWithUrl(score=75.0, url="https://example.com/doc2"),
        ]

        # No candidate_by_id provided
        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
        )

        assert result == [
            "https://example.com/doc1",
            "https://example.com/doc2",
        ]

    def test_boundary_score_exactly_at_threshold(self) -> None:
        """Score exactly at threshold should be included."""
        ranked = [
            FakeScoredItem(candidate_id="c1", score=60.0),  # Exactly at threshold
        ]
        candidate_by_id = {
            "c1": FakeCandidate(id="c1", url="https://example.com/doc1"),
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert result == ["https://example.com/doc1"]

    def test_boundary_score_just_below_threshold(self) -> None:
        """Score just below threshold should be excluded."""
        ranked = [
            FakeScoredItem(candidate_id="c1", score=59.9),
        ]
        candidate_by_id = {
            "c1": FakeCandidate(id="c1", url="https://example.com/doc1"),
        }

        result = select_auto_ingest_urls(
            ranked,
            top_k=10,
            min_score=60.0,
            candidate_by_id=candidate_by_id,
        )

        assert result == []
