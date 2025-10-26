from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crawler.lifecycle_model import (
    InvalidLifecycleTransition,
    LifecycleStatus,
    LifecycleTimeline,
)


def test_full_pipeline_sequence_reaches_ingested() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    timeline = LifecycleTimeline()
    timeline = timeline.advance(
        LifecycleStatus.SEEDED,
        occurred_at=base,
        attributes={"crawler.provider": "web"},
    )
    timeline = timeline.advance(
        LifecycleStatus.QUEUED,
        occurred_at=base,
        duration_ms=5.0,
    )
    timeline = timeline.advance(
        LifecycleStatus.FETCHED,
        occurred_at=base,
        duration_ms=42.5,
        attributes={"http.status_code": 200},
    )
    timeline = timeline.advance(
        LifecycleStatus.PARSED,
        occurred_at=base,
        attributes={"parser.media_type": "text/html"},
    )
    timeline = timeline.advance(
        LifecycleStatus.NORMALIZED,
        occurred_at=base,
        duration_ms=3.2,
    )
    timeline = timeline.advance(
        LifecycleStatus.DELTA_NEW,
        occurred_at=base,
        attributes={"delta.version": 2},
    )
    timeline = timeline.advance(
        LifecycleStatus.INGESTED,
        occurred_at=base,
        duration_ms=8.7,
        attributes={"ingest.chunks": 4},
    )

    assert timeline.is_complete is True
    assert timeline.current_status is LifecycleStatus.INGESTED
    assert timeline.events[-1].attributes_dict()["ingest.chunks"] == 4
    assert timeline.events[0].attributes == (("crawler.provider", "web"),)


def test_denied_source_requires_skip_or_retire() -> None:
    timeline = LifecycleTimeline()
    timeline = timeline.advance(LifecycleStatus.SEEDED)
    timeline = timeline.advance(LifecycleStatus.QUEUED)
    timeline = timeline.advance(
        LifecycleStatus.DENIED,
        attributes={"policy.reason": "robots_deny"},
    )

    with pytest.raises(InvalidLifecycleTransition):
        timeline.advance(LifecycleStatus.PARSED)

    completed = timeline.advance(LifecycleStatus.SKIPPED)
    assert completed.is_complete is True
    assert completed.current_status is LifecycleStatus.SKIPPED

    with pytest.raises(InvalidLifecycleTransition):
        completed.advance(LifecycleStatus.QUEUED)


def test_fetch_gone_can_retire_directly() -> None:
    timeline = LifecycleTimeline()
    timeline = timeline.advance(LifecycleStatus.SEEDED)
    timeline = timeline.advance(LifecycleStatus.QUEUED)
    timeline = timeline.advance(
        LifecycleStatus.FETCHED,
        attributes={"http.status_code": 410},
    )
    timeline = timeline.advance(
        LifecycleStatus.RETIRED,
        attributes={"lifecycle.reason": "gone_410"},
    )

    assert timeline.is_complete is True
    assert timeline.current_status is LifecycleStatus.RETIRED


def test_timeline_must_start_with_seeded() -> None:
    timeline = LifecycleTimeline()

    with pytest.raises(InvalidLifecycleTransition):
        timeline.advance(LifecycleStatus.QUEUED)


def test_negative_duration_rejected() -> None:
    timeline = LifecycleTimeline()

    with pytest.raises(ValueError):
        timeline.advance(LifecycleStatus.SEEDED, duration_ms=-1.0)

