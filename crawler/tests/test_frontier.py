"""Tests for crawler frontier scheduling and robots policies."""

from datetime import datetime, timedelta, timezone

from crawler import (
    CrawlSignals,
    FrontierAction,
    HostPolitenessPolicy,
    HostVisitState,
    RecrawlFrequency,
    RobotsPolicy,
    SourceDescriptor,
    decide_frontier_action,
)


def aware(**kwargs):
    return datetime(**kwargs, tzinfo=timezone.utc)


def test_robots_disallow_skips_source():
    descriptor = SourceDescriptor(host="example.com", path="/private/data")
    robots = RobotsPolicy(disallow=("/private",))

    decision = decide_frontier_action(descriptor, robots=robots)

    assert decision.action is FrontierAction.SKIP
    assert decision.reason == "robots_disallow"
    assert "robots_disallow" in decision.policy_events
    assert decision.earliest_visit_at is None


def test_crawl_delay_enforces_minimum_wait():
    now = aware(year=2024, month=1, day=1)
    last_visit = now - timedelta(seconds=10)
    descriptor = SourceDescriptor(host="example.com", path="/news", provider="news")
    robots = RobotsPolicy(crawl_delay=30.0)
    signals = CrawlSignals(
        last_crawled_at=last_visit,
        manual_recrawl_interval=timedelta(seconds=1),
    )

    decision = decide_frontier_action(
        descriptor,
        signals,
        robots=robots,
        now=now,
    )

    assert decision.action is FrontierAction.DEFER
    assert decision.reason == "robots_crawl_delay"
    assert decision.earliest_visit_at == last_visit + timedelta(seconds=30)


def test_host_parallelism_defers_until_slot_available():
    now = aware(year=2024, month=1, day=1, hour=8)
    descriptor = SourceDescriptor(host="example.com", path="/docs")
    host_policy = HostPolitenessPolicy(
        max_parallelism=1, min_delay=timedelta(seconds=5)
    )
    host_state = HostVisitState(
        active_requests=1,
        next_available_at=now + timedelta(seconds=20),
        last_completed_at=now - timedelta(seconds=2),
    )

    decision = decide_frontier_action(
        descriptor,
        robots=RobotsPolicy(),
        host_policy=host_policy,
        host_state=host_state,
        now=now,
    )

    assert decision.action is FrontierAction.DEFER
    assert decision.reason == "host_parallel_limit"
    assert decision.earliest_visit_at == host_state.next_available_at


def test_frequency_overrides_from_metadata():
    now = aware(year=2024, month=1, day=2)
    last_visit = now - timedelta(days=1)
    descriptor = SourceDescriptor(
        host="example.com",
        path="/static/index.html",
        metadata={"recrawl_frequency": "rare"},
    )
    signals = CrawlSignals(last_crawled_at=last_visit)

    decision = decide_frontier_action(
        descriptor,
        signals,
        now=now,
    )

    assert decision.action is FrontierAction.DEFER
    assert decision.reason == "recrawl_schedule"
    assert decision.earliest_visit_at == last_visit + timedelta(days=7)


def test_observed_changes_promote_recrawl():
    now = aware(year=2024, month=3, day=10, hour=12)
    last_visit = now - timedelta(hours=3)
    descriptor = SourceDescriptor(host="example.com", path="/archive")
    signals = CrawlSignals(
        last_crawled_at=last_visit,
        observed_change_interval=timedelta(hours=2),
        override_recrawl_frequency=RecrawlFrequency.INFREQUENT.value,
    )

    decision = decide_frontier_action(
        descriptor,
        signals,
        now=now,
    )

    assert decision.action is FrontierAction.ENQUEUE
    assert decision.reason == "ready"


def test_consecutive_unchanged_backoff():
    now = aware(year=2024, month=5, day=5)
    last_visit = now - timedelta(hours=12)
    descriptor = SourceDescriptor(host="example.com", path="/docs/guide")
    signals = CrawlSignals(
        last_crawled_at=last_visit,
        consecutive_unchanged=3,
    )

    decision = decide_frontier_action(
        descriptor,
        signals,
        now=now,
    )

    assert decision.action is FrontierAction.DEFER
    assert decision.reason == "recrawl_schedule"
    assert decision.earliest_visit_at > now + timedelta(days=1)


def test_manual_recrawl_interval_non_positive_is_ignored():
    now = aware(year=2024, month=6, day=1)
    last_visit = now - timedelta(hours=1)
    descriptor = SourceDescriptor(host="example.com", path="/docs/policy")
    signals = CrawlSignals(
        last_crawled_at=last_visit,
        manual_recrawl_interval=timedelta(seconds=-30),
    )

    decision = decide_frontier_action(
        descriptor,
        signals,
        now=now,
    )

    assert decision.action is FrontierAction.DEFER
    assert decision.reason == "recrawl_schedule"
    # Falls back to the standard interval of one day.
    assert decision.earliest_visit_at == last_visit + timedelta(days=1)


def test_last_modified_signals_slower_recrawl():
    now = aware(year=2024, month=6, day=2)
    last_visit = now - timedelta(hours=3)
    descriptor = SourceDescriptor(host="example.com", path="/archive/old")
    signals = CrawlSignals(
        last_crawled_at=last_visit,
        last_modified=last_visit - timedelta(days=10),
    )

    decision = decide_frontier_action(
        descriptor,
        signals,
        now=now,
    )

    assert decision.action is FrontierAction.DEFER
    assert decision.reason == "recrawl_schedule"
    # Freshness signal lengthens the interval up to 2x the standard schedule.
    assert decision.earliest_visit_at == last_visit + timedelta(days=2)


def test_failure_retire_threshold():
    descriptor = SourceDescriptor(host="example.com", path="/data")
    signals = CrawlSignals(consecutive_failures=6)

    decision = decide_frontier_action(descriptor, signals)

    assert decision.action is FrontierAction.RETIRE
    assert decision.reason == "retire_after_failures"


def test_failure_backoff_is_capped():
    now = aware(year=2024, month=7, day=1, hour=12)
    descriptor = SourceDescriptor(host="example.com", path="/slow")
    # Stay below the retire threshold to exercise the capped backoff logic.
    signals = CrawlSignals(consecutive_failures=4)

    decision = decide_frontier_action(
        descriptor,
        signals,
        robots=RobotsPolicy(),
        now=now,
    )

    assert decision.action is FrontierAction.DEFER
    assert decision.reason == "failure_backoff"
    assert decision.earliest_visit_at == now + timedelta(minutes=4)
    assert decision.earliest_visit_at <= now + timedelta(minutes=15)
