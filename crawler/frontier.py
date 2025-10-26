"""Frontier policies for robots compliance and recrawl scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import Decision


class FrontierAction(str, Enum):
    """Supported high-level decisions for the crawl frontier."""

    ENQUEUE = "enqueue"
    DEFER = "defer"
    SKIP = "skip"
    RETIRE = "retire"


@dataclass(frozen=True)
class RobotsPolicy:
    """Materialized `robots.txt` directives for a single host.

    The allow/disallow patterns are treated as prefix matches only. Wildcard
    semantics such as ``*`` or ``$`` must be modeled explicitly by callers when
    needed.
    """

    allow: Tuple[str, ...] = field(default_factory=tuple)
    disallow: Tuple[str, ...] = field(default_factory=tuple)
    crawl_delay: Optional[float] = None  # seconds

    def __post_init__(self) -> None:
        object.__setattr__(self, "allow", _normalize_patterns(self.allow))
        object.__setattr__(self, "disallow", _normalize_patterns(self.disallow))
        if self.crawl_delay is not None and self.crawl_delay < 0:
            raise ValueError("crawl_delay_negative")


@dataclass(frozen=True)
class HostPolitenessPolicy:
    """Concurrency and pacing rules for a host."""

    max_parallelism: int = 1
    min_delay: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        if self.max_parallelism < 1:
            raise ValueError("max_parallelism_invalid")
        if self.min_delay < timedelta(0):
            raise ValueError("min_delay_negative")


@dataclass(frozen=True)
class HostVisitState:
    """Current visit state for politeness evaluation."""

    active_requests: int = 0
    last_completed_at: Optional[datetime] = None
    next_available_at: Optional[datetime] = None


class RecrawlFrequency(str, Enum):
    """Predefined scheduling buckets for common content classes.

    ``FREQUENT`` targets change-prone sources such as news feeds, ``STANDARD``
    covers reference material and general documentation, and ``INFREQUENT`` is
    intended for largely static pages or archives that rarely update.
    """

    FREQUENT = "frequent"  # news, change-prone
    STANDARD = "standard"  # documentation, reference
    INFREQUENT = "rare"  # static pages, archives


RECRAWL_INTERVALS = {
    RecrawlFrequency.FREQUENT: timedelta(hours=1),
    RecrawlFrequency.STANDARD: timedelta(days=1),
    RecrawlFrequency.INFREQUENT: timedelta(days=7),
}


@dataclass(frozen=True)
class CrawlSignals:
    """Latest recorded signals for a source."""

    last_crawled_at: Optional[datetime] = None
    last_etag: Optional[str] = None
    last_modified: Optional[datetime] = None
    observed_change_interval: Optional[timedelta] = None
    consecutive_unchanged: int = 0
    consecutive_failures: int = 0
    manual_recrawl_interval: Optional[timedelta] = None
    override_recrawl_frequency: Optional[str] = None
    retire: bool = False


@dataclass(frozen=True)
class SourceDescriptor:
    """Minimal identity for a source when making frontier decisions."""

    host: str
    path: str
    provider: str = "web"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_host = (self.host or "").strip().lower()
        if not normalized_host:
            raise ValueError("host_required")
        normalized_path = _ensure_leading_slash(self.path)
        object.__setattr__(self, "host", normalized_host)
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "provider", (self.provider or "web").strip().lower())


@dataclass(frozen=True)
class FrontierDecision(Decision):
    """Outcome of the frontier policy evaluation using the shared payload."""

    @classmethod
    def from_legacy(
        cls,
        action: FrontierAction,
        earliest_visit_at: Optional[datetime],
        reason: str,
        policy_events: Tuple[str, ...] = (),
    ) -> "FrontierDecision":
        attributes = {
            "earliest_visit_at": earliest_visit_at,
            "policy_events": tuple(policy_events),
        }
        return cls(action.value, reason, attributes)

    @property
    def action(self) -> FrontierAction:
        return FrontierAction(self.decision)

    @property
    def earliest_visit_at(self) -> Optional[datetime]:
        return self.attributes.get("earliest_visit_at")

    @property
    def policy_events(self) -> Tuple[str, ...]:
        return tuple(self.attributes.get("policy_events", ()))


FAILURE_RETIRE_THRESHOLD = 5
FAILURE_BACKOFF_THRESHOLD = 3
FAILURE_BACKOFF_CAP = timedelta(minutes=15)


def decide_frontier_action(
    source: SourceDescriptor,
    signals: Optional[CrawlSignals] = None,
    *,
    robots: Optional[RobotsPolicy] = None,
    host_policy: Optional[HostPolitenessPolicy] = None,
    host_state: Optional[HostVisitState] = None,
    now: Optional[datetime] = None,
) -> FrontierDecision:
    """Evaluate scheduling and compliance rules for a source."""

    current_time = _ensure_aware(now)
    active_signals = signals or CrawlSignals()
    events = []

    if active_signals.retire:
        return FrontierDecision.from_legacy(
            FrontierAction.RETIRE, None, "manual_retire", tuple(events)
        )

    if active_signals.consecutive_failures >= FAILURE_RETIRE_THRESHOLD:
        events.append("retire_after_failures")
        return FrontierDecision.from_legacy(
            FrontierAction.RETIRE,
            None,
            "retire_after_failures",
            tuple(events),
        )

    robots_policy = robots or RobotsPolicy()
    robots_allowed, robots_event = _check_robots(robots_policy, source.path)
    if not robots_allowed:
        events.append("robots_disallow")
        return FrontierDecision.from_legacy(
            FrontierAction.SKIP, None, "robots_disallow", tuple(events)
        )
    if robots_event:
        events.append(robots_event)

    host_rules = host_policy or HostPolitenessPolicy()
    host_status = host_state or HostVisitState()

    frequency = _resolve_frequency(source, active_signals)
    interval = _compute_recrawl_interval(frequency, active_signals)

    earliest = current_time
    reason = "ready"

    if active_signals.last_crawled_at:
        scheduled_time = active_signals.last_crawled_at + interval
        if scheduled_time > earliest:
            earliest = scheduled_time
            reason = "recrawl_schedule"

    if robots_policy.crawl_delay is not None:
        delay = timedelta(seconds=robots_policy.crawl_delay)
        reference_time = active_signals.last_crawled_at or current_time
        robots_ready = reference_time + delay
        if robots_ready > earliest:
            earliest = robots_ready
            reason = "robots_crawl_delay"

    host_ready, host_reason = _apply_host_rules(host_rules, host_status, current_time)
    if host_ready > earliest:
        earliest = host_ready
        reason = host_reason

    if active_signals.consecutive_failures >= FAILURE_BACKOFF_THRESHOLD:
        backoff_seconds = min(
            60 * active_signals.consecutive_failures,
            FAILURE_BACKOFF_CAP.total_seconds(),
        )
        failure_ready = current_time + timedelta(seconds=backoff_seconds)
        if failure_ready > earliest:
            earliest = failure_ready
            reason = "failure_backoff"

    if earliest > current_time:
        return FrontierDecision.from_legacy(
            FrontierAction.DEFER, earliest, reason, tuple(events)
        )
    return FrontierDecision.from_legacy(
        FrontierAction.ENQUEUE, earliest, reason, tuple(events)
    )


def _ensure_aware(candidate: Optional[datetime]) -> datetime:
    if candidate is None:
        return datetime.now(timezone.utc)
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _normalize_patterns(patterns: Sequence[str]) -> Tuple[str, ...]:
    normalized = []
    for pattern in patterns:
        value = (pattern or "").strip()
        if not value:
            continue
        if not value.startswith("/"):
            value = f"/{value}"
        normalized.append(value)
    return tuple(normalized)


def _ensure_leading_slash(path: str) -> str:
    value = (path or "/").strip()
    if not value.startswith("/"):
        value = f"/{value}"
    if not value:
        return "/"
    return value


def _check_robots(policy: RobotsPolicy, path: str) -> Tuple[bool, Optional[str]]:
    matched_allow = _longest_match(policy.allow, path)
    matched_disallow = _longest_match(policy.disallow, path)

    if matched_disallow and (
        not matched_allow or len(matched_disallow) > len(matched_allow)
    ):
        return False, None
    if matched_allow:
        return True, "robots_allow"
    return True, None


def _longest_match(patterns: Iterable[str], path: str) -> Optional[str]:
    best: Optional[str] = None
    for pattern in patterns:
        if path.startswith(pattern) and (best is None or len(pattern) > len(best)):
            best = pattern
    return best


def _resolve_frequency(
    source: SourceDescriptor, signals: CrawlSignals
) -> RecrawlFrequency:
    override_value: Optional[str] = None
    metadata_value = None
    if source.metadata:
        metadata_value = source.metadata.get("recrawl_frequency")
    if signals.override_recrawl_frequency:
        override_value = signals.override_recrawl_frequency
    candidate = override_value or metadata_value
    if isinstance(candidate, RecrawlFrequency):
        return candidate
    if isinstance(candidate, str):
        normalized = candidate.strip().lower()
        for frequency in RecrawlFrequency:
            if normalized in {frequency.value, frequency.name.lower()}:
                return frequency
    return (
        RecrawlFrequency.FREQUENT
        if source.provider == "news"
        else RecrawlFrequency.STANDARD
    )


def _compute_recrawl_interval(
    frequency: RecrawlFrequency, signals: CrawlSignals
) -> timedelta:
    interval = RECRAWL_INTERVALS[frequency]
    if signals.manual_recrawl_interval is not None:
        return _ensure_positive(signals.manual_recrawl_interval, interval)

    if signals.override_recrawl_frequency:
        normalized = signals.override_recrawl_frequency.strip().lower()
        for candidate in RecrawlFrequency:
            if normalized in {candidate.value, candidate.name.lower()}:
                interval = RECRAWL_INTERVALS[candidate]
                break

    if signals.observed_change_interval:
        interval = min(
            interval, _ensure_positive(signals.observed_change_interval, interval)
        )

    if signals.consecutive_unchanged > 0:
        factor = 1.0 + min(5, signals.consecutive_unchanged) * 0.5
        interval = interval * factor

    if signals.last_modified and signals.last_crawled_at:
        freshness = signals.last_crawled_at - signals.last_modified
        if freshness > timedelta(0):
            scale = min(2.0, 1.0 + freshness.total_seconds() / interval.total_seconds())
            interval = interval * scale

    return interval


def _ensure_positive(candidate: timedelta, fallback: timedelta) -> timedelta:
    if candidate <= timedelta(0):
        return fallback
    return candidate


def _apply_host_rules(
    policy: HostPolitenessPolicy, state: HostVisitState, now: datetime
) -> Tuple[datetime, str]:
    earliest = now
    reason = "ready"

    if state.last_completed_at:
        last_gap = state.last_completed_at + policy.min_delay
        if last_gap > earliest:
            earliest = last_gap
            reason = "host_min_delay"

    if state.next_available_at:
        # host_next_available: honour scheduler-provided timestamp before new work
        if state.next_available_at > earliest:
            earliest = state.next_available_at
            reason = "host_next_available"

    if state.active_requests >= policy.max_parallelism:
        # host_parallel_limit: guard against exceeding configured in-flight slots
        if state.next_available_at and state.next_available_at > earliest:
            earliest = state.next_available_at
        elif policy.min_delay > timedelta(0):
            earliest = max(earliest, now + policy.min_delay)
        else:
            earliest = max(earliest, now + timedelta(seconds=1))
        reason = "host_parallel_limit"

    return earliest, reason
