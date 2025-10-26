"""Lifecycle evaluation for soft-deleting and retiring crawler documents.

Retirement marks a source as inactive but does not permanently lock it. A
subsequent successful fetch may reactivate the document, which is intentional
so that operators can "revive" content if a previously missing page returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Tuple

from .fetcher import FetchResult, FetchStatus


class LifecycleState(str, Enum):
    """High-level lifecycle states for crawler-managed resources."""

    ACTIVE = "active"
    RETIRED = "retired"
    DELETED = "deleted"


@dataclass(frozen=True)
class RetirePolicy:
    """Configuration for retiring sources after repeated absence signals."""

    consecutive_not_found_threshold: int = 3
    not_found_interval: timedelta = timedelta(days=7)

    def __post_init__(self) -> None:
        if self.consecutive_not_found_threshold <= 0:
            raise ValueError("consecutive_not_found_threshold_invalid")
        if self.not_found_interval <= timedelta(0):
            raise ValueError("not_found_interval_invalid")


@dataclass(frozen=True)
class RetireSignals:
    """Historical lifecycle hints gathered from previous crawl attempts."""

    consecutive_not_found: int = 0
    first_not_found_at: Optional[datetime] = None
    manual_state: Optional[LifecycleState] = None
    manual_reason: Optional[str] = None
    permanent_redirect_target: Optional[str] = None

    def __post_init__(self) -> None:
        if self.consecutive_not_found < 0:
            raise ValueError("consecutive_not_found_negative")
        if self.first_not_found_at is not None:
            object.__setattr__(
                self,
                "first_not_found_at",
                _ensure_aware(self.first_not_found_at),
            )
        if self.manual_reason is not None:
            object.__setattr__(self, "manual_reason", self.manual_reason.strip())


@dataclass(frozen=True)
class LifecycleDecision:
    """Lifecycle outcome emitted after evaluating retire signals."""

    state: LifecycleState
    reason: str
    policy_events: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_reason = (self.reason or "").strip()
        if not normalized_reason:
            raise ValueError("lifecycle_reason_required")
        object.__setattr__(self, "reason", normalized_reason)
        object.__setattr__(self, "policy_events", tuple(self.policy_events))

    @property
    def should_retire(self) -> bool:
        """Return ``True`` when the lifecycle state requires retirement."""

        return self.state in {LifecycleState.RETIRED, LifecycleState.DELETED}


def evaluate_lifecycle(
    *,
    fetch: Optional[FetchResult],
    signals: Optional[RetireSignals] = None,
    policy: Optional[RetirePolicy] = None,
    now: Optional[datetime] = None,
) -> LifecycleDecision:
    """Determine whether a document should remain active or be retired.

    The evaluation intentionally keeps retirement reversible. If later fetches
    succeed, callers may treat the document as active again without additional
    coordination, mirroring the "revive" semantics documented in the crawler
    lifecycle guide.
    """

    current_time = _ensure_aware(now)
    applied_policy = policy or RetirePolicy()
    applied_signals = signals or RetireSignals()

    manual_state = applied_signals.manual_state
    if manual_state is not None:
        event = "manual_policy"
        if manual_state is LifecycleState.RETIRED:
            event = "manual_retire"
        elif manual_state is LifecycleState.DELETED:
            event = "manual_delete"
        reason = applied_signals.manual_reason or f"manual_{manual_state.value}"
        return LifecycleDecision(manual_state, reason, (event,))

    redirect_target = applied_signals.permanent_redirect_target
    if redirect_target:
        reason = f"permanent_redirect:{redirect_target}"
        return LifecycleDecision(
            LifecycleState.RETIRED,
            reason,
            ("permanent_redirect",),
        )

    if _should_retire_for_not_found(applied_signals, applied_policy, current_time):
        reason = f"not_found_streak:{applied_signals.consecutive_not_found}"
        return LifecycleDecision(
            LifecycleState.RETIRED,
            reason,
            ("not_found_streak",),
        )

    if fetch is None:
        return LifecycleDecision(LifecycleState.ACTIVE, "active")

    status_code = _extract_status_code(fetch)

    if fetch.status is FetchStatus.GONE:
        if status_code == 410:
            return LifecycleDecision(
                LifecycleState.RETIRED,
                "gone_410",
                ("gone_410",),
            )
        if status_code == 404:
            streak = applied_signals.consecutive_not_found + 1
            first_seen = applied_signals.first_not_found_at or current_time
            if _streak_exceeds_policy(streak, first_seen, applied_policy, current_time):
                reason = f"not_found_streak:{streak}"
                return LifecycleDecision(
                    LifecycleState.RETIRED,
                    reason,
                    ("not_found_streak",),
                )
            return LifecycleDecision(LifecycleState.ACTIVE, "not_found")
        return LifecycleDecision(
            LifecycleState.RETIRED,
            fetch.detail or "gone",
            ("permanent_failure",),
        )

    if fetch.status is FetchStatus.TEMPORARY_ERROR and status_code in {301, 308}:
        reason = (
            f"permanent_redirect:{status_code}" if status_code else "permanent_redirect"
        )
        return LifecycleDecision(
            LifecycleState.RETIRED,
            reason,
            ("permanent_redirect",),
        )

    return LifecycleDecision(LifecycleState.ACTIVE, "active")


def _should_retire_for_not_found(
    signals: RetireSignals,
    policy: RetirePolicy,
    now: datetime,
) -> bool:
    if signals.consecutive_not_found < policy.consecutive_not_found_threshold:
        return False
    if signals.first_not_found_at is None:
        return False
    return (now - signals.first_not_found_at) >= policy.not_found_interval


def _streak_exceeds_policy(
    streak: int,
    first_seen: datetime,
    policy: RetirePolicy,
    now: datetime,
) -> bool:
    if streak < policy.consecutive_not_found_threshold:
        return False
    return (now - first_seen) >= policy.not_found_interval


def _extract_status_code(result: FetchResult) -> Optional[int]:
    status_code = result.metadata.status_code
    if status_code is not None:
        return status_code
    detail = result.detail or ""
    if detail.startswith("status_"):
        try:
            return int(detail.split("_", 1)[1])
        except ValueError:
            return None
    return None


def _ensure_aware(candidate: Optional[datetime]) -> datetime:
    if candidate is None:
        return datetime.now(timezone.utc)
    if candidate.tzinfo is None:
        return candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


__all__ = [
    "LifecycleDecision",
    "LifecycleState",
    "RetirePolicy",
    "RetireSignals",
    "evaluate_lifecycle",
]
