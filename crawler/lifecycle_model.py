"""Lifecycle status model to describe pipeline transitions and telemetry.

State chart::

    seeded -> queued -> (fetched | denied)
    fetched -> (parsed | unsupported | retired)
    denied -> (skipped | retired)
    parsed -> normalized -> delta:(new | changed | unchanged | near_duplicate)
    delta:new|changed -> (ingested | retired)
    delta:unchanged|near_duplicate -> (skipped | retired)

Terminal statuses are ``ingested``, ``skipped`` and ``retired``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional, Tuple


class LifecycleStatus(str, Enum):
    """Linearized lifecycle statuses emitted by the crawler pipeline."""

    SEEDED = "seeded"
    QUEUED = "queued"
    FETCHED = "fetched"
    DENIED = "denied"
    PARSED = "parsed"
    UNSUPPORTED = "unsupported"
    NORMALIZED = "normalized"
    DELTA_NEW = "delta:new"
    DELTA_CHANGED = "delta:changed"
    DELTA_UNCHANGED = "delta:unchanged"
    DELTA_NEAR_DUPLICATE = "delta:near_duplicate"
    INGESTED = "ingested"
    SKIPPED = "skipped"
    RETIRED = "retired"


FINAL_STATUSES = (
    LifecycleStatus.INGESTED,
    LifecycleStatus.SKIPPED,
    LifecycleStatus.RETIRED,
)


ALLOWED_TRANSITIONS = {
    LifecycleStatus.SEEDED: (LifecycleStatus.QUEUED,),
    LifecycleStatus.QUEUED: (LifecycleStatus.FETCHED, LifecycleStatus.DENIED),
    LifecycleStatus.FETCHED: (
        LifecycleStatus.PARSED,
        LifecycleStatus.UNSUPPORTED,
        LifecycleStatus.RETIRED,
    ),
    LifecycleStatus.DENIED: (LifecycleStatus.SKIPPED, LifecycleStatus.RETIRED),
    LifecycleStatus.PARSED: (LifecycleStatus.NORMALIZED,),
    LifecycleStatus.UNSUPPORTED: (LifecycleStatus.SKIPPED,),
    LifecycleStatus.NORMALIZED: (
        LifecycleStatus.DELTA_NEW,
        LifecycleStatus.DELTA_CHANGED,
        LifecycleStatus.DELTA_UNCHANGED,
        LifecycleStatus.DELTA_NEAR_DUPLICATE,
    ),
    LifecycleStatus.DELTA_NEW: (LifecycleStatus.INGESTED, LifecycleStatus.RETIRED),
    LifecycleStatus.DELTA_CHANGED: (LifecycleStatus.INGESTED, LifecycleStatus.RETIRED),
    LifecycleStatus.DELTA_UNCHANGED: (LifecycleStatus.SKIPPED, LifecycleStatus.RETIRED),
    LifecycleStatus.DELTA_NEAR_DUPLICATE: (
        LifecycleStatus.SKIPPED,
        LifecycleStatus.RETIRED,
    ),
    LifecycleStatus.INGESTED: (),
    LifecycleStatus.SKIPPED: (),
    LifecycleStatus.RETIRED: (),
}


class InvalidLifecycleTransition(ValueError):
    """Raised when a lifecycle status transition is not permitted."""


@dataclass(frozen=True)
class LifecycleEvent:
    """Lifecycle status annotated with minimal telemetry."""

    status: LifecycleStatus
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: Optional[float] = None
    attributes: Tuple[Tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        occurred = self.occurred_at
        if occurred.tzinfo is None:
            occurred = occurred.replace(tzinfo=timezone.utc)
        else:
            occurred = occurred.astimezone(timezone.utc)
        object.__setattr__(self, "occurred_at", occurred)

        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_negative")

        raw_attributes = self.attributes
        normalized: Tuple[Tuple[str, object], ...] = ()
        if raw_attributes:
            if isinstance(raw_attributes, Mapping):
                items = raw_attributes.items()
            elif isinstance(raw_attributes, tuple):
                items = raw_attributes
            else:
                raise TypeError("attributes_must_be_mapping_or_tuple")
            normalized = tuple(
                (str(key), value)
                for key, value in sorted(items, key=lambda item: str(item[0]))
            )
        object.__setattr__(self, "attributes", normalized)

    @classmethod
    def from_mapping(
        cls,
        status: LifecycleStatus,
        *,
        occurred_at: Optional[datetime] = None,
        duration_ms: Optional[float] = None,
        attributes: Optional[Mapping[str, object]] = None,
    ) -> "LifecycleEvent":
        """Factory helper accepting attribute mappings."""

        attr_items: Tuple[Tuple[str, object], ...] = ()
        if attributes:
            attr_items = tuple(
                sorted((str(key), value) for key, value in attributes.items())
            )
        return cls(
            status=status,
            occurred_at=occurred_at or datetime.now(timezone.utc),
            duration_ms=duration_ms,
            attributes=attr_items,
        )

    def attributes_dict(self) -> Mapping[str, object]:
        """Return the event attributes as a mapping."""

        return dict(self.attributes)


@dataclass(frozen=True)
class LifecycleTimeline:
    """Immutable lifecycle timeline enforcing allowed status transitions."""

    events: Tuple[LifecycleEvent, ...] = ()

    def advance(
        self,
        status: LifecycleStatus,
        *,
        occurred_at: Optional[datetime] = None,
        duration_ms: Optional[float] = None,
        attributes: Optional[Mapping[str, object]] = None,
    ) -> "LifecycleTimeline":
        """Return a new timeline extended with the provided lifecycle status."""

        event = LifecycleEvent.from_mapping(
            status,
            occurred_at=occurred_at,
            duration_ms=duration_ms,
            attributes=attributes,
        )
        if not self.events:
            if status is not LifecycleStatus.SEEDED:
                raise InvalidLifecycleTransition("timeline_must_start_with_seeded")
            return LifecycleTimeline(events=(event,))

        previous = self.events[-1].status
        allowed = ALLOWED_TRANSITIONS.get(previous, ())
        if status not in allowed:
            raise InvalidLifecycleTransition(
                f"invalid_transition:{previous.value}->{status.value}"
            )
        return LifecycleTimeline(events=self.events + (event,))

    @property
    def current_status(self) -> LifecycleStatus:
        """Return the last recorded lifecycle status."""

        if not self.events:
            raise InvalidLifecycleTransition("timeline_empty")
        return self.events[-1].status

    @property
    def is_complete(self) -> bool:
        """Return ``True`` when the lifecycle reached a terminal status."""

        if not self.events:
            return False
        return self.events[-1].status in FINAL_STATUSES


__all__ = [
    "ALLOWED_TRANSITIONS",
    "FINAL_STATUSES",
    "InvalidLifecycleTransition",
    "LifecycleEvent",
    "LifecycleStatus",
    "LifecycleTimeline",
]
