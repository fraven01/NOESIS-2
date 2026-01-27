from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4


class TimeSource(Protocol):
    def now_iso(self) -> str: ...


class EventIdSource(Protocol):
    def next_event_id(self) -> str: ...


@dataclass(frozen=True)
class SystemTimeSource:
    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UUIDEventIdSource:
    def next_event_id(self) -> str:
        return str(uuid4())


__all__ = [
    "TimeSource",
    "EventIdSource",
    "SystemTimeSource",
    "UUIDEventIdSource",
]
