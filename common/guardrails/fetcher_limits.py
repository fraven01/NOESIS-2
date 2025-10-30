"""Shared fetcher limit contracts used by crawler and AI-Core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class FetcherLimits:
    """Configurable fetcher limits that enforce security constraints."""

    max_bytes: Optional[int] = None
    timeout: Optional[timedelta] = None
    mime_whitelist: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("max_bytes_invalid")
        if self.timeout is not None and self.timeout <= timedelta(0):
            raise ValueError("timeout_invalid")
        if self.mime_whitelist is not None:
            if not self.mime_whitelist:
                raise ValueError("mime_whitelist_empty")
            object.__setattr__(
                self,
                "mime_whitelist",
                tuple(entry.strip().lower() for entry in self.mime_whitelist),
            )

    def enforce(self, metadata: Any, telemetry: Any) -> Tuple[bool, Tuple[str, ...]]:
        """Return whether the fetch obeyed limits and any violation reasons."""

        violations: list[str] = []
        bytes_downloaded = _coerce_int(getattr(telemetry, "bytes_downloaded", 0))
        if self.max_bytes is not None and bytes_downloaded > self.max_bytes:
            violations.append("max_bytes_exceeded")

        latency = getattr(telemetry, "latency", None)
        if self.timeout is not None and latency is not None:
            try:
                latency_value = float(latency)
            except (TypeError, ValueError):
                latency_value = None
            if (
                latency_value is not None
                and latency_value > self.timeout.total_seconds()
            ):
                violations.append("timeout_exceeded")

        if self.mime_whitelist is not None:
            content_type = getattr(metadata, "content_type", None)
            if isinstance(content_type, str) and content_type:
                normalized = content_type.lower()
                if not _mime_matches_whitelist(normalized, self.mime_whitelist):
                    violations.append("mime_not_allowed")

        return (not violations, tuple(violations))


def _mime_matches_whitelist(content_type: str, whitelist: Tuple[str, ...]) -> bool:
    for entry in whitelist:
        if not entry:
            continue
        if entry.endswith("/*"):
            prefix = entry[:-1]
            if content_type.startswith(prefix):
                return True
        elif content_type == entry:
            return True
    return False


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["FetcherLimits"]
