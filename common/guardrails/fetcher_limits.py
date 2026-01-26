"""Shared fetcher limit contracts used by crawler and AI-Core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Optional, Sequence, Tuple


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

    @classmethod
    def from_dict(
        cls, mapping: Mapping[str, Any], *, context: str = "fetcher_limits"
    ) -> "FetcherLimits | None":
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{context}_must_be_mapping")
        allowed_keys = {"max_bytes", "timeout_seconds", "mime_whitelist"}
        invalid = sorted(set(mapping.keys()) - allowed_keys)
        if invalid:
            raise ValueError(f"{context}.unknown_keys:{','.join(invalid)}")

        kwargs: dict[str, Any] = {}
        has_value = False
        if "max_bytes" in mapping:
            max_bytes = mapping["max_bytes"]
            if not isinstance(max_bytes, int):
                raise TypeError(f"{context}.max_bytes_must_be_int")
            if max_bytes <= 0:
                raise ValueError(f"{context}.max_bytes_positive")
            kwargs["max_bytes"] = max_bytes
            has_value = True

        if "timeout_seconds" in mapping:
            timeout_seconds = mapping["timeout_seconds"]
            if not isinstance(timeout_seconds, (int, float)):
                raise TypeError(f"{context}.timeout_seconds_must_be_number")
            if timeout_seconds <= 0:
                raise ValueError(f"{context}.timeout_seconds_positive")
            kwargs["timeout"] = timedelta(seconds=float(timeout_seconds))
            has_value = True

        if "mime_whitelist" in mapping:
            mime_value = mapping["mime_whitelist"]
            if not isinstance(mime_value, Sequence) or isinstance(mime_value, str):
                raise TypeError(f"{context}.mime_whitelist_sequence")
            mime_items = []
            for entry in mime_value:
                if not isinstance(entry, str):
                    raise TypeError(f"{context}.mime_whitelist_string")
                candidate = entry.strip().lower()
                if not candidate:
                    raise ValueError(f"{context}.mime_whitelist_empty_entry")
                mime_items.append(candidate)
            if not mime_items:
                raise ValueError(f"{context}.mime_whitelist_empty")
            kwargs["mime_whitelist"] = tuple(mime_items)
            has_value = True

        if not has_value:
            return None
        return cls(**kwargs)

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
