"""HTTP protocol handlers for document streaming.

Pure functions and classes for HTTP-level concerns,
isolated from Django views and business logic.
"""

import re
import email.utils as email_utils
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RangeRequest:
    """Parsed HTTP Range header representation."""

    start: int
    end: int
    total_size: int

    @property
    def length(self) -> int:
        """Content length for this range."""
        return self.end - self.start + 1

    @property
    def content_range_header(self) -> str:
        """RFC 7233 Content-Range header value."""
        return f"bytes {self.start}-{self.end}/{self.total_size}"

    def is_valid(self) -> bool:
        """Check if range is within bounds."""
        return (
            0 <= self.start < self.total_size
            and self.start <= self.end < self.total_size
        )


@dataclass(frozen=True)
class CacheMetadata:
    """File metadata for HTTP caching."""

    etag: str
    last_modified: str
    last_modified_timestamp: float

    @classmethod
    def from_file_stats(cls, file_size: int, mtime: float) -> "CacheMetadata":
        """Generate cache metadata from file stats.

        Args:
            file_size: File size in bytes
            mtime: Modification time (POSIX timestamp)

        Returns:
            CacheMetadata with weak ETag and RFC 2822 Last-Modified
        """
        weak_etag = f'W/"{file_size:x}-{int(mtime):x}"'
        last_modified = email_utils.formatdate(mtime, usegmt=True)
        return cls(
            etag=weak_etag,
            last_modified=last_modified,
            last_modified_timestamp=mtime,
        )


class HttpRangeHandler:
    """Parser and validator for HTTP Range requests (RFC 7233).

    Pure class with no external dependencies - testable without Django or filesystem.
    """

    _RANGE_REGEX = re.compile(r"bytes=(\d*)-(\d*)")

    @classmethod
    def parse(cls, range_header: str, file_size: int) -> Optional[RangeRequest]:
        """Parse Range header and validate bounds.

        Args:
            range_header: Raw Range header value (e.g., "bytes=100-200" or "bytes=-500")
            file_size: Total file size in bytes

        Returns:
            RangeRequest if valid, None if malformed or unsatisfiable

        Examples:
            >>> HttpRangeHandler.parse("bytes=0-99", 1000)
            RangeRequest(start=0, end=99, total_size=1000)

            >>> HttpRangeHandler.parse("bytes=-500", 1000)  # Suffix range
            RangeRequest(start=500, end=999, total_size=1000)

            >>> HttpRangeHandler.parse("bytes=2000-3000", 1000)  # Out of bounds
            None
        """
        match = cls._RANGE_REGEX.match(range_header)
        if not match:
            return None

        start_str, end_str = match.groups()

        # Suffix range (bytes=-N): last N bytes
        if not start_str and end_str:
            length = int(end_str)
            start = max(file_size - length, 0)
            end = file_size - 1
        # Normal range (bytes=M-N or bytes=M-)
        else:
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1

        # Validate and cap bounds
        if start < 0 or start >= file_size:
            return None

        end = min(end, file_size - 1)

        return RangeRequest(start=start, end=end, total_size=file_size)


class CacheControlStrategy:
    """HTTP caching logic for conditional requests (RFC 7232).

    Handles ETag generation and If-None-Match / If-Modified-Since validation.
    """

    @staticmethod
    def should_return_304(
        cache_meta: CacheMetadata,
        if_none_match: Optional[str] = None,
        if_modified_since: Optional[str] = None,
    ) -> bool:
        """Check if 304 Not Modified should be returned.

        Args:
            cache_meta: Current file cache metadata
            if_none_match: If-None-Match header (comma-separated ETags)
            if_modified_since: If-Modified-Since header (RFC 2822 date)

        Returns:
            True if response should be 304 Not Modified

        Priority: If-None-Match takes precedence over If-Modified-Since (RFC 7232 §3.2)
        """
        # If-None-Match can contain multiple ETags
        if if_none_match:
            tags = [tag.strip() for tag in if_none_match.split(",")]
            if cache_meta.etag in tags:
                return True

        # If-Modified-Since check
        if if_modified_since:
            try:
                ims_dt = email_utils.parsedate_to_datetime(if_modified_since)
                if ims_dt.timestamp() >= cache_meta.last_modified_timestamp:
                    return True
            except (ValueError, TypeError):
                pass  # Ignore malformed dates

        return False

    @staticmethod
    def cache_headers() -> dict[str, str]:
        """Standard cache headers for document responses."""
        return {
            "Cache-Control": "private, max-age=3600",
            "Vary": "Authorization, Cookie",
            "Accept-Ranges": "bytes",
        }
