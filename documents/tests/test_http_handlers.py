"""Unit tests for HTTP protocol helpers in `documents.http_handlers`."""

from datetime import datetime, timezone

import pytest

from documents.http_handlers import (
    CacheControlStrategy,
    CacheMetadata,
    HttpRangeHandler,
)


def test_parse_normal_range():
    """Normal range headers are parsed with start/end/length."""
    result = HttpRangeHandler.parse("bytes=100-200", file_size=1000)

    assert result is not None
    assert result.start == 100
    assert result.end == 200
    assert result.length == 101
    assert result.is_valid()
    assert result.content_range_header == "bytes 100-200/1000"


def test_parse_suffix_range():
    """Suffix ranges return the last N bytes of the file."""
    result = HttpRangeHandler.parse("bytes=-250", file_size=1000)

    assert result is not None
    assert result.start == 750
    assert result.end == 999
    assert result.length == 250


def test_parse_open_ended_range():
    """Open-ended ranges run until EOF."""
    result = HttpRangeHandler.parse("bytes=900-", file_size=1000)

    assert result is not None
    assert result.start == 900
    assert result.end == 999


def test_parse_out_of_bounds_returns_none():
    """Ranges that start beyond the file size should be unsatisfiable."""
    assert HttpRangeHandler.parse("bytes=2000-3000", file_size=1000) is None


@pytest.mark.parametrize(
    "header",
    [
        "bytes=invalid",
        "chunks=0-100",
        "",
    ],
)
def test_parse_malformed_headers(header):
    """Malformed headers must return None."""
    assert HttpRangeHandler.parse(header, file_size=1000) is None


def test_range_request_detects_invalid_length():
    """Start greater than end should still return an invalid RangeRequest."""
    result = HttpRangeHandler.parse("bytes=100-99", file_size=1000)

    assert result is not None
    assert not result.is_valid()


def test_cache_metadata_from_file_stats():
    """Cache metadata derives a weak ETag and RFC 2822 Last-Modified."""
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    cache_meta = CacheMetadata.from_file_stats(file_size=1024, mtime=timestamp)

    assert cache_meta.etag.startswith('W/"')
    assert cache_meta.etag.endswith('"')
    assert "GMT" in cache_meta.last_modified
    assert cache_meta.last_modified_timestamp == pytest.approx(timestamp)


def test_cache_strategy_should_return_304_on_etag_match():
    """ETag match should trigger a 304 response."""
    cache_meta = CacheMetadata.from_file_stats(file_size=3, mtime=0)

    assert CacheControlStrategy.should_return_304(
        cache_meta,
        if_none_match=cache_meta.etag,
    )


def test_cache_strategy_should_return_304_on_if_modified_since():
    """If-Modified-Since greater than or equal to metadata should trigger 304."""
    cache_meta = CacheMetadata.from_file_stats(
        file_size=1,
        mtime=1700000000.0,
    )

    assert CacheControlStrategy.should_return_304(
        cache_meta,
        if_modified_since=cache_meta.last_modified,
    )


def test_cache_strategy_cache_headers():
    """Cache headers expose the expected keys for downstream consumers."""
    headers = CacheControlStrategy.cache_headers()

    assert headers["Cache-Control"] == "private, max-age=3600"
    assert headers["Accept-Ranges"] == "bytes"
    assert "Vary" in headers
