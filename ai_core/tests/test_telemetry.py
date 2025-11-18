from __future__ import annotations

from crawler.fetcher import (
    FetchFailure,
    FetchMetadata,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetchTelemetry,
    PolitenessContext,
)
from common.guardrails import FetcherLimits

from ai_core.telemetry.crawler import (
    build_fetch_payload,
    build_manual_fetch_payload,
    summarize_fetch_attempt,
)


def _build_fetch_result(
    payload: bytes,
    *,
    status_code: int = 200,
    status: FetchStatus = FetchStatus.FETCHED,
    retries: int = 0,
    retry_reason: str | None = None,
) -> tuple[FetchRequest, FetchResult]:
    request = FetchRequest(
        canonical_source="https://example.org/resource",
        politeness=PolitenessContext(host="example.org"),
    )
    metadata = FetchMetadata(
        status_code=status_code,
        content_type="text/plain",
        etag="etag-123",
        last_modified="Thu, 21 Dec 2000 16:01:07 GMT",
        content_length=len(payload),
    )
    telemetry = FetchTelemetry(
        latency=0.42,
        bytes_downloaded=len(payload),
        attempt=retries + 1,
        retries=retries,
        retry_reason=retry_reason,
        backoff_total_ms=12.0,
    )
    result = FetchResult(
        status=status,
        request=request,
        payload=payload,
        metadata=metadata,
        telemetry=telemetry,
    )
    return request, result


def test_build_fetch_payload_serializes_limits_and_failures():
    request, result = _build_fetch_result(
        b"payload-bytes", retries=1, retry_reason="retry"
    )
    limits = FetcherLimits(max_bytes=1024)
    failure = FetchFailure(reason="timeout", temporary=True)

    fetch_input, snapshot, body_bytes, etag = build_fetch_payload(
        request,
        result,
        fetch_limits=limits,
        failure=failure,
        media_type="text/plain",
    )

    assert snapshot.used is True
    assert snapshot.http_status == 200
    assert snapshot.media_type == "text/plain"
    assert snapshot.fetched_bytes == len(body_bytes) == len("payload-bytes")
    assert snapshot.retries == 1
    assert snapshot.retry_reason == "retry"
    assert snapshot.backoff_total_ms == 12.0

    assert fetch_input["limits"]["max_bytes"] == 1024
    assert fetch_input["failure"]["reason"] == "timeout"
    assert fetch_input["headers"]["Content-Type"] == "text/plain"
    assert fetch_input["request"]["canonical_source"] == request.canonical_source
    assert etag == "etag-123"


def test_build_manual_fetch_payload_defaults_elapsed():
    request = FetchRequest(
        canonical_source="https://example.org/manual",
        politeness=PolitenessContext(host="example.org"),
    )

    fetch_input, snapshot = build_manual_fetch_payload(
        request,
        body=b"manual-bytes",
        media_type="application/octet-stream",
    )

    assert snapshot.used is False
    assert snapshot.http_status == 200
    assert snapshot.media_type == "application/octet-stream"
    assert snapshot.fetched_bytes == len("manual-bytes")
    assert fetch_input["headers"]["Content-Type"] == "application/octet-stream"
    assert fetch_input["body"] == b"manual-bytes"


def test_summarize_fetch_attempt_uses_result_telemetry():
    request, result = _build_fetch_result(
        b"denied",
        status=FetchStatus.TEMPORARY_ERROR,
        status_code=503,
        retry_reason="upstream",
    )

    snapshot = summarize_fetch_attempt(result, media_type="text/plain")
    details = snapshot.as_details()

    assert snapshot.used is True
    assert details["http_status"] == 503
    assert details["fetch_retry_reason"] == "upstream"
    assert details["fetched_bytes"] == len("denied")
