from __future__ import annotations

from dataclasses import dataclass

from common.guardrails import FetcherLimits
from crawler.fetcher import FetchFailure, FetchRequest, FetchResult

from ai_core.contracts.payloads import FetchPayload, FetchRequestData
from ai_core.infra.observability import emit_event, record_span


@dataclass(frozen=True)
class FetchTelemetrySnapshot:
    used: bool
    http_status: int | None
    fetched_bytes: int | None
    media_type: str | None
    elapsed: float | None
    retries: int | None
    retry_reason: str | None
    backoff_total_ms: float | None

    def as_details(self) -> dict[str, object]:
        return {
            "fetch_used": self.used,
            "http_status": self.http_status,
            "fetched_bytes": self.fetched_bytes,
            "media_type_effective": self.media_type,
            "fetch_elapsed": self.elapsed,
            "fetch_retries": self.retries,
            "fetch_retry_reason": self.retry_reason,
            "fetch_backoff_total_ms": self.backoff_total_ms,
        }


def emit_fetch_started(origin: str, provider: str) -> None:
    emit_event("fetch_started", {"origin": origin, "provider": provider})


def record_fetch_attempt(fetch_result: FetchResult) -> None:
    telemetry = fetch_result.telemetry
    record_span(
        "crawler.fetch",
        attributes={
            "crawler.fetch.status": fetch_result.status.value,
            "crawler.fetch.bytes": telemetry.bytes_downloaded,
            "crawler.fetch.retry_reason": telemetry.retry_reason,
        },
    )
    emit_event(
        "fetch_finished",
        {
            "status": fetch_result.status.value,
            "status_code": fetch_result.metadata.status_code,
            "bytes": telemetry.bytes_downloaded,
        },
    )


def summarize_fetch_attempt(
    fetch_result: FetchResult, *, media_type: str | None
) -> FetchTelemetrySnapshot:
    telemetry = fetch_result.telemetry
    return FetchTelemetrySnapshot(
        used=True,
        http_status=fetch_result.metadata.status_code,
        fetched_bytes=telemetry.bytes_downloaded,
        media_type=media_type,
        elapsed=telemetry.latency,
        retries=telemetry.retries,
        retry_reason=telemetry.retry_reason,
        backoff_total_ms=telemetry.backoff_total_ms,
    )


def _build_header_mapping(fetch_result: FetchResult) -> dict[str, str]:
    headers: dict[str, str] = {}
    metadata = fetch_result.metadata
    if metadata.content_type:
        headers["Content-Type"] = metadata.content_type
    if metadata.etag:
        headers["ETag"] = metadata.etag
    if metadata.last_modified:
        headers["Last-Modified"] = metadata.last_modified
    if metadata.content_length is not None:
        headers["Content-Length"] = str(metadata.content_length)
    return headers


def build_fetch_payload(
    fetch_request: FetchRequest,
    fetch_result: FetchResult,
    *,
    fetch_limits: FetcherLimits | None = None,
    failure: FetchFailure | None = None,
    media_type: str | None = None,
) -> FetchPayload:
    payload_bytes = getattr(fetch_result, "payload", None)
    if payload_bytes is None:
        payload_bytes = getattr(fetch_result, "body", None)
    body_bytes: bytes = payload_bytes or b""
    telemetry = fetch_result.telemetry
    politeness = fetch_request.politeness
    request_payload = FetchRequestData(
        canonical_source=fetch_request.canonical_source,
        metadata=dict(fetch_request.metadata or {}),
        politeness_host=politeness.host if politeness else None,
        politeness_user_agent=politeness.user_agent if politeness else None,
        politeness_crawl_delay=politeness.crawl_delay if politeness else None,
    )
    timeout_seconds = None
    max_bytes_limit = None
    if fetch_limits is not None:
        max_bytes_limit = fetch_limits.max_bytes
        if fetch_limits.timeout is not None:
            timeout_seconds = fetch_limits.timeout.total_seconds()
    return FetchPayload(
        request=request_payload,
        status_code=fetch_result.metadata.status_code,
        body=body_bytes,
        headers=_build_header_mapping(fetch_result),
        elapsed_ms=telemetry.latency * 1000,
        retries=telemetry.retries,
        retry_reason=telemetry.retry_reason,
        downloaded_bytes=telemetry.bytes_downloaded,
        backoff_total_ms=telemetry.backoff_total_ms,
        max_bytes_limit=max_bytes_limit,
        timeout_seconds=timeout_seconds,
        failure_reason=failure.reason if failure else None,
        failure_temporary=failure.temporary if failure else None,
    )


def build_manual_fetch_payload(
    fetch_request: FetchRequest,
    *,
    body: bytes,
    media_type: str,
    elapsed: float = 0.05,
) -> FetchPayload:
    politeness = fetch_request.politeness
    request_payload = FetchRequestData(
        canonical_source=fetch_request.canonical_source,
        metadata=dict(fetch_request.metadata or {}),
        politeness_host=politeness.host if politeness else None,
        politeness_user_agent=politeness.user_agent if politeness else None,
        politeness_crawl_delay=politeness.crawl_delay if politeness else None,
    )
    return FetchPayload(
        request=request_payload,
        status_code=200,
        body=body,
        headers={"Content-Type": media_type},
        elapsed_ms=elapsed * 1000,
        retries=0,
        retry_reason=None,
        downloaded_bytes=len(body),
        backoff_total_ms=0.0,
        max_bytes_limit=None,
        timeout_seconds=None,
        failure_reason=None,
        failure_temporary=None,
    )


__all__ = [
    "FetchTelemetrySnapshot",
    "build_fetch_payload",
    "build_manual_fetch_payload",
    "emit_fetch_started",
    "record_fetch_attempt",
    "summarize_fetch_attempt",
]
