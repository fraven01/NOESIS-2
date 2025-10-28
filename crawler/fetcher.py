"""Fetcher contract describing result semantics and telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple

from ai_core.rag.guardrails import FetcherLimits

from .errors import CrawlerError, ErrorClass


class FetchStatus(str, Enum):
    """Supported fetcher outcomes."""

    FETCHED = "fetched"
    NOT_MODIFIED = "not_modified"
    GONE = "gone"
    TEMPORARY_ERROR = "temporary_error"
    POLICY_DENIED = "policy_denied"


@dataclass(frozen=True)
class PolitenessContext:
    """Runtime politeness context for a fetch request."""

    host: str
    slot: Optional[str] = None
    user_agent: Optional[str] = None
    crawl_delay: Optional[float] = None

    def __post_init__(self) -> None:
        normalized_host = (self.host or "").strip().lower()
        if not normalized_host:
            raise ValueError("host_required")
        if self.crawl_delay is not None and self.crawl_delay < 0:
            raise ValueError("crawl_delay_negative")
        object.__setattr__(self, "host", normalized_host)
        if self.slot is not None:
            object.__setattr__(self, "slot", self.slot.strip())
        if self.user_agent is not None:
            object.__setattr__(self, "user_agent", self.user_agent.strip())


@dataclass(frozen=True)
class FetchRequest:
    """Canonical source and politeness data passed to the fetcher."""

    canonical_source: str
    politeness: PolitenessContext
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = (self.canonical_source or "").strip()
        if not canonical:
            raise ValueError("canonical_source_required")
        object.__setattr__(self, "canonical_source", canonical)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata_must_be_mapping")


@dataclass(frozen=True)
class FetchMetadata:
    """HTTP metadata captured alongside the fetch result."""

    status_code: Optional[int]
    content_type: Optional[str]
    etag: Optional[str]
    last_modified: Optional[str]
    content_length: Optional[int]


@dataclass(frozen=True)
class FetchTelemetry:
    """Telemetry for latency, retries, backoff, and retry triggering reasons."""

    latency: Optional[float]
    bytes_downloaded: int
    attempt: int = 1
    retries: int = 0
    retry_reason: Optional[str] = None
    backoff_total_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.latency is not None and self.latency < 0:
            raise ValueError("latency_negative")
        if self.bytes_downloaded < 0:
            raise ValueError("bytes_negative")
        if self.attempt < 1:
            raise ValueError("attempt_invalid")
        if self.retries < 0:
            raise ValueError("retries_negative")
        if self.backoff_total_ms < 0:
            raise ValueError("backoff_negative")


@dataclass(frozen=True)
class FetchResult:
    """Structured fetcher result exposed to downstream ingestion."""

    status: FetchStatus
    request: FetchRequest
    payload: Optional[bytes]
    metadata: FetchMetadata
    telemetry: FetchTelemetry
    detail: Optional[str] = None
    policy_events: Tuple[str, ...] = ()
    error: Optional[CrawlerError] = None


@dataclass(frozen=True)
class FetchFailure:
    """Non-HTTP failure details captured by the fetcher."""

    reason: str
    temporary: bool = True

    def __post_init__(self) -> None:
        normalized = (self.reason or "").strip()
        if not normalized:
            raise ValueError("reason_required")
        object.__setattr__(self, "reason", normalized)


def evaluate_fetch_response(
    request: FetchRequest,
    *,
    status_code: Optional[int],
    body: Optional[bytes],
    headers: Optional[Mapping[str, str]] = None,
    elapsed: Optional[float] = None,
    retries: int = 0,
    limits: Optional[FetcherLimits] = None,
    failure: Optional[FetchFailure] = None,
    retry_reason: Optional[str] = None,
    downloaded_bytes: Optional[int] = None,
    backoff_total_ms: float = 0.0,
) -> FetchResult:
    """Compose a :class:`FetchResult` from HTTP metadata and limits."""

    normalized_headers = _normalize_headers(headers)
    content_type = normalized_headers.get("content-type")
    etag = normalized_headers.get("etag")
    last_modified = normalized_headers.get("last-modified")
    payload = body if body is not None else None
    content_length = len(payload) if payload is not None else None
    if downloaded_bytes is not None:
        bytes_downloaded = downloaded_bytes
    else:
        bytes_downloaded = len(body) if body is not None else 0

    effective_retry_reason = retry_reason
    if effective_retry_reason is None and failure is not None:
        effective_retry_reason = failure.reason
    if (
        effective_retry_reason is None
        and status_code is not None
        and status_code in {429, 500, 502, 503, 504}
    ):
        effective_retry_reason = f"status_{status_code}"

    telemetry = FetchTelemetry(
        latency=elapsed,
        bytes_downloaded=bytes_downloaded,
        attempt=retries + 1,
        retries=retries,
        retry_reason=effective_retry_reason,
        backoff_total_ms=backoff_total_ms,
    )

    metadata = FetchMetadata(
        status_code=status_code,
        content_type=content_type,
        etag=etag,
        last_modified=last_modified,
        content_length=content_length,
    )

    policy_events: Tuple[str, ...] = ()
    if limits is not None:
        allowed, violations = limits.enforce(metadata, telemetry)
        if not allowed:
            trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
            detail = ",".join(violations) or None
            return FetchResult(
                status=FetchStatus.POLICY_DENIED,
                request=request,
                payload=None,
                metadata=trimmed_metadata,
                telemetry=trimmed_telemetry,
                detail=detail,
                policy_events=violations,
                error=_build_fetch_error(
                    status=FetchStatus.POLICY_DENIED,
                    request=request,
                    metadata=trimmed_metadata,
                    detail=detail,
                    policy_events=violations,
                    failure=failure,
                ),
            )

    if status_code is None:
        return _result_from_failure(request, metadata, telemetry, failure)

    if status_code == 304:
        trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
        return FetchResult(
            status=FetchStatus.NOT_MODIFIED,
            request=request,
            payload=None,
            metadata=trimmed_metadata,
            telemetry=trimmed_telemetry,
            detail="not_modified",
            policy_events=policy_events,
            error=None,
        )

    if status_code in {404, 410}:
        trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
        detail = f"status_{status_code}"
        return FetchResult(
            status=FetchStatus.GONE,
            request=request,
            payload=None,
            metadata=trimmed_metadata,
            telemetry=trimmed_telemetry,
            detail=detail,
            policy_events=policy_events,
            error=_build_fetch_error(
                status=FetchStatus.GONE,
                request=request,
                metadata=trimmed_metadata,
                detail=detail,
                policy_events=policy_events,
                failure=failure,
            ),
        )

    if 200 <= status_code <= 299:
        return FetchResult(
            status=FetchStatus.FETCHED,
            request=request,
            payload=payload,
            metadata=metadata,
            telemetry=telemetry,
            detail="ok",
            policy_events=policy_events,
            error=None,
        )

    if status_code in {401, 403, 451}:
        trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
        detail = f"status_{status_code}"
        return FetchResult(
            status=FetchStatus.POLICY_DENIED,
            request=request,
            payload=None,
            metadata=trimmed_metadata,
            telemetry=trimmed_telemetry,
            detail=detail,
            policy_events=("http_policy",),
            error=_build_fetch_error(
                status=FetchStatus.POLICY_DENIED,
                request=request,
                metadata=trimmed_metadata,
                detail=detail,
                policy_events=("http_policy",),
                failure=failure,
            ),
        )

    if status_code in {429, 500, 502, 503, 504}:
        trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
        detail = f"status_{status_code}"
        return FetchResult(
            status=FetchStatus.TEMPORARY_ERROR,
            request=request,
            payload=None,
            metadata=trimmed_metadata,
            telemetry=trimmed_telemetry,
            detail=detail,
            policy_events=policy_events,
            error=_build_fetch_error(
                status=FetchStatus.TEMPORARY_ERROR,
                request=request,
                metadata=trimmed_metadata,
                detail=detail,
                policy_events=policy_events,
                failure=failure,
            ),
        )

    if failure is not None and not failure.temporary:
        trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
        detail = failure.reason
        return FetchResult(
            status=FetchStatus.GONE,
            request=request,
            payload=None,
            metadata=trimmed_metadata,
            telemetry=trimmed_telemetry,
            detail=detail,
            policy_events=policy_events,
            error=_build_fetch_error(
                status=FetchStatus.GONE,
                request=request,
                metadata=trimmed_metadata,
                detail=detail,
                policy_events=policy_events,
                failure=failure,
            ),
        )

    trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
    detail = f"status_{status_code}"
    return FetchResult(
        status=FetchStatus.TEMPORARY_ERROR,
        request=request,
        payload=None,
        metadata=trimmed_metadata,
        telemetry=trimmed_telemetry,
        detail=detail,
        policy_events=policy_events,
        error=_build_fetch_error(
            status=FetchStatus.TEMPORARY_ERROR,
            request=request,
            metadata=trimmed_metadata,
            detail=detail,
            policy_events=policy_events,
            failure=failure,
        ),
    )


def _result_from_failure(
    request: FetchRequest,
    metadata: FetchMetadata,
    telemetry: FetchTelemetry,
    failure: Optional[FetchFailure],
) -> FetchResult:
    if failure is None or failure.temporary:
        detail = failure.reason if failure else "failure"
        trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
        return FetchResult(
            status=FetchStatus.TEMPORARY_ERROR,
            request=request,
            payload=None,
            metadata=trimmed_metadata,
            telemetry=trimmed_telemetry,
            detail=detail,
            error=_build_fetch_error(
                status=FetchStatus.TEMPORARY_ERROR,
                request=request,
                metadata=trimmed_metadata,
                detail=detail,
                policy_events=(),
                failure=failure,
            ),
        )
    trimmed_metadata, trimmed_telemetry = _drop_payload(metadata, telemetry)
    return FetchResult(
        status=FetchStatus.GONE,
        request=request,
        payload=None,
        metadata=trimmed_metadata,
        telemetry=trimmed_telemetry,
        detail=failure.reason,
        error=_build_fetch_error(
            status=FetchStatus.GONE,
            request=request,
            metadata=trimmed_metadata,
            detail=failure.reason,
            policy_events=(),
            failure=failure,
        ),
    )


def _normalize_headers(headers: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    if not headers:
        return {}
    normalized = {}
    for key, value in headers.items():
        if key is None:
            continue
        normalized[key.lower()] = value
    return normalized


def _mime_matches_whitelist(content_type: str, whitelist: Sequence[str]) -> bool:
    for allowed in whitelist:
        if not allowed:
            continue
        if allowed.endswith("/*"):
            prefix = allowed[:-1]
            if content_type.startswith(prefix):
                return True
        elif content_type == allowed:
            return True
    return False


def _drop_payload(
    metadata: FetchMetadata, telemetry: FetchTelemetry
) -> Tuple[FetchMetadata, FetchTelemetry]:
    trimmed_metadata = FetchMetadata(
        status_code=metadata.status_code,
        content_type=metadata.content_type,
        etag=metadata.etag,
        last_modified=metadata.last_modified,
        content_length=None,
    )
    trimmed_telemetry = FetchTelemetry(
        latency=telemetry.latency,
        bytes_downloaded=0,
        attempt=telemetry.attempt,
        retries=telemetry.retries,
        retry_reason=telemetry.retry_reason,
        backoff_total_ms=telemetry.backoff_total_ms,
    )
    return trimmed_metadata, trimmed_telemetry


def _build_fetch_error(
    *,
    status: FetchStatus,
    request: FetchRequest,
    metadata: FetchMetadata,
    detail: Optional[str],
    policy_events: Tuple[str, ...],
    failure: Optional[FetchFailure],
) -> Optional[CrawlerError]:
    if status in {FetchStatus.FETCHED, FetchStatus.NOT_MODIFIED}:
        return None

    error_class = _determine_fetch_error_class(
        status=status,
        detail=detail,
        failure=failure,
        status_code=metadata.status_code,
        policy_events=policy_events,
    )

    provider = _provider_from_request(request)
    attributes: Dict[str, object] = {}
    if policy_events:
        attributes["policy_events"] = policy_events
    if failure is not None:
        attributes["failure_reason"] = failure.reason
        attributes["failure_temporary"] = failure.temporary
    if detail and not detail.startswith("status_"):
        attributes.setdefault("detail", detail)

    status_code = _status_code_from_detail(metadata.status_code, detail)
    reason = detail or (failure.reason if failure else status.value)

    return CrawlerError(
        error_class=error_class,
        reason=reason,
        source=request.canonical_source,
        provider=provider,
        status_code=status_code,
        attributes=attributes,
    )


def _determine_fetch_error_class(
    *,
    status: FetchStatus,
    detail: Optional[str],
    failure: Optional[FetchFailure],
    status_code: Optional[int],
    policy_events: Tuple[str, ...],
) -> ErrorClass:
    if status is FetchStatus.POLICY_DENIED:
        if policy_events and "rate_limited" in policy_events:
            return ErrorClass.RATE_LIMIT
        return ErrorClass.POLICY_DENY

    if status is FetchStatus.GONE:
        if status_code == 404:
            return ErrorClass.NOT_FOUND
        return ErrorClass.GONE

    if status is FetchStatus.TEMPORARY_ERROR:
        reason = detail or (failure.reason if failure else "")
        if status_code == 429 or reason == "status_429":
            return ErrorClass.UPSTREAM_429
        if _is_timeout_reason(reason):
            return ErrorClass.TIMEOUT
        return ErrorClass.TRANSIENT_NETWORK

    return ErrorClass.TRANSIENT_NETWORK


def _is_timeout_reason(reason: Optional[str]) -> bool:
    if not reason:
        return False
    return "timeout" in reason.lower()


def _status_code_from_detail(
    status_code: Optional[int], detail: Optional[str]
) -> Optional[int]:
    if status_code is not None:
        return status_code
    if detail and detail.startswith("status_"):
        try:
            return int(detail.split("_", 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def _provider_from_request(request: FetchRequest) -> Optional[str]:
    provider = request.metadata.get("provider")
    if isinstance(provider, str):
        cleaned = provider.strip()
        return cleaned or None
    return None
