from datetime import timedelta

from crawler.errors import ErrorClass
from crawler.fetcher import (
    FetchFailure,
    FetchMetadata,
    FetchRequest,
    FetchStatus,
    FetchTelemetry,
    FetcherLimits,
    PolitenessContext,
    evaluate_fetch_response,
)
from common.guardrails import FetcherLimits as SharedFetcherLimits


def make_request(url: str = "https://example.com/foo") -> FetchRequest:
    return FetchRequest(
        canonical_source=url,
        politeness=PolitenessContext(host="example.com"),
        metadata={"provider": "web"},
    )


def test_successful_fetch_includes_payload_and_metadata():
    body = b"<html>hello</html>"
    result = evaluate_fetch_response(
        make_request(),
        status_code=200,
        body=body,
        headers={"Content-Type": "text/html", "ETag": "abc"},
        elapsed=0.5,
        retries=1,
    )

    assert result.status is FetchStatus.FETCHED
    assert result.payload == body
    assert result.metadata == FetchMetadata(
        status_code=200,
        content_type="text/html",
        etag="abc",
        last_modified=None,
        content_length=len(body),
    )
    assert result.telemetry == FetchTelemetry(
        latency=0.5,
        bytes_downloaded=len(body),
        attempt=2,
        retries=1,
        retry_reason=None,
    )
    assert result.error is None


def test_not_modified_discard_payload_and_sets_status():
    result = evaluate_fetch_response(
        make_request(),
        status_code=304,
        body=b"ignored",
        headers={"Content-Length": "7"},
        elapsed=0.2,
    )

    assert result.status is FetchStatus.NOT_MODIFIED
    assert result.payload is None
    assert result.metadata.content_length is None
    assert result.detail == "not_modified"
    assert result.error is None


def test_gone_status_for_not_found_codes():
    for code in (404, 410):
        result = evaluate_fetch_response(
            make_request(),
            status_code=code,
            body=None,
            headers={},
        )
        assert result.status is FetchStatus.GONE
        assert result.detail == f"status_{code}"
        assert result.error is not None
        if code == 404:
            assert result.error.error_class is ErrorClass.NOT_FOUND
        else:
            assert result.error.error_class is ErrorClass.GONE
        assert result.error.source == "https://example.com/foo"


def test_temporary_error_for_transient_status_codes():
    result = evaluate_fetch_response(
        make_request(),
        status_code=503,
        body=None,
        headers={},
        retries=2,
    )
    assert result.status is FetchStatus.TEMPORARY_ERROR
    assert result.telemetry.retries == 2
    assert result.telemetry.retry_reason == "status_503"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.TRANSIENT_NETWORK


def test_policy_denied_for_security_limits():
    limits = FetcherLimits(max_bytes=10, mime_whitelist=("application/pdf",))
    body = b"too big for limit"
    result = evaluate_fetch_response(
        make_request(),
        status_code=200,
        body=body,
        headers={"Content-Type": "text/html"},
        limits=limits,
    )

    assert result.status is FetchStatus.POLICY_DENIED
    assert result.payload is None
    assert set(result.policy_events) == {"max_bytes_exceeded", "mime_not_allowed"}
    assert result.error is not None
    assert result.error.error_class is ErrorClass.POLICY_DENY


def test_policy_denied_when_latency_exceeds_timeout():
    limits = FetcherLimits(timeout=timedelta(seconds=0.1))
    result = evaluate_fetch_response(
        make_request(),
        status_code=200,
        body=b"ok",
        elapsed=0.5,
        limits=limits,
    )

    assert result.status is FetchStatus.POLICY_DENIED
    assert result.detail == "timeout_exceeded"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.POLICY_DENY


def test_failure_without_status_defaults_to_temporary_error():
    failure = FetchFailure(reason="dns_failure", temporary=True)
    result = evaluate_fetch_response(
        make_request(),
        status_code=None,
        body=None,
        failure=failure,
    )

    assert result.status is FetchStatus.TEMPORARY_ERROR
    assert result.detail == "dns_failure"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.TRANSIENT_NETWORK


def test_permanent_failure_converts_to_gone():
    failure = FetchFailure(reason="cert_revoked", temporary=False)
    result = evaluate_fetch_response(
        make_request(),
        status_code=None,
        body=None,
        failure=failure,
    )

    assert result.status is FetchStatus.GONE
    assert result.detail == "cert_revoked"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.GONE


def test_mime_whitelist_with_wildcard():
    limits = FetcherLimits(mime_whitelist=("text/*",))
    result = evaluate_fetch_response(
        make_request(),
        status_code=200,
        body=b"ok",
        headers={"Content-Type": "text/plain"},
        limits=limits,
    )

    assert result.status is FetchStatus.FETCHED
    assert result.error is None


def test_policy_denied_for_http_policy_codes():
    for code in (401, 403, 451):
        result = evaluate_fetch_response(
            make_request(),
            status_code=code,
            body=None,
            headers={},
        )
        assert result.status is FetchStatus.POLICY_DENIED
        assert result.detail == f"status_{code}"
        assert result.policy_events == ("http_policy",)
        assert result.error is not None
        assert result.error.error_class is ErrorClass.POLICY_DENY


def test_fetcher_limits_are_shared() -> None:
    assert FetcherLimits is SharedFetcherLimits
