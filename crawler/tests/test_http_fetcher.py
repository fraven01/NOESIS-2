import ssl
from datetime import timedelta

import httpx
import pytest

from crawler.errors import ErrorClass
from crawler.fetcher import FetchStatus, FetcherLimits, FetchRequest, PolitenessContext
from crawler.http_fetcher import FetchRetryPolicy, HttpFetcher, HttpFetcherConfig


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def make_request(**metadata) -> FetchRequest:
    merged = {"provider": "web"}
    merged.update(metadata)
    return FetchRequest(
        canonical_source="https://example.com/resource",
        politeness=PolitenessContext(host="example.com"),
        metadata=merged,
    )


def test_fetch_success_returns_payload_and_metadata():
    body = b"payload"
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        clock.advance(0.2)
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, content=body)

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits(max_bytes=1024)
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport, clock=clock, sleeper=clock.sleep)
    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.FETCHED
    assert result.payload == body
    assert result.metadata.content_type == "text/plain"
    assert pytest.approx(result.telemetry.latency or 0.0, rel=1e-3) == 0.2
    assert result.error is None


def test_conditional_request_uses_etag_and_last_modified_headers():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(304, headers={"ETag": "xyz"})

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport)
    request = make_request(etag="abc", last_modified="Wed, 01 Jan 2020 00:00:00 GMT")

    result = fetcher.fetch(request)

    assert result.status is FetchStatus.NOT_MODIFIED
    assert seen_headers.get("if-none-match") == "abc"
    assert seen_headers.get("if-modified-since") == "Wed, 01 Jan 2020 00:00:00 GMT"
    assert result.error is None


def test_mime_not_in_whitelist_is_policy_denied():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "application/zip"}, content=b"binary")

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits(mime_whitelist=("text/plain",))
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport)

    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.POLICY_DENIED
    assert "mime_not_allowed" in result.policy_events
    assert result.error is not None
    assert result.error.error_class is ErrorClass.POLICY_DENY


def test_stream_exceeding_max_bytes_aborts_and_denies_policy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, content=b"x" * 20)

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits(max_bytes=10)
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport)

    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.POLICY_DENIED
    assert "max_bytes_exceeded" in result.policy_events
    assert result.error is not None
    assert result.error.error_class is ErrorClass.POLICY_DENY


def test_timeout_returns_policy_denied_with_detail():
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        clock.advance(0.2)
        raise httpx.ReadTimeout("timeout", request=request)

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits(timeout=timedelta(seconds=0.1))
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport, clock=clock, sleeper=clock.sleep)

    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.POLICY_DENIED
    assert result.detail == "timeout_exceeded"
    assert "timeout_exceeded" in result.policy_events
    assert result.telemetry.retry_reason == "read_timeout"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.POLICY_DENY


def test_retries_until_successful_response():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 2:
            return httpx.Response(503, content=b"")
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    retry_policy = FetchRetryPolicy(max_tries=3, retry_on=(503,), initial_backoff=0.0, jitter=0.0)
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits, retry_policy=retry_policy), transport=transport)

    result = fetcher.fetch(make_request())

    assert attempts["count"] == 2
    assert result.status is FetchStatus.FETCHED
    assert result.telemetry.retries == 1
    assert result.telemetry.retry_reason is None
    assert result.error is None


def test_retry_limit_returns_temporary_error():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(429, content=b"")

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    retry_policy = FetchRetryPolicy(max_tries=3, retry_on=(429,), initial_backoff=0.0, jitter=0.0)
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits, retry_policy=retry_policy), transport=transport)

    result = fetcher.fetch(make_request())

    assert attempts["count"] == 3
    assert result.status is FetchStatus.TEMPORARY_ERROR
    assert result.telemetry.retries == 2
    assert result.telemetry.retry_reason == "status_429"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.UPSTREAM_429


def test_too_many_redirects_is_temporary_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TooManyRedirects("redirect", request=request)

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport)

    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.TEMPORARY_ERROR
    assert result.detail == "too_many_redirects"
    assert result.telemetry.retry_reason == "too_many_redirects"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.TRANSIENT_NETWORK


def test_tls_error_marks_source_as_gone():
    def handler(request: httpx.Request) -> httpx.Response:
        exc = ssl.SSLCertVerificationError("revoked")
        raise httpx.ConnectError("tls", request=request) from exc

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport)

    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.GONE
    assert result.detail == "tls_error"
    assert result.error is not None
    assert result.error.error_class is ErrorClass.GONE


def test_crawl_delay_is_reflected_in_latency():
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    fetcher = HttpFetcher(HttpFetcherConfig(limits=limits), transport=transport, clock=clock, sleeper=clock.sleep)
    request = FetchRequest(
        canonical_source="https://example.com/resource",
        politeness=PolitenessContext(host="example.com", crawl_delay=0.5),
        metadata={"provider": "web"},
    )

    result = fetcher.fetch(request)

    assert result.status is FetchStatus.FETCHED
    assert result.telemetry.latency is not None
    assert result.telemetry.latency >= 0.5
    assert result.telemetry.retry_reason is None
    assert result.error is None


def test_metadata_headers_override_defaults():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    config = HttpFetcherConfig(limits=limits, default_headers={"X-Test": "foo"}, user_agent="crawler/1.0")
    fetcher = HttpFetcher(config, transport=transport)
    request = make_request(headers={"User-Agent": "tenant-agent/2.0", "X-Test": "override"})

    result = fetcher.fetch(request)

    assert result.status is FetchStatus.FETCHED
    assert seen_headers["user-agent"] == "tenant-agent/2.0"
    assert seen_headers["x-test"] == "override"
    assert seen_headers["accept-encoding"] == "gzip, deflate, br"
    assert result.error is None


def test_default_user_agent_respects_settings(settings):
    settings.CRAWLER_HTTP_USER_AGENT = "crawler-suite/2.1"
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    fetcher = HttpFetcher(transport=transport)

    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.FETCHED
    assert seen_headers["user-agent"] == "crawler-suite/2.1"
    assert result.error is None


def test_unsupported_scheme_is_rejected():
    fetcher = HttpFetcher()
    request = FetchRequest(
        canonical_source="ftp://example.com/resource",
        politeness=PolitenessContext(host="example.com"),
    )

    with pytest.raises(ValueError) as excinfo:
        fetcher.fetch(request)

    assert str(excinfo.value) == "unsupported_scheme"


def test_backoff_totals_are_reported_in_telemetry():
    attempts = {"count": 0}
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, content=b"")
        clock.advance(0.05)
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    limits = FetcherLimits()
    retry_policy = FetchRetryPolicy(
        max_tries=2, retry_on=(503,), initial_backoff=0.1, jitter=0.0
    )
    fetcher = HttpFetcher(
        HttpFetcherConfig(limits=limits, retry_policy=retry_policy),
        transport=transport,
        clock=clock,
        sleeper=clock.sleep,
    )

    result = fetcher.fetch(make_request())

    assert result.status is FetchStatus.FETCHED
    assert attempts["count"] == 2
    assert result.telemetry.retries == 1
    assert result.telemetry.retry_reason is None
    assert pytest.approx(result.telemetry.backoff_total_ms, rel=1e-6) == 100.0
    assert result.error is None

