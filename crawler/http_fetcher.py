"""HTTP adapter that implements the crawler fetcher contract using requests."""

from __future__ import annotations

import os
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Tuple
from urllib.parse import urlparse

import requests
import requests.adapters
from django.conf import settings

from .fetcher import (
    FetchFailure,
    FetchRequest,
    FetchResult,
    FetchStatus,
    FetcherLimits,
    evaluate_fetch_response,
)


DEFAULT_USER_AGENT = "noesis-crawler/1.0"


def _resolve_default_user_agent() -> str:
    """Return the configured crawler user agent with environment fallback."""

    configured = None
    if getattr(settings, "configured", False):
        configured = getattr(settings, "CRAWLER_HTTP_USER_AGENT", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    env_value = os.getenv("CRAWLER_HTTP_USER_AGENT")
    if env_value and env_value.strip():
        return env_value.strip()

    return DEFAULT_USER_AGENT


@dataclass(frozen=True)
class FetchRetryPolicy:
    """Retry configuration used by :class:`HttpFetcher`."""

    max_tries: int = 1
    retry_on: Tuple[int, ...] = (429, 500, 502, 503, 504)
    retry_failures: Tuple[str, ...] = ("network_error", "dns_error")
    initial_backoff: float = 1.0
    jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.max_tries < 1:
            raise ValueError("max_tries_invalid")
        if self.initial_backoff < 0:
            raise ValueError("initial_backoff_invalid")
        if self.jitter < 0:
            raise ValueError("jitter_invalid")
        object.__setattr__(self, "retry_on", tuple(int(code) for code in self.retry_on))
        object.__setattr__(
            self, "retry_failures", tuple(str(reason) for reason in self.retry_failures)
        )

    def backoff_for(self, attempt: int) -> float:
        """Return the backoff (seconds) before the given retry attempt."""

        if attempt <= 0 or self.initial_backoff == 0:
            return 0.0
        base = self.initial_backoff * (2 ** (attempt - 1))
        if self.jitter == 0:
            return base
        jitter_span = self.jitter * base
        return random.uniform(max(0.0, base - jitter_span), base + jitter_span)


@dataclass(frozen=True)
class HttpFetcherConfig:
    """Configuration bundle for :class:`HttpFetcher`."""

    limits: Optional[FetcherLimits] = None
    retry_policy: FetchRetryPolicy = field(default_factory=FetchRetryPolicy)
    user_agent: str = field(default_factory=_resolve_default_user_agent)
    default_headers: Mapping[str, str] = field(default_factory=dict)
    max_redirects: int = 10
    allowed_schemes: Tuple[str, ...] = ("http", "https")
    request_timeout: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_schemes",
            tuple(scheme.lower() for scheme in self.allowed_schemes),
        )
        immutable_headers = tuple(
            (str(key), str(value)) for key, value in self.default_headers.items()
        )
        object.__setattr__(self, "default_headers", dict(immutable_headers))


class HttpFetcher:
    """Streaming HTTP fetcher that honors the crawler fetcher contract."""

    def __init__(
        self,
        config: Optional[HttpFetcherConfig] = None,
        *,
        transport: Optional[requests.adapters.HTTPAdapter] = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config or HttpFetcherConfig()
        self._limits = self._config.limits
        self._retry_policy = self._config.retry_policy
        self._user_agent = self._config.user_agent
        self._default_headers = dict(self._config.default_headers)
        self._max_redirects = self._config.max_redirects
        self._allowed_schemes = self._config.allowed_schemes
        self._request_timeout = self._config.request_timeout
        self._transport = transport
        self._clock = clock
        self._sleep = sleeper

    def fetch(self, request: FetchRequest) -> FetchResult:
        """Fetch ``request.canonical_source`` and return a :class:`FetchResult`."""

        self._ensure_supported_scheme(request)

        last_result: Optional[FetchResult] = None
        backoff_total = 0.0
        for attempt in range(1, self._retry_policy.max_tries + 1):
            if attempt > 1:
                backoff = self._retry_policy.backoff_for(attempt - 1)
                if backoff > 0:
                    self._sleep(backoff)
                backoff_total += max(0.0, backoff)
            result = self._perform_attempt(
                request,
                attempt,
                backoff_total_ms=backoff_total * 1000.0,
            )
            last_result = result
            if not self._should_retry(result, attempt):
                return result
            # Ensure response is closed before retrying to free connection
            if result.body is None and result.status == FetchStatus.OK:
                pass  # Should not happen given logic, but good practice
        assert last_result is not None
        return last_result

    def _perform_attempt(
        self, request: FetchRequest, attempt: int, *, backoff_total_ms: float
    ) -> FetchResult:
        headers = self._build_headers(request)
        url = request.canonical_source

        start = self._clock()
        crawl_delay = request.politeness.crawl_delay
        if crawl_delay and crawl_delay > 0:
            self._sleep(crawl_delay)

        try:
            timeout = self._determine_timeout()
            with requests.Session() as session:
                if self._transport:
                    session.mount("https://", self._transport)
                    session.mount("http://", self._transport)

                session.max_redirects = self._max_redirects

                with session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=timeout,
                    allow_redirects=True,
                ) as response:
                    body, downloaded, status_code, response_headers = self._read_stream(
                        response
                    )
        except requests.exceptions.Timeout:
            elapsed = self._elapsed_since(start)
            elapsed = self._ensure_timeout_violation(elapsed)
            timeout_reason = "timeout"
            return evaluate_fetch_response(
                request,
                status_code=None,
                body=None,
                elapsed=elapsed,
                retries=attempt - 1,
                limits=self._limits,
                failure=FetchFailure(reason=timeout_reason, temporary=True),
                retry_reason=timeout_reason,
                backoff_total_ms=backoff_total_ms,
            )
        except requests.exceptions.TooManyRedirects:
            elapsed = self._elapsed_since(start)
            failure = FetchFailure(reason="too_many_redirects", temporary=True)
            return evaluate_fetch_response(
                request,
                status_code=None,
                body=None,
                elapsed=elapsed,
                retries=attempt - 1,
                limits=self._limits,
                failure=failure,
                retry_reason="too_many_redirects",
                backoff_total_ms=backoff_total_ms,
            )
        except requests.exceptions.InvalidSchema:
            elapsed = self._elapsed_since(start)
            failure = FetchFailure(reason="invalid_url", temporary=False)
            return evaluate_fetch_response(
                request,
                status_code=None,
                body=None,
                elapsed=elapsed,
                retries=attempt - 1,
                limits=self._limits,
                failure=failure,
                backoff_total_ms=backoff_total_ms,
            )
        except requests.exceptions.InvalidURL:
            elapsed = self._elapsed_since(start)
            failure = FetchFailure(reason="invalid_url", temporary=False)
            return evaluate_fetch_response(
                request,
                status_code=None,
                body=None,
                elapsed=elapsed,
                retries=attempt - 1,
                limits=self._limits,
                failure=failure,
                backoff_total_ms=backoff_total_ms,
            )
        except requests.exceptions.RequestException as exc:
            elapsed = self._elapsed_since(start)
            failure = self._classify_request_error(exc)
            return evaluate_fetch_response(
                request,
                status_code=None,
                body=None,
                elapsed=elapsed,
                retries=attempt - 1,
                limits=self._limits,
                failure=failure,
                retry_reason=failure.reason,
                backoff_total_ms=backoff_total_ms,
            )

        elapsed = self._elapsed_since(start)
        body_bytes = bytes(body)
        return evaluate_fetch_response(
            request,
            status_code=status_code,
            body=body_bytes,
            headers=response_headers,
            elapsed=elapsed,
            retries=attempt - 1,
            limits=self._limits,
            retry_reason=self._retry_reason_from_status(status_code),
            downloaded_bytes=downloaded,
            backoff_total_ms=backoff_total_ms,
        )

    def _read_stream(
        self, response: requests.Response
    ) -> Tuple[bytearray, int, Optional[int], Mapping[str, str]]:
        buffer = bytearray()
        status_code = response.status_code
        headers = dict(response.headers)
        max_bytes = self._limits.max_bytes if self._limits else None
        downloaded = 0

        # Use iter_content for streaming
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            downloaded += len(chunk)
            if max_bytes is not None:
                remaining = max_bytes - len(buffer)
                if remaining <= 0:
                    break
                if len(chunk) >= remaining:
                    if remaining > 0:
                        buffer.extend(chunk[:remaining])
                    break
                buffer.extend(chunk)
                continue
            buffer.extend(chunk)
        return buffer, downloaded, status_code, headers

    def _determine_timeout(self) -> float:
        if self._limits and self._limits.timeout is not None:
            return self._limits.timeout.total_seconds()
        if self._request_timeout is not None:
            return self._request_timeout
        return 30.0

    def _ensure_supported_scheme(self, request: FetchRequest) -> None:
        parsed = urlparse(request.canonical_source)
        scheme = (parsed.scheme or "").lower()
        if scheme not in self._allowed_schemes:
            raise ValueError("unsupported_scheme")

    def _build_headers(self, request: FetchRequest) -> MutableMapping[str, str]:
        """Merge headers with caller precedence (metadata overrides defaults, UA last)."""
        headers: Dict[str, str] = dict(self._default_headers)

        for key in ("headers", "extra_headers", "additional_headers"):
            maybe_headers = request.metadata.get(key)
            if isinstance(maybe_headers, Mapping):
                for header_key, header_value in maybe_headers.items():
                    if header_key is None or header_value is None:
                        continue
                    headers[str(header_key)] = str(header_value)

        user_agent = request.politeness.user_agent or self._user_agent
        if user_agent:
            headers.setdefault("User-Agent", user_agent)

        headers.setdefault(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        )
        headers.setdefault("Accept-Language", "en-US,en;q=0.5")
        headers.setdefault("Accept-Encoding", "gzip, deflate, br")
        headers.setdefault("Sec-Fetch-Dest", "document")
        headers.setdefault("Sec-Fetch-Mode", "navigate")
        headers.setdefault("Sec-Fetch-Site", "none")
        headers.setdefault("Sec-Fetch-User", "?1")
        headers.setdefault("Upgrade-Insecure-Requests", "1")
        headers.setdefault("Connection", "keep-alive")

        etag = request.metadata.get("etag")
        if isinstance(etag, str) and etag:
            headers.setdefault("If-None-Match", etag)

        last_modified = request.metadata.get("last_modified")
        if isinstance(last_modified, str) and last_modified:
            headers.setdefault("If-Modified-Since", last_modified)

        return headers

    def _classify_request_error(
        self, exc: requests.exceptions.RequestException
    ) -> FetchFailure:
        if isinstance(exc, requests.exceptions.SSLError):
            return FetchFailure(reason="tls_error", temporary=False)
        if isinstance(exc, (requests.exceptions.ConnectionError, socket.gaierror)):
            # requests wraps gaierror in ConnectionError often
            return FetchFailure(reason="network_error", temporary=True)
        if isinstance(exc, requests.exceptions.ChunkedEncodingError):
            return FetchFailure(reason="network_error", temporary=True)
        return FetchFailure(reason="network_error", temporary=True)

    def _retry_reason_from_status(self, status_code: Optional[int]) -> Optional[str]:
        if status_code is None:
            return None
        if status_code in self._retry_policy.retry_on:
            return f"status_{status_code}"
        return None

    def _timeout_reason(self, exc: Exception) -> str:
        # requests timeouts are usually just Timeout or ConnectTimeout/ReadTimeout
        name = exc.__class__.__name__
        if "connect" in name.lower():
            return "connect_timeout"
        if "read" in name.lower():
            return "read_timeout"
        return "timeout"

    def _should_retry(self, result: FetchResult, attempt: int) -> bool:
        if attempt >= self._retry_policy.max_tries:
            return False
        if result.status is not FetchStatus.TEMPORARY_ERROR:
            return False
        status_code = result.metadata.status_code
        if status_code is not None and status_code in self._retry_policy.retry_on:
            return True
        detail = result.detail or ""
        if detail in self._retry_policy.retry_failures:
            return True
        return False

    def _elapsed_since(self, start: float) -> float:
        elapsed = self._clock() - start
        return elapsed if elapsed >= 0 else 0.0

    def _ensure_timeout_violation(self, elapsed: float) -> float:
        if self._limits and self._limits.timeout is not None:
            min_elapsed = self._limits.timeout.total_seconds()
            if elapsed <= min_elapsed:
                return min_elapsed + 1e-6
        return elapsed
