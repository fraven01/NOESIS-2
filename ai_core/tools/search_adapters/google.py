"""Google Custom Search adapter implementation."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Literal, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from pydantic import SecretStr
from requests import Session
from requests.exceptions import RequestException, Timeout

from ai_core.infra.observability import record_span
from ai_core.tools.web_search import (
    BaseSearchAdapter,
    ProviderSearchResult,
    RawSearchResult,
    SearchAdapterResponse,
    SearchProviderBadResponse,
    SearchProviderQuotaExceeded,
    SearchProviderTimeout,
)
from common.logging import get_logger

_LOGGER = get_logger(__name__)


class GoogleSearchAdapter(BaseSearchAdapter):
    """Adapter calling the Google Custom Search JSON API."""

    provider_name: Literal["google"] = "google"
    endpoint: str = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        *,
        api_key: SecretStr,
        search_engine_id: str,
        endpoint: str | None = None,
        session: Session | None = None,
        timeout: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.endpoint = (endpoint or self.endpoint).rstrip("/")
        self._session = session or requests.Session()
        self._timeout = float(timeout)
        self._logger = logger or _LOGGER

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        limit = self._coerce_limit(max_results)
        params = {
            "q": query,
            "key": self.api_key.get_secret_value(),
            "cx": self.search_engine_id,
            "num": str(limit),
            "safe": "active",
        }
        start_time = time.perf_counter()
        self._logger.debug("google_search.request", params=params)
        try:
            response = self._session.get(
                self.endpoint,
                params=params,
                timeout=self._timeout,
            )
        except Timeout as exc:  # pragma: no cover - handled via retry logic
            latency_ms = self._latency_ms(start_time)
            self._record_span(latency_ms, None, 0, None, error_kind="Timeout")
            raise SearchProviderTimeout(
                "google_search_timeout",
                http_status=408,
            ) from exc
        except RequestException as exc:  # pragma: no cover - network failure
            latency_ms = self._latency_ms(start_time)
            self._record_span(latency_ms, None, 0, None, error_kind="RequestException")
            raise SearchProviderBadResponse("google_search_request_failed") from exc

        latency_ms = self._latency_ms(start_time)
        status = int(getattr(response, "status_code", 500))
        quota_remaining = self._quota_remaining(response.headers)

        if status in {403, 429}:
            retry_ms = self._retry_after_ms(response.headers)
            self._record_span(
                latency_ms,
                status,
                0,
                quota_remaining,
                error_kind="SearchProviderQuotaExceeded",
            )
            raise SearchProviderQuotaExceeded(
                "google_search_quota_exceeded",
                retry_in_ms=retry_ms,
                http_status=status,
            )
        if 500 <= status < 600:
            self._record_span(
                latency_ms,
                status,
                0,
                quota_remaining,
                error_kind="SearchProviderBadResponse",
            )
            raise SearchProviderBadResponse(
                "google_search_server_error", http_status=status
            )
        if status >= 400:
            self._record_span(
                latency_ms,
                status,
                0,
                quota_remaining,
                error_kind="SearchProviderBadResponse",
            )
            raise SearchProviderBadResponse(
                "google_search_http_error", http_status=status
            )

        try:
            payload = response.json()
        except ValueError as exc:
            self._record_span(
                latency_ms,
                status,
                0,
                quota_remaining,
                error_kind="SearchProviderBadResponse",
                error_message="invalid_json",
            )
            raise SearchProviderBadResponse(
                "google_search_invalid_json", http_status=status
            ) from exc

        items = self._extract_items(payload)
        raw_results: list[RawSearchResult] = []
        provider_results: list[ProviderSearchResult] = []

        for item in items:
            raw = self._to_raw_result(item)
            if raw is None:
                continue
            source = (raw.display_link or self.provider_name).lower()
            raw_results.append(raw)
            provider_results.append(
                ProviderSearchResult(
                    url=raw.link,
                    title=raw.title,
                    snippet=raw.snippet,
                    source=source,
                    score=None,
                    content_type=raw.mime,
                )
            )
            if len(provider_results) >= limit:
                break

        self._logger.debug(
            "google search completed",
            extra={
                "status": status,
                "result_count": len(provider_results),
                "raw_result_count": len(raw_results),
                "quota_remaining": quota_remaining,
            },
        )

        self._record_span(
            latency_ms,
            status,
            len(provider_results),
            quota_remaining,
        )

        return SearchAdapterResponse(
            results=tuple(provider_results),
            status_code=status,
            raw_results=tuple(raw_results),
            quota_remaining=quota_remaining,
        )

    def _coerce_limit(self, requested: int) -> int:
        try:
            limit = int(requested)
        except (TypeError, ValueError):
            limit = 10
        return max(1, min(limit, 10))

    def _latency_ms(self, start_time: float) -> int:
        return max(0, int((time.perf_counter() - start_time) * 1000))

    def _retry_after_ms(self, headers: Mapping[str, str]) -> int | None:
        for key in ("Retry-After", "retry-after"):
            if key in headers:
                value = headers[key]
                break
        else:
            return None
        value = (value or "").strip()
        if not value:
            return None
        try:
            seconds = float(value)
        except ValueError:
            try:
                dt = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (dt - now).total_seconds()
            if delta <= 0:
                return None
            seconds = delta
        if seconds <= 0:
            return None
        return int(seconds * 1000)

    def _quota_remaining(self, headers: Mapping[str, str]) -> int | None:
        for key in (
            "x-ratelimit-remaining",
            "X-RateLimit-Remaining",
            "X-RateLimit-Remaining-Requests",
        ):
            if key in headers:
                value = headers.get(key)
                break
        else:
            value = None
        if not value:
            return None
        try:
            remaining = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return max(remaining, 0)

    def _extract_items(self, payload: object) -> Sequence[Mapping[str, object]]:
        if not isinstance(payload, Mapping):
            return ()
        items = payload.get("items")
        if not isinstance(items, Sequence):
            return ()
        filtered: list[Mapping[str, object]] = []
        for item in items:
            if isinstance(item, Mapping):
                filtered.append(item)
        return filtered

    def _to_raw_result(self, item: Mapping[str, object]) -> RawSearchResult | None:
        title = self._clean_text(item.get("title"))
        snippet = self._clean_text(item.get("snippet") or item.get("htmlSnippet"))
        link = self._clean_url(item.get("link"))
        if not title or not snippet or not link:
            return None
        mime = self._clean_text(item.get("mime") or item.get("fileFormat"))
        display_link = self._clean_text(
            item.get("displayLink") or item.get("formattedUrl")
        )
        return RawSearchResult(
            title=title,
            snippet=snippet,
            link=link,
            mime=mime,
            display_link=display_link,
        )

    def _clean_text(self, value: object) -> str | None:
        if isinstance(value, str):
            text = unescape(value).strip()
            if text:
                return " ".join(text.split())
        return None

    def _clean_url(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        url = value.strip()
        if not url:
            return None
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.netloc:
            return None
        query_items: list[tuple[str, str]] = []
        for key, val in parse_qsl(parsed.query, keep_blank_values=False):
            query_items.append((key, val))
        normalized_query = urlencode(query_items, doseq=True)
        path = parsed.path or "/"
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
                normalized_query,
                "",
            )
        )

    def _record_span(
        self,
        latency_ms: int,
        http_status: int | None,
        result_count: int,
        quota_remaining: int | None,
        *,
        error_kind: str | None = None,
        error_message: str | None = None,
    ) -> None:
        attributes = {
            "provider": self.provider_name,
            "latency_ms": latency_ms,
            "result.count": result_count,
        }
        if http_status is not None:
            attributes["http.status"] = http_status
        if quota_remaining is not None:
            attributes["quota.remaining"] = quota_remaining
        if error_kind:
            attributes["error.kind"] = error_kind
        if error_message:
            attributes["error.message"] = error_message
        record_span("tool.web_search.provider", attributes=attributes)
