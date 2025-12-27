"""Web search worker implementation with provider abstraction and telemetry."""

from __future__ import annotations

import inspect
import logging
import math
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence
from urllib.parse import parse_qsl, urlsplit, urlunsplit, urlencode
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
)

from ai_core.infra.observability import record_span
from ai_core.tool_contracts import ToolContext
from ai_core.tools.errors import InputError

LOGGER = logging.getLogger(__name__)


class SearchProviderError(RuntimeError):
    """Base error class for search provider failures."""

    def __init__(
        self,
        message: str,
        *,
        retry_in_ms: int | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_in_ms = retry_in_ms
        self.http_status = http_status


class SearchProviderTimeout(SearchProviderError):
    """Raised when the provider times out."""


class SearchProviderQuotaExceeded(SearchProviderError):
    """Raised when the provider rate limit is exceeded."""


class SearchProviderBadResponse(SearchProviderError):
    """Raised when the provider response cannot be processed."""


@dataclass(slots=True)
class RawSearchResult:
    """Un-normalised search result returned directly by a provider."""

    title: str
    snippet: str
    link: str
    mime: str | None = None
    display_link: str | None = None


@dataclass(slots=True)
class ProviderSearchResult:
    """Raw search result returned by a provider adapter."""

    url: str
    title: str
    snippet: str
    source: str
    score: float | None = None
    content_type: str | None = None


@dataclass(slots=True)
class SearchAdapterResponse:
    """Structured response from an adapter search call."""

    results: Sequence[ProviderSearchResult]
    status_code: int
    raw_results: Sequence[RawSearchResult] = ()
    quota_remaining: int | None = None


class BaseSearchAdapter(Protocol):
    """Interface for web search provider adapters."""

    provider_name: str  # pragma: no cover - attribute contract

    def search(self, query: str, *, max_results: int) -> SearchAdapterResponse:
        """Execute the search request against the provider."""


# Backwards compatibility for callers importing the legacy protocol name.
SearchAdapter = BaseSearchAdapter


class WebSearchInput(BaseModel):
    """Input payload for the web search worker."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def _normalize_query(cls, value: str) -> str:
        without_zero_width = "".join(
            ch for ch in value if unicodedata.category(ch) != "Cf"
        )
        trimmed = without_zero_width.strip()
        normalized = " ".join(trimmed.split())
        if not normalized:
            raise ValueError("query must not be empty")
        tokens = normalized.split()
        if tokens and all(cls._is_operator_only_token(token) for token in tokens):
            raise ValueError("invalid_query")
        return normalized

    @staticmethod
    def _is_operator_only_token(token: str) -> bool:
        stripped = token.strip()
        if not stripped:
            return False
        if ":" not in stripped:
            return False
        prefix, suffix = stripped.split(":", 1)
        return bool(prefix) and not suffix


class SearchResult(BaseModel):
    """Validated search result exposed to callers."""

    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    title: str
    snippet: str
    source: str
    score: float | None = None
    is_pdf: bool = False

    @field_validator("title", "snippet", "source")
    @classmethod
    def _trim_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text


class ToolOutcome(BaseModel):
    """Decision metadata for tool invocations."""

    model_config = ConfigDict(frozen=True)

    decision: str
    rationale: str
    meta: dict[str, object]

    @field_validator("decision")
    @classmethod
    def _validate_decision(cls, value: str) -> str:
        allowed = {"ok", "error"}
        decision = value.strip().lower()
        if decision not in allowed:
            raise ValueError("decision must be 'ok' or 'error'")
        return decision

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("rationale must not be empty")
        return text


class WebSearchResponse(BaseModel):
    """Envelope returned by the web search worker."""

    model_config = ConfigDict(frozen=True)

    results: list[SearchResult]
    outcome: ToolOutcome


class WebSearchWorker:
    """Execute a web search against an adapter with observability hooks."""

    _TRACKING_PREFIXES = ("utm_",)
    _TRACKING_PARAMS = {
        "gclid",
        "fbclid",
        "igshid",
        "msclkid",
        "mc_eid",
        "vero_conv",
        "vero_id",
        "yclid",
    }

    def __init__(
        self,
        adapter: BaseSearchAdapter,
        *,
        max_results: int = 10,
        max_attempts: int = 3,
        backoff_factor: float = 0.6,
        oversample_factor: int = 2,
        sleep: Callable[[float], None] | None = None,
        timer: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if oversample_factor <= 0:
            raise ValueError("oversample_factor must be greater than zero")
        provider_name = getattr(adapter, "provider_name", None)
        if not provider_name:
            legacy = getattr(adapter, "provider", None)
            provider_name = legacy() if callable(legacy) else legacy
        if not provider_name:
            raise ValueError("adapter must define provider_name")
        self._provider = str(provider_name)
        self._adapter = adapter
        self._max_results = max_results
        self._max_attempts = max_attempts
        self._backoff_factor = backoff_factor
        self._oversample_factor = oversample_factor
        self._sleep = sleep or (
            lambda duration: None if duration <= 0 else time.sleep(duration)
        )
        self._timer = timer or time.perf_counter
        self._logger = logger or LOGGER
        self._adapter_limit_kwarg = self._determine_limit_keyword(adapter)

    def run(
        self, *, query: str, context: ToolContext
    ) -> WebSearchResponse:
        """Run the web search with validation, deduplication, and telemetry.

        BREAKING CHANGE (Option A - Strict Separation):
        WebSearchContext has been removed. Use ToolContext instead.
        worker_call_id is now in context.metadata["worker_call_id"].

        Args:
            query: Search query string
            context: ToolContext with scope, business, and metadata

        Returns:
            WebSearchResponse with results and outcome
        """

        try:
            ctx = self._validate_context(context)
        except InputError as exc:
            return self._context_failure(exc)
        try:
            search_input = WebSearchInput(query=query)
        except ValidationError as exc:
            return self._validation_failure(ctx, exc)

        provider = self._provider
        start_time = self._timer()
        span_attributes: dict[str, object]
        http_status = 500
        normalized_results: list[SearchResult] = []
        error: SearchProviderError | InputError | None = None
        raw_result_count = 0
        normalized_result_count = 0
        quota_remaining: int | None = None

        try:
            response = self._execute_with_retries(search_input.query)
            http_status = response.status_code
            raw_result_count = (
                len(response.raw_results)
                if response.raw_results
                else len(response.results)
            )
            normalized_results = self._normalise_results(response.results, provider)
            normalized_result_count = len(normalized_results)
            quota_remaining = response.quota_remaining
            decision = "ok"
            rationale = "search_completed"
        except SearchProviderError as exc:
            error = exc
            http_status = exc.http_status or self._status_from_error(exc)
            decision = "error"
            rationale = self._rationale_from_error(exc)
            normalized_results = []
            raw_result_count = raw_result_count or 0
            normalized_result_count = 0
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.exception("web search worker failed", exc_info=exc)
            error = SearchProviderBadResponse(str(exc))
            http_status = 500
            decision = "error"
            rationale = "unexpected_error"
            normalized_results = []
            raw_result_count = raw_result_count or 0
            normalized_result_count = 0

        latency_ms = max(0, int(math.floor((self._timer() - start_time) * 1000)))
        span_attributes = self._build_span_attributes(
            ctx,
            provider=provider,
            query=search_input.query,
            http_status=http_status,
            raw_result_count=raw_result_count,
            normalized_result_count=normalized_result_count,
            error=error,
            quota_remaining=quota_remaining,
        )
        record_span(
            "tool.web_search", attributes=span_attributes, trace_id=ctx.scope.trace_id
        )

        outcome_meta = self._build_outcome_meta(
            ctx,
            provider=provider,
            latency_ms=latency_ms,
            http_status=http_status,
            raw_result_count=raw_result_count,
            normalized_result_count=normalized_result_count,
            error=error,
            quota_remaining=quota_remaining,
        )

        if error is not None:
            results: list[SearchResult] = []
        else:
            results = normalized_results

        outcome = ToolOutcome(
            decision=decision,
            rationale=rationale,
            meta=outcome_meta,
        )
        return WebSearchResponse(results=results, outcome=outcome)

    def _validate_context(
        self, context: ToolContext
    ) -> ToolContext:
        """Validate ToolContext and ensure worker_call_id is set.

        Args:
            context: ToolContext instance

        Returns:
            ToolContext with worker_call_id in metadata
        """
        # Ensure worker_call_id is set in metadata (for WebSearch-specific tracking)
        worker_call_id = (context.metadata.get("worker_call_id") or "").strip()
        if not worker_call_id:
            worker_call_id = str(uuid4())
            # ToolContext is frozen, so we need to rebuild with updated metadata
            updated_metadata = {**context.metadata, "worker_call_id": worker_call_id}
            context = context.model_copy(update={"metadata": updated_metadata})
        return context

    def _execute_with_retries(self, query: str) -> SearchAdapterResponse:
        attempts = 0
        last_error: SearchProviderError | None = None
        limit = self._max_results * self._oversample_factor
        while attempts < self._max_attempts:
            try:
                response = self._invoke_adapter(query, limit)
                return response
            except TypeError as exc:
                fallback = self._handle_adapter_type_error(query, limit, exc)
                if fallback is not None:
                    return fallback
                raise
            except SearchProviderQuotaExceeded as exc:
                last_error = exc
                attempts += 1
                if attempts >= self._max_attempts:
                    break
                self._sleep(self._retry_delay(attempts, exc.retry_in_ms))
            except SearchProviderTimeout as exc:
                last_error = exc
                attempts += 1
                if attempts >= self._max_attempts:
                    break
                self._sleep(self._retry_delay(attempts, exc.retry_in_ms))
            except SearchProviderBadResponse as exc:
                last_error = exc
                break
        assert last_error is not None
        raise last_error

    def _invoke_adapter(self, query: str, limit: int) -> SearchAdapterResponse:
        return self._adapter.search(query, **{self._adapter_limit_kwarg: limit})

    def _handle_adapter_type_error(
        self, query: str, limit: int, error: TypeError
    ) -> SearchAdapterResponse | None:
        message = str(error)
        attempted = self._adapter_limit_kwarg
        if (
            "unexpected keyword argument" not in message
            or f"'{attempted}'" not in message
        ):
            return None
        alternate = "limit" if attempted == "max_results" else "max_results"
        try:
            response = self._adapter.search(query, **{alternate: limit})
        except TypeError:
            return None
        self._adapter_limit_kwarg = alternate
        return response

    def _retry_delay(self, attempt: int, retry_in_ms: int | None) -> float:
        if retry_in_ms is not None and retry_in_ms > 0:
            return retry_in_ms / 1000.0
        backoff = self._backoff_factor * (2 ** max(0, attempt - 1))
        return min(backoff, 10.0)

    def _determine_limit_keyword(self, adapter: BaseSearchAdapter) -> str:
        search_callable = getattr(adapter, "search", None)
        if search_callable is None:
            return "max_results"
        try:
            parameters = inspect.signature(search_callable).parameters
        except (TypeError, ValueError):
            return "max_results"
        if "max_results" in parameters:
            parameter = parameters["max_results"]
            if parameter.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return "max_results"
        if "limit" in parameters:
            parameter = parameters["limit"]
            if parameter.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return "limit"
        return "max_results"

    def _normalise_results(
        self,
        results: Sequence[ProviderSearchResult],
        provider: str,
    ) -> list[SearchResult]:
        unique: list[SearchResult] = []
        seen: set[str] = set()
        for raw in results:
            cleaned_url = self._clean_url(raw.url)
            if not cleaned_url:
                continue
            key = cleaned_url
            if key in seen:
                continue
            seen.add(key)
            source = (raw.source or provider).strip().lower()
            payload = {
                "url": cleaned_url,
                "title": raw.title,
                "snippet": raw.snippet,
                "source": source,
                "score": raw.score,
                "is_pdf": self._is_pdf(cleaned_url, raw.content_type),
            }
            try:
                result = SearchResult.model_validate(payload)
            except ValidationError:
                self._logger.debug(
                    "dropping invalid search result", extra={"payload": payload}
                )
                continue
            unique.append(result)
            if len(unique) >= self._max_results:
                break
        return unique

    def _clean_url(self, url: str) -> str | None:
        if not url:
            return None
        parsed = urlsplit(url)
        if not parsed.scheme or not parsed.netloc:
            return None
        query_items = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            key_lower = key.lower()
            if key_lower in self._TRACKING_PARAMS:
                continue
            if any(key_lower.startswith(prefix) for prefix in self._TRACKING_PREFIXES):
                continue
            query_items.append((key, value))
        normalized_query = urlencode(sorted(query_items))
        path = parsed.path or "/"
        cleaned = urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                path,
                normalized_query,
                "",
            )
        )
        return cleaned

    def _is_pdf(self, url: str, content_type: str | None) -> bool:
        path = urlsplit(url).path.lower()
        if path.endswith(".pdf"):
            return True
        if content_type and "pdf" in content_type.lower():
            return True
        return False

    def _build_span_attributes(
        self,
        ctx: ToolContext,
        *,
        provider: str,
        query: str,
        http_status: int,
        raw_result_count: int,
        normalized_result_count: int,
        error: SearchProviderError | InputError | None,
        error_kind_override: str | None = None,
        error_message_override: str | None = None,
        quota_remaining: int | None = None,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {
            "tenant_id": ctx.scope.tenant_id,
            "trace_id": ctx.scope.trace_id,
            "workflow_id": ctx.business.workflow_id,
            "case_id": ctx.business.case_id,
            "run_id": ctx.scope.run_id,
            "worker_call_id": ctx.metadata.get("worker_call_id"),
            "provider": provider,
            "query": query,
            "http.status": http_status,
            "result.count": normalized_result_count,
            "raw_result_count": raw_result_count,
            "normalized_result_count": normalized_result_count,
        }
        if quota_remaining is not None:
            attributes["quota.remaining"] = quota_remaining
        error_kind, error_message, _ = self._error_details(
            error,
            kind_override=error_kind_override,
            message_override=error_message_override,
        )
        if error_kind is not None:
            attributes["error.kind"] = error_kind
        if error_message is not None:
            attributes["error.message"] = error_message
        return attributes

    def _build_outcome_meta(
        self,
        ctx: ToolContext,
        *,
        provider: str,
        latency_ms: int,
        http_status: int,
        raw_result_count: int,
        normalized_result_count: int,
        error: SearchProviderError | InputError | None,
        error_kind_override: str | None = None,
        error_message_override: str | None = None,
        quota_remaining: int | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "tenant_id": ctx.scope.tenant_id,
            "trace_id": ctx.scope.trace_id,
            "workflow_id": ctx.business.workflow_id,
            "case_id": ctx.business.case_id,
            "run_id": ctx.scope.run_id,
            "worker_call_id": ctx.metadata.get("worker_call_id"),
            "provider": provider,
            "latency_ms": latency_ms,
            "http_status": http_status,
            "result_count": normalized_result_count,
            "raw_result_count": raw_result_count,
            "normalized_result_count": normalized_result_count,
        }
        if quota_remaining is not None:
            meta["quota_remaining"] = quota_remaining
        error_kind, error_message, retry_in_ms = self._error_details(
            error,
            kind_override=error_kind_override,
            message_override=error_message_override,
        )
        if error_kind is not None and error_message is not None:
            error_meta: dict[str, object] = {
                "kind": error_kind,
                "message": error_message,
            }
            if retry_in_ms is not None:
                error_meta["retry_in_ms"] = retry_in_ms
            meta["error"] = error_meta
        return meta

    def _validation_failure(
        self,
        ctx: ToolContext,
        exc: ValidationError,
    ) -> WebSearchResponse:
        message = "invalid_query"
        error = InputError(
            message, "Query validation failed", context={"errors": exc.errors()}
        )
        outcome = ToolOutcome(
            decision="error",
            rationale="invalid_query",
            meta=self._build_outcome_meta(
                ctx,
                provider=self._provider,
                latency_ms=0,
                http_status=400,
                raw_result_count=0,
                normalized_result_count=0,
                error=error,
                error_kind_override="ValidationError",
                error_message_override=message,
            ),
        )
        record_span(
            "tool.web_search",
            attributes=self._build_span_attributes(
                ctx,
                provider=self._provider,
                query="",
                http_status=400,
                raw_result_count=0,
                normalized_result_count=0,
                error=error,
                error_kind_override="ValidationError",
                error_message_override=message,
            ),
            trace_id=ctx.trace_id,
        )
        return WebSearchResponse(results=[], outcome=outcome)

    def _context_failure(self, error: InputError) -> WebSearchResponse:
        error_meta = {
            "kind": type(error).__name__,
            "message": self._truncate(str(error)),
        }
        meta = {
            "tenant_id": None,
            "trace_id": None,
            "workflow_id": None,
            "case_id": None,
            "run_id": None,
            "worker_call_id": None,
            "provider": self._provider,
            "latency_ms": 0,
            "http_status": 400,
            "result_count": 0,
            "raw_result_count": 0,
            "normalized_result_count": 0,
            "error": error_meta,
        }
        outcome = ToolOutcome(decision="error", rationale="invalid_context", meta=meta)
        record_span(
            "tool.web_search.validation",
            attributes={
                "provider": meta["provider"],
                "error.kind": error_meta["kind"],
                "error.message": error_meta["message"],
            },
        )
        return WebSearchResponse(results=[], outcome=outcome)

    def _error_details(
        self,
        error: SearchProviderError | InputError | None,
        *,
        kind_override: str | None = None,
        message_override: str | None = None,
    ) -> tuple[str | None, str | None, int | None]:
        if kind_override is not None:
            error_kind: str | None = kind_override
        elif error is not None:
            error_kind = type(error).__name__
        else:
            error_kind = None

        if message_override is not None:
            error_message: str | None = self._truncate(message_override)
        elif error is not None:
            error_message = self._truncate(str(error))
        else:
            error_message = None

        retry_in_ms: int | None = None
        if error is not None:
            retry_in_ms = getattr(error, "retry_in_ms", None)

        return error_kind, error_message, retry_in_ms

    def _status_from_error(self, error: SearchProviderError) -> int:
        if isinstance(error, SearchProviderTimeout):
            return 504
        if isinstance(error, SearchProviderQuotaExceeded):
            return 429
        if isinstance(error, SearchProviderBadResponse):
            return 502
        return 500

    def _rationale_from_error(self, error: SearchProviderError) -> str:
        if isinstance(error, SearchProviderTimeout):
            return "provider_timeout"
        if isinstance(error, SearchProviderQuotaExceeded):
            return "provider_rate_limited"
        if isinstance(error, SearchProviderBadResponse):
            return "provider_bad_response"
        return "provider_error"

    def _truncate(self, value: str, limit: int = 256) -> str:
        text = value.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"


__all__ = [
    "BaseSearchAdapter",
    "SearchAdapter",
    "SearchAdapterResponse",
    "ProviderSearchResult",
    "RawSearchResult",
    "SearchProviderError",
    "SearchProviderTimeout",
    "SearchProviderQuotaExceeded",
    "SearchProviderBadResponse",
    "SearchResult",
    "ToolOutcome",
    "WebSearchInput",
    "WebSearchResponse",
    "WebSearchWorker",
]
