"""Common error types for AI tool contracts.

Alle Fehlerwerte werden als reine Strings serialisiert (StrEnum) und sind für JSON-Schemas
stabil.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python <= 3.10 fallback
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal StrEnum fallback for older Python versions."""


class ToolErrorType(StrEnum):
    """Deterministic tool error type identifiers."""

    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    UPSTREAM = "UPSTREAM"
    VALIDATION = "VALIDATION"
    RETRYABLE = "RETRYABLE"
    FATAL = "FATAL"


_DOC_HINT = "See README.md (Fehlercodes Abschnitt) for remediation guidance."


class ToolError(ValueError):
    """Base class for tool errors with structured metadata."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
        doc_hint: str | None = _DOC_HINT,
        retry_after_ms: int | None = None,
        upstream_status: int | None = None,
        endpoint: str | None = None,
        attempt: int | None = None,
    ) -> None:
        detail = f"{code}: {message}"
        if doc_hint:
            detail = f"{detail}. {doc_hint}"
        super().__init__(detail)
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.doc_hint = doc_hint
        self.retry_after_ms = retry_after_ms
        self.upstream_status = upstream_status
        self.endpoint = endpoint
        self.attempt = attempt


class InputError(ToolError):
    """Raised when tool input validation fails.

    Examples:
        - Required fields missing or invalid types.
        - Business context IDs missing for a graph.
        - Payload schema mismatch at boundary.
    """


class TransientError(ToolError):
    """Retryable error for transient failures (timeouts, temporary upstream issues).

    Examples:
        - Network blip or DNS failure.
        - Temporary database connectivity issues.
        - Short-lived service unavailable errors.
    """


class PermanentError(ToolError):
    """Non-retryable error for permanent failures (invalid input, bad config).

    Examples:
        - Invalid credentials or malformed configuration.
        - Unsupported document format or invalid file reference.
        - Validation that cannot succeed on retry.
    """


class RateLimitedError(ToolError):
    """Retryable error for upstream rate limiting.

    Examples:
        - LiteLLM returns HTTP 429.
        - Embedding provider throttles requests.
        - Custom tenant rate limits exceeded.
    """


class UpstreamError(ToolError):
    """Retryable error for upstream service failures.

    Examples:
        - Upstream dependency returns 5xx.
        - Provider API is unavailable or degraded.
        - Circuit breaker open after repeated failures.
    """


__all__ = [
    "ToolErrorType",
    "ToolError",
    "InputError",
    "TransientError",
    "PermanentError",
    "RateLimitedError",
    "UpstreamError",
]
