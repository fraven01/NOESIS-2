"""Structlog-based logging helpers with contextual enrichment."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from typing import Dict, Iterator, MutableMapping

import structlog
from opentelemetry import trace

from common.redaction import Redactor, hash_email, hash_str, hash_user_id  # noqa: F401

try:  # pragma: no cover - optional instrumentation
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
except Exception:  # pragma: no cover - fallback when OTEL extras missing
    LoggingInstrumentor = None  # type: ignore[assignment]

__all__ = [
    "configure_logging",
    "get_logger",
    "bind_log_context",
    "clear_log_context",
    "get_log_context",
    "log_context",
    "mask_value",
    "hash_str",
    "hash_email",
    "hash_user_id",
    "RequestTaskContextFilter",
]


_CONTEXT_FIELDS: tuple[str, ...] = ("trace_id", "case_id", "tenant", "key_alias")
_LOG_CONTEXT: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "log_context", default=None
)

def _deployment_environment() -> str:
    """Return the deployment environment from supported environment variables."""

    return (
        os.getenv("DEPLOY_ENV")
        or os.getenv("DEPLOYMENT_ENVIRONMENT")
        or "unknown"
    )


_SERVICE_CONTEXT: dict[str, str] = {
    "service.name": os.getenv("SERVICE_NAME", "noesis2"),
    "service.version": os.getenv("SERVICE_VERSION", "unknown"),
    "deployment.environment": _deployment_environment(),
}

_TIME_STAMPER = structlog.processors.TimeStamper(fmt="iso", key="timestamp")
_JSON_RENDERER = structlog.processors.JSONRenderer()
_CONFIGURED = False
_REDACTOR: Redactor | None = None


def get_log_context() -> dict[str, str]:
    """Return a copy of the active logging context."""

    current = _LOG_CONTEXT.get()
    return dict(current) if current else {}


def clear_log_context() -> None:
    """Clear all contextual values from the logging context."""

    _LOG_CONTEXT.set({})


def _filter_allowed(data: Dict[str, object]) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in data.items():
        if key not in _CONTEXT_FIELDS or value is None:
            continue
        filtered[key] = str(value)
    return filtered


def bind_log_context(**kwargs: object) -> contextvars.Token[dict[str, str] | None]:
    """Bind values to the logging context and return the reset token."""

    current = get_log_context()
    filtered = _filter_allowed(kwargs)
    merged = {**current, **filtered}
    return _LOG_CONTEXT.set(merged)


def _reset_log_context(token: contextvars.Token[dict[str, str] | None]) -> None:
    try:
        _LOG_CONTEXT.reset(token)
    except ValueError:
        clear_log_context()


@contextlib.contextmanager
def log_context(**kwargs: object) -> Iterator[None]:
    """Context manager for temporarily binding logging metadata."""

    token = bind_log_context(**kwargs)
    try:
        yield
    finally:
        _reset_log_context(token)


def mask_value(value: str | None) -> str:
    """Mask sensitive values unless staging has opted-in for full fidelity."""

    if not value:
        return "-"
    value = str(value)
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def _masking_enabled() -> bool:
    from django.conf import settings  # imported lazily to avoid circular import

    return not getattr(settings, "LOGGING_ALLOW_UNMASKED_CONTEXT", False)


def _context_processor(
    _: structlog.typing.WrappedLogger,
    __: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    context = get_log_context()
    if not context:
        return event_dict

    mask = _masking_enabled()
    for field in _CONTEXT_FIELDS:
        raw_value = context.get(field)
        if raw_value is None:
            continue
        event_dict[field] = mask_value(raw_value) if mask else str(raw_value)
    return event_dict


def _service_processor(
    _: structlog.typing.WrappedLogger,
    __: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    for key, value in _SERVICE_CONTEXT.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict


def _otel_trace_processor(
    _: structlog.typing.WrappedLogger,
    __: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    span = trace.get_current_span()
    span_context = span.get_span_context() if span else None

    trace_id: str | None = None
    span_id: str | None = None

    if span_context and span_context.is_valid:
        trace_id = f"{span_context.trace_id:032x}"
        span_id = f"{span_context.span_id:016x}"

    event_dict.setdefault("trace_id", trace_id)
    event_dict.setdefault("span_id", span_id)

    if trace_id:
        gcp_project = os.getenv("GCP_PROJECT")
        if gcp_project:
            event_dict.setdefault(
                "logging.googleapis.com/trace",
                f"projects/{gcp_project}/traces/{trace_id}",
            )
    return event_dict


def _structlog_processors(redactor: Redactor) -> list[structlog.types.Processor]:
    def _render_to_json(
        logger: structlog.typing.WrappedLogger,
        name: str,
        event_dict: MutableMapping[str, object],
    ) -> str:
        return _JSON_RENDERER(logger, name, event_dict)

    return [
        structlog.stdlib.filter_by_level,
        _service_processor,
        _context_processor,
        structlog.stdlib.add_log_level,
        _TIME_STAMPER,
        _otel_trace_processor,
        redactor,
        _render_to_json,
    ]


def _configure_stdlib_logging(level: int, redactor: Redactor) -> None:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=_JSON_RENDERER,
            foreign_pre_chain=[
                _service_processor,
                _context_processor,
                structlog.stdlib.add_log_level,
                _TIME_STAMPER,
                _otel_trace_processor,
                redactor,
            ],
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)


def _instrument_logging() -> None:
    if LoggingInstrumentor is None:
        return
    try:  # pragma: no cover - depends on optional instrumentation
        LoggingInstrumentor().instrument(set_logging_format=False)
    except Exception:
        pass


def _log_level_from_env() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logging() -> None:
    """Configure structlog and stdlib logging once."""

    global _CONFIGURED, _REDACTOR
    if _CONFIGURED:
        return

    level = _log_level_from_env()
    redactor = Redactor()
    _configure_stdlib_logging(level, redactor)

    structlog.configure(
        processors=_structlog_processors(redactor),
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _REDACTOR = redactor
    _instrument_logging()
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to service context."""

    logger = structlog.get_logger(name) if name else structlog.get_logger()
    return logger.bind(**_SERVICE_CONTEXT)


class RequestTaskContextFilter(logging.Filter):
    """Inject request/task context metadata into stdlib log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        context = get_log_context()
        mask = _masking_enabled()

        for field in _CONTEXT_FIELDS:
            raw_value = context.get(field)
            if mask:
                value = mask_value(raw_value)
            else:
                value = str(raw_value) if raw_value else "-"
            setattr(record, field, value)

        return True
