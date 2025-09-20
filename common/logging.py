"""Logging helpers for enriching records with request/task context."""

from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Dict, Iterator

from django.conf import settings

__all__ = [
    "RequestTaskContextFilter",
    "bind_log_context",
    "clear_log_context",
    "get_log_context",
    "log_context",
    "mask_value",
]


_CONTEXT_FIELDS: tuple[str, ...] = ("trace_id", "case_id", "tenant", "key_alias")

# Store the current logging context using contextvars so it works across async
# boundaries (Django ASGI, Celery tasks).
_LOG_CONTEXT: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "log_context", default=None
)


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
        # Token no longer applies; fall back to clearing the context.
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


class RequestTaskContextFilter(logging.Filter):
    """Inject request/task context metadata into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        context = get_log_context()
        mask = not getattr(settings, "LOGGING_ALLOW_UNMASKED_CONTEXT", False)

        for field in _CONTEXT_FIELDS:
            raw_value = context.get(field)
            if mask:
                value = mask_value(raw_value)
            else:
                value = str(raw_value) if raw_value else "-"
            setattr(record, field, value)

        return True
