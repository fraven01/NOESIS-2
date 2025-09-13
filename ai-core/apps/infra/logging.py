"""Structured JSON logging with trace correlation."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any, Dict

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


class TraceIdFilter(logging.Filter):
    """Inject the current trace id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        record.trace_id = _trace_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        msg = (
            record.msg
            if isinstance(record.msg, dict)
            else {"message": record.getMessage()}
        )
        trace = getattr(record, "trace_id", None)
        if trace:
            msg.setdefault("trace", trace)
        return json.dumps(msg)


def get_logger(name: str = "ai-core") -> logging.Logger:
    """Return a logger configured for JSON output and trace correlation."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler.addFilter(TraceIdFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_trace_id(trace_id: str | None) -> None:
    """Store ``trace_id`` in a context variable for later correlation."""

    _trace_id.set(trace_id)


def event(event_type: str, payload: Dict[str, Any]) -> None:
    """Emit a structured log event with ``event_type`` and ``payload``."""

    data = dict(payload)
    data["event"] = event_type
    logger = get_logger("events")
    logger.info(data)
