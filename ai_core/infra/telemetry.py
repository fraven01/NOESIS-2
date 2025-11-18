"""Utilities for structured observability logs emitted by LangGraph nodes."""

from __future__ import annotations

import logging
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)


def _build_extra(namespace: str | None, fields: Mapping[str, Any]) -> dict[str, Any]:
    prefix = f"{namespace}." if namespace else ""
    extra = {
        f"{prefix}{key}": value for key, value in fields.items() if value is not None
    }
    return extra


def _safe_log(
    level: int,
    event: str,
    *,
    namespace: str | None = None,
    exc_info: BaseException | bool | None = None,
    **fields: Any,
) -> None:
    extra = _build_extra(namespace, fields) if fields else None
    try:
        if extra:
            LOGGER.log(level, event, extra=extra, exc_info=exc_info)
        else:
            LOGGER.log(level, event, exc_info=exc_info)
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.warning(
            "telemetry.log_failed",
            extra={"telemetry.event": event},
            exc_info=True,
        )


def log_info(event: str, *, namespace: str | None = None, **fields: Any) -> None:
    _safe_log(logging.INFO, event, namespace=namespace, **fields)


def log_warning(event: str, *, namespace: str | None = None, **fields: Any) -> None:
    _safe_log(logging.WARNING, event, namespace=namespace, **fields)


def log_error(event: str, *, namespace: str | None = None, **fields: Any) -> None:
    _safe_log(logging.ERROR, event, namespace=namespace, **fields)


def log_exception(
    event: str,
    *,
    namespace: str | None = None,
    error: BaseException | None = None,
    **fields: Any,
) -> None:
    _safe_log(logging.ERROR, event, namespace=namespace, exc_info=error, **fields)


def auto_ingest_fallback_threshold(
    *,
    original_threshold: float,
    fallback_threshold: float,
    result_count: int,
) -> None:
    log_info(
        "auto_ingest.fallback_threshold",
        namespace="auto_ingest",
        original_threshold=original_threshold,
        fallback_threshold=fallback_threshold,
        result_count=result_count,
    )


def auto_ingest_insufficient_quality(*, min_score: float, fallback_min: float) -> None:
    log_error(
        "auto_ingest.insufficient_quality",
        namespace="auto_ingest",
        min_score=min_score,
        fallback_min=fallback_min,
    )


def auto_ingest_triggered(
    *,
    url_count: int,
    min_score: float,
    selected_count: int,
    avg_score: float,
) -> None:
    log_info(
        "auto_ingest.triggered",
        namespace="auto_ingest",
        url_count=url_count,
        min_score=min_score,
        selected_count=selected_count,
        avg_score=avg_score,
    )


def auto_ingest_trigger_failed(error: BaseException) -> None:
    log_exception(
        "auto_ingest.trigger_failed",
        namespace="auto_ingest",
        error=error,
    )


__all__ = [
    "auto_ingest_fallback_threshold",
    "auto_ingest_insufficient_quality",
    "auto_ingest_trigger_failed",
    "auto_ingest_triggered",
    "log_error",
    "log_exception",
    "log_info",
    "log_warning",
]
