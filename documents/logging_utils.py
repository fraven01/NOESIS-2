"""Structured logging helpers for document and asset operations."""

from __future__ import annotations

import contextlib
import contextvars
import io
import inspect
import math
import time
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional
import logging
import sys
from uuid import UUID

from common.logging import get_logger
from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from . import metrics

_ENTRY_EXTRA: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "documents_log_entry_extra", default=None
)
_EXIT_EXTRA: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "documents_log_exit_extra", default=None
)


def log_extra_entry(**fields: Any) -> None:
    """Attach additional metadata to the next entry log."""

    _merge_context(_ENTRY_EXTRA, fields)


def log_extra_exit(**fields: Any) -> None:
    """Attach additional metadata to the next exit log."""

    _merge_context(_EXIT_EXTRA, fields)


def log_extra(
    *,
    entry: Optional[Mapping[str, Any]] = None,
    exit: Optional[Mapping[str, Any]] = None,
) -> None:
    """Attach metadata to both entry and exit logs in a single call."""

    if entry:
        log_extra_entry(**dict(entry))
    if exit:
        log_extra_exit(**dict(exit))


@contextlib.contextmanager
def suppress_logging() -> Iterable[None]:
    root_logger = logging.getLogger()
    swapped: list[tuple[logging.StreamHandler, object]] = []
    previous_raise: bool = logging.raiseExceptions
    logging.raiseExceptions = False
    previous_stderr = sys.stderr
    stderr_buffer = io.StringIO()
    sys.stderr = stderr_buffer
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.StreamHandler):
            continue
        if getattr(handler, "_noesis_file_handler", False):
            continue
        stream = getattr(handler, "stream", None)
        if not hasattr(handler, "setStream"):
            continue
        if stream is None:
            continue
        swapped.append((handler, stream))
        try:
            handler.setStream(io.StringIO())
        except ValueError:
            handler.stream = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = previous_stderr
        logging.raiseExceptions = previous_raise
        for handler, stream in swapped:
            try:
                if stream is None or getattr(stream, "closed", False):
                    target = sys.stderr
                else:
                    target = stream
                try:
                    handler.setStream(target)
                except ValueError:
                    handler.stream = target
            except Exception:
                pass


def log_call(event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator emitting structured start/stop logs for the wrapped callable."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            entry_token = _ENTRY_EXTRA.set({})
            exit_token = _EXIT_EXTRA.set({})
            try:
                bound = signature.bind_partial(*args, **kwargs)
                entry_fields = _coerce_fields(_extract_entry_fields(bound.arguments))
                extra_entry = _ENTRY_EXTRA.get() or {}
                if extra_entry:
                    entry_fields.update(_coerce_fields(extra_entry))
                workflow_label = entry_fields.get("workflow_id")
                tracer = trace.get_tracer(func.__module__ or __name__)
                with tracer.start_as_current_span(
                    event, kind=SpanKind.INTERNAL
                ) as span:
                    _set_span_attributes(span, entry_fields)
                    logger.info(event, phase="start", **entry_fields)
                    try:
                        result = func(*args, **kwargs)
                    except Exception as exc:  # pragma: no cover - exercised via tests
                        exit_fields = dict(entry_fields)
                        extra_exit = _EXIT_EXTRA.get() or {}
                        exit_fields.update(_coerce_fields(extra_exit))
                        workflow = exit_fields.get("workflow_id") or workflow_label
                        duration_ms = _duration_ms(start)
                        error_message = _safe_error_message(exc)
                        _set_span_attributes(span, exit_fields)
                        span.set_attribute("error.type", type(exc).__name__)
                        span.set_attribute("error.message", error_message)
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, error_message))
                        metrics.observe_event(
                            event, "error", duration_ms, workflow_id=workflow
                        )
                        logger.error(
                            event,
                            status="error",
                            duration_ms=duration_ms,
                            error_kind=type(exc).__name__,
                            error_msg=error_message,
                            **exit_fields,
                        )
                        raise
                    else:
                        exit_fields = dict(entry_fields)
                        extra_exit = _EXIT_EXTRA.get() or {}
                        exit_fields.update(_coerce_fields(extra_exit))
                        workflow = exit_fields.get("workflow_id") or workflow_label
                        status = str(exit_fields.pop("status", "ok"))
                        duration_ms = _duration_ms(start)
                        logger.info(
                            event,
                            status=status,
                            duration_ms=duration_ms,
                            **exit_fields,
                        )
                        _set_span_attributes(span, exit_fields)
                        span.set_status(Status(StatusCode.OK))
                        metrics.observe_event(
                            event, status, duration_ms, workflow_id=workflow
                        )
                        return result
            finally:
                _ENTRY_EXTRA.reset(entry_token)
                _EXIT_EXTRA.reset(exit_token)

        return wrapper

    return decorator


def document_log_fields(document: Any) -> dict[str, Any]:
    """Return standard logging fields for a normalized document."""

    fields: dict[str, Any] = {}
    ref = getattr(document, "ref", None)
    if ref is not None:
        _add_field(fields, "tenant_id", getattr(ref, "tenant_id", None))
        _add_field(fields, "document_id", getattr(ref, "document_id", None))
        _add_field(fields, "collection_id", getattr(ref, "collection_id", None))
        _add_field(fields, "version", getattr(ref, "version", None))
        _add_field(fields, "workflow_id", getattr(ref, "workflow_id", None))
    meta = getattr(document, "meta", None)
    if meta is not None and "workflow_id" not in fields:
        _add_field(fields, "workflow_id", getattr(meta, "workflow_id", None))
    _add_field(fields, "source", getattr(document, "source", None))
    fields.update(blob_log_fields(getattr(document, "blob", None)))
    checksum = getattr(document, "checksum", None)
    if isinstance(checksum, str):
        fields["checksum_prefix"] = checksum[:8]
    assets = getattr(document, "assets", None)
    if assets is not None:
        try:
            fields["asset_count"] = len(assets)  # type: ignore[arg-type]
        except TypeError:
            fields["asset_count"] = len(
                list(assets if isinstance(assets, Iterable) else [])
            )
    return fields


def asset_log_fields(asset: Any) -> dict[str, Any]:
    """Return standard logging fields for an asset payload."""

    fields: dict[str, Any] = {}
    ref = getattr(asset, "ref", None)
    if ref is not None:
        _add_field(fields, "tenant_id", getattr(ref, "tenant_id", None))
        _add_field(fields, "asset_id", getattr(ref, "asset_id", None))
        _add_field(fields, "document_id", getattr(ref, "document_id", None))
        _add_field(fields, "collection_id", getattr(ref, "collection_id", None))
        _add_field(fields, "workflow_id", getattr(ref, "workflow_id", None))
    if "workflow_id" not in fields:
        _add_field(fields, "workflow_id", getattr(asset, "workflow_id", None))
    _add_field(fields, "media_type", getattr(asset, "media_type", None))
    fields.update(blob_log_fields(getattr(asset, "blob", None)))
    checksum = getattr(asset, "checksum", None)
    if isinstance(checksum, str):
        fields["checksum_prefix"] = checksum[:8]
    caption_method = getattr(asset, "caption_method", None)
    if caption_method:
        fields["caption_method"] = caption_method
    caption_confidence = getattr(asset, "caption_confidence", None)
    if caption_confidence is not None:
        try:
            fields["caption_confidence"] = float(caption_confidence)
        except (TypeError, ValueError):
            pass
    caption_model = getattr(asset, "caption_model", None)
    if caption_model:
        fields["model"] = caption_model
    return fields


def blob_log_fields(blob: Any) -> dict[str, Any]:
    """Return logging metadata for a blob locator without sensitive payloads."""

    if blob is None:
        return {}
    fields: dict[str, Any] = {}
    size = getattr(blob, "size", None)
    if isinstance(size, (int, float)):
        fields["size_bytes"] = int(size)
    sha256 = getattr(blob, "sha256", None)
    if isinstance(sha256, str):
        fields["sha256_prefix"] = sha256[:8]
    if hasattr(blob, "uri"):
        uri = getattr(blob, "uri")
        if isinstance(uri, str):
            fields["uri_kind"] = uri_kind_from_uri(uri)
    elif hasattr(blob, "kind"):
        kind = getattr(blob, "kind")
        if isinstance(kind, str):
            fields["uri_kind"] = uri_kind_from_uri(kind + "://")
    return fields


def uri_kind_from_uri(uri: str) -> Optional[str]:
    """Map a URI or scheme identifier to a stable storage kind."""

    if not uri:
        return None
    scheme = uri.split(":", 1)[0]
    if scheme.startswith("http"):
        return "http"
    if scheme in {"memory", "s3", "gcs"}:
        return scheme
    return scheme or None


def _merge_context(
    context: contextvars.ContextVar[dict[str, Any] | None], fields: Mapping[str, Any]
) -> None:
    if not fields:
        return
    current = context.get() or {}
    updated = dict(current)
    for key, value in fields.items():
        if value is None:
            continue
        updated[key] = value
    context.set(updated)


def _extract_entry_fields(arguments: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, value in arguments.items():
        if name in {"self", "cls"}:
            continue
        if name == "tenant_id":
            _add_field(fields, "tenant_id", value)
        elif name == "document_id":
            _add_field(fields, "document_id", value)
        elif name == "collection_id":
            _add_field(fields, "collection_id", value)
        elif name == "asset_id":
            _add_field(fields, "asset_id", value)
        elif name == "version":
            _add_field(fields, "version", value)
        elif name == "workflow_id":
            _add_field(fields, "workflow_id", value)
        elif name in {"doc", "document"}:
            fields.update(document_log_fields(value))
        elif name == "asset":
            fields.update(asset_log_fields(value))
        elif name == "assets" and isinstance(value, Iterable):
            try:
                fields["asset_count"] = len(value)  # type: ignore[arg-type]
            except TypeError:
                fields["asset_count"] = len(list(value))
        elif name == "data" and isinstance(value, (bytes, bytearray)):
            fields["size_bytes"] = len(value)
        elif name == "uri" and isinstance(value, str):
            fields["uri_kind"] = uri_kind_from_uri(value)
        elif name == "limit" and isinstance(value, (int, float)):
            fields["limit"] = int(value)
        elif name == "cursor":
            fields["cursor_present"] = bool(value)
        elif name == "media_type" and isinstance(value, str):
            fields["media_type"] = value
    return fields


def _add_field(fields: MutableMapping[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, UUID):
        fields[key] = str(value)
    else:
        fields[key] = value


def _coerce_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in data.items():
        normalized = _coerce_value(value)
        if normalized is not None:
            coerced[key] = normalized
    return coerced


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    return str(value)


def _duration_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _safe_error_message(exc: Exception) -> str:
    message = str(exc)
    if len(message) > 256:
        return message[:253] + "..."
    return message


_SPAN_FIELD_MAP: Mapping[str, str] = {
    "tenant_id": "noesis.tenant_id",
    "collection_id": "noesis.collection_id",
    "document_id": "noesis.document_id",
    "asset_id": "noesis.asset_id",
    "version": "noesis.version",
    "workflow_id": "noesis.workflow_id",
    "source": "noesis.source",
    "uri_kind": "noesis.uri_kind",
    "size_bytes": "noesis.size_bytes",
    "model": "noesis.caption.model",
    "caption_method": "noesis.caption.method",
    "caption_confidence": "noesis.caption.confidence",
}


def _set_span_attributes(span: Span, fields: Mapping[str, Any]) -> None:
    for key, attribute in _SPAN_FIELD_MAP.items():
        if key not in fields:
            continue
        value = _coerce_value(fields[key])
        if value is None:
            continue
        span.set_attribute(attribute, value)


__all__ = [
    "asset_log_fields",
    "blob_log_fields",
    "document_log_fields",
    "log_call",
    "log_extra",
    "log_extra_entry",
    "log_extra_exit",
    "suppress_logging",
    "uri_kind_from_uri",
]
