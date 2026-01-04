"""Structlog-based logging helpers with contextual enrichment."""

from __future__ import annotations

import contextlib
import contextvars
import copy
import logging
import logging.config
import os
import re
import sys
from typing import Any, Dict, Iterator, MutableMapping, Sequence, TextIO
from collections.abc import Mapping

import structlog
from opentelemetry import trace

from common.redaction import Redactor, hash_email, hash_str, hash_user_id  # noqa: F401

# NOTE: Avoid importing `ai_core.*` at module import time.
# `pytest-django` imports Django settings early, and settings may import logging helpers.
# Importing `ai_core.infra.*` here can pull in Django settings and create circular imports.

try:  # pragma: no cover - optional instrumentation
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
except Exception:  # pragma: no cover - fallback when OTEL extras missing
    LoggingInstrumentor = None  # type: ignore[assignment]

__all__ = [
    "configure_logging",
    "configure_django_logging",
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


_CONTEXT_FIELDS: tuple[str, ...] = (
    "trace_id",
    "span_id",
    "case_id",
    "tenant_id",
    "key_alias",
    "collection_id",
    "workflow_id",
    "run_id",
    "ingestion_run_id",
    "document_id",
    "document_version_id",
)
_LOG_CONTEXT: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "log_context", default=None
)


def _deployment_environment() -> str:
    """Return the deployment environment from supported environment variables."""

    return os.getenv("DEPLOY_ENV") or os.getenv("DEPLOYMENT_ENVIRONMENT") or "unknown"


_SERVICE_CONTEXT: dict[str, str] = {
    "service.name": os.getenv("SERVICE_NAME", "noesis2"),
    "service.version": os.getenv("SERVICE_VERSION", "unknown"),
    "deployment.environment": _deployment_environment(),
}

_TIME_STAMPER = structlog.processors.TimeStamper(fmt="iso", key="timestamp")
_JSON_RENDERER = structlog.processors.JSONRenderer()
_CONFIGURED = False
_REDACTOR: Redactor | None = None
_CONFIGURED_STREAM: TextIO | None = None
_FILE_STREAM: TextIO | None = None
_FILE_STREAM_PATH: str | None = None
_FILE_HANDLER: logging.StreamHandler | None = None
_MAX_LOG_STRUCTURED_BYTES = 64 * 1024
_LOG_JSON_DUMP_KWARGS: dict[str, object] = {
    "ensure_ascii": False,
    "separators": None,
}
_PII_LOG_CONFIG_CACHE: contextvars.ContextVar[tuple[int, dict[str, object]] | None] = (
    contextvars.ContextVar("log_pii_config_cache", default=None)
)
_PII_FAST_PATH_MARKERS: tuple[str, ...] = (
    "@",
    "+",
    "Bearer ",
    "eyJ",
    "-----BEGIN",
    "token=",
    "key=",
    "secret=",
    "password=",
    "pass=",
    "session=",
    "auth=",
)

_GENERIC_PLACEHOLDER_PATTERN = re.compile(r"\[REDACTED(?:_[A-Z0-9_]+)?\]")
_DETERMINISTIC_PLACEHOLDER_PATTERN = re.compile(r"<[A-Z0-9_]+_[0-9a-fA-F]{8}>")
_ENCODED_GENERIC_PLACEHOLDER_PATTERN = re.compile(
    r"%5BREDACTED(?:_[A-Z0-9_]+)?%5D", re.IGNORECASE
)
_ENCODED_DETERMINISTIC_PLACEHOLDER_PATTERN = re.compile(
    r"%3C[A-Z0-9_]+_[0-9a-fA-F]{8}%3E", re.IGNORECASE
)


def _collapse_placeholders(original: str, masked: str) -> str:
    """Normalise tagged placeholders to the generic ``[REDACTED]`` token."""

    if masked == original:
        return masked

    collapsed = _GENERIC_PLACEHOLDER_PATTERN.sub("[REDACTED]", masked)
    collapsed = _DETERMINISTIC_PLACEHOLDER_PATTERN.sub("[REDACTED]", collapsed)
    collapsed = _ENCODED_GENERIC_PLACEHOLDER_PATTERN.sub("[REDACTED]", collapsed)
    collapsed = _ENCODED_DETERMINISTIC_PLACEHOLDER_PATTERN.sub("[REDACTED]", collapsed)
    return collapsed


_PII_KEY_PATTERN = re.compile(
    r"(?i)(email|token|secret|password|pass|session|auth|key)\s*[:=]"
)


def _resolve_mask_text() -> Any:
    try:
        from ai_core.infra.pii import mask_text as imported
    except Exception:
        return None
    return imported


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
    from django.core.exceptions import ImproperlyConfigured

    try:
        allow_unmasked = getattr(settings, "LOGGING_ALLOW_UNMASKED_CONTEXT", False)
    except ImproperlyConfigured:
        return True

    normalized: bool
    if isinstance(allow_unmasked, str):
        coerced = allow_unmasked.strip().lower()
        if coerced in {"1", "true", "yes", "y", "on"}:
            normalized = True
        elif coerced in {"0", "false", "no", "n", "off", ""}:
            normalized = False
        else:
            normalized = bool(coerced)
    else:
        normalized = bool(allow_unmasked)

    return not normalized


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
    # Opportunistic JSON-string redaction regardless of logger name
    try:
        event_dict = _jsonish_redaction_processor(_, __, event_dict)
    except Exception:
        pass
    # Targeted safety net for JSON payloads commonly logged as 'payload'
    try:
        raw_payload = event_dict.get("payload")
        if isinstance(raw_payload, str) and "{" in raw_payload and '":' in raw_payload:
            from ai_core.infra.pii_flags import get_pii_config  # lazy

            cfg = get_pii_config() or {}
            deterministic = bool(cfg.get("deterministic", False))
            hmac_secret = cfg.get("hmac_secret") if deterministic else None
            mode = str(cfg.get("mode", "industrial"))
            policy = str(cfg.get("policy", "balanced"))
            try:
                from ai_core.infra.pii import mask_text  # lazy
            except Exception:
                mask_text = None  # type: ignore[assignment]

            masked = (
                mask_text(  # type: ignore[misc]
                    raw_payload,
                    policy,
                    deterministic,
                    hmac_secret,
                    mode=mode,
                    name_detection=False,
                    session_scope=cfg.get("session_scope"),
                    structured_max_length=_MAX_LOG_STRUCTURED_BYTES,
                    json_dump_kwargs=_LOG_JSON_DUMP_KWARGS,
                )
                if mask_text is not None
                else raw_payload
            )
            if masked != raw_payload:
                event_dict["payload"] = _collapse_placeholders(raw_payload, masked)
    except Exception:
        pass
    return event_dict


def _ensure_trace_keys(
    _: structlog.typing.WrappedLogger,
    __: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    context = get_log_context()
    if context:
        mask = _masking_enabled()
        for field in (
            "trace_id",
            "case_id",
            "tenant_id",
            "key_alias",
            "collection_id",
            "workflow_id",
            "run_id",
            "ingestion_run_id",
            "document_id",
            "document_version_id",
        ):
            value = event_dict.get(field)
            if value not in (None, ""):
                continue
            raw_value = context.get(field)
            if raw_value is None:
                continue
            event_dict[field] = mask_value(raw_value) if mask else str(raw_value)

    for key in ("trace_id", "span_id", "case_id", "tenant_id"):
        event_dict.setdefault(key, None)

    return event_dict


def _stringify_ids_for_payload(
    _: structlog.typing.WrappedLogger,
    __: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    is_payload_logger = bool(event_dict.pop("_payload_logger", False))

    if not is_payload_logger:
        return event_dict

    for key in ("trace_id", "span_id"):
        if event_dict.get(key) is None:
            event_dict[key] = ""

    return event_dict


def _pii_redaction_processor_factory() -> structlog.types.Processor | None:
    from django.conf import settings as django_settings

    if not getattr(django_settings, "configured", False):
        return None

    def _resolve_scoped_pii_config() -> dict[str, object]:
        from ai_core.infra.pii_flags import (  # lazy (avoid settings import at module import time)
            get_pii_config,
            get_pii_config_version,
        )

        cached = _PII_LOG_CONFIG_CACHE.get()
        version = get_pii_config_version()
        if cached is not None and cached[0] == version:
            return cached[1]
        config = get_pii_config()
        _PII_LOG_CONFIG_CACHE.set((version, config))
        return config

    def _processor(
        _: structlog.typing.WrappedLogger,
        __: str,
        event_dict: MutableMapping[str, object],
    ) -> MutableMapping[str, object]:
        config = _resolve_scoped_pii_config()
        if not config.get("logging_redaction"):
            return event_dict

        mode = str(config.get("mode", "industrial"))
        policy = str(config.get("policy", "balanced"))
        if mode == "off" or policy == "off":
            return event_dict

        deterministic = bool(config.get("deterministic"))
        hmac_secret = config.get("hmac_secret") if deterministic else None
        if deterministic and not isinstance(hmac_secret, (bytes, bytearray)):
            hmac_secret = None
            deterministic = False
        name_detection = bool(config.get("name_detection", False))
        session_scope = config.get("session_scope")

        def _looks_interesting(value: str) -> bool:
            if any(marker in value for marker in _PII_FAST_PATH_MARKERS):
                return True

            lower_value = value.lower()
            if "{" in value and '":' in value:
                return True
            if "=" in value and any(
                token in lower_value
                for token in (
                    "email",
                    "token",
                    "secret",
                    "password",
                    "pass",
                    "session",
                    "auth",
                    "key",
                )
            ):
                return True

            if _PII_KEY_PATTERN.search(value):
                return True

            consecutive = 0
            for char in value:
                if char.isdigit():
                    consecutive += 1
                    if consecutive >= 7:
                        return True
                else:
                    consecutive = 0

            digit_sequences = [len(chunk) for chunk in re.split(r"\D+", value) if chunk]
            if digit_sequences:
                if any(length >= 7 for length in digit_sequences):
                    return True
                if (
                    len(digit_sequences) >= 2
                    and sum(digit_sequences) >= 7
                    and any(length >= 3 for length in digit_sequences[:-1])
                    and digit_sequences[-1] >= 4
                ):
                    return True

            return False

        for key, raw_value in list(event_dict.items()):
            if not isinstance(raw_value, str):
                continue

            # Exempt known system identifiers from heuristic redaction
            if key in _CONTEXT_FIELDS or key in {"sha256_prefix", "checksum_prefix"}:
                continue

            # Always try structured masking for JSON-like strings to ensure keys such as
            # "access_token" are redacted, independent of the fast-path heuristic.
            is_jsonish = "{" in raw_value and '":' in raw_value

            if not is_jsonish and (
                not name_detection and not _looks_interesting(raw_value)
            ):
                continue

            structured_limit = (
                _MAX_LOG_STRUCTURED_BYTES
                if len(raw_value) <= _MAX_LOG_STRUCTURED_BYTES
                else 0
            )
            json_kwargs = _LOG_JSON_DUMP_KWARGS if structured_limit else None

            masker = _resolve_mask_text()
            if masker is None:
                continue

            masked = masker(
                raw_value,
                policy,
                deterministic,
                hmac_secret,
                mode=mode,
                name_detection=name_detection,
                session_scope=session_scope,
                structured_max_length=structured_limit,
                json_dump_kwargs=json_kwargs,
            )
            if masked != raw_value:
                event_dict[key] = _collapse_placeholders(raw_value, masked)
        return event_dict

    return _processor


def _jsonish_redaction_processor(
    _: structlog.typing.WrappedLogger,
    __: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    """Lightweight fallback to redact JSON-like string fields when enabled.

    This runs regardless of Django settings initialization state and only
    processes values that look like small JSON strings. It preserves spacing
    by passing the same dump kwargs used in the main PII processor.
    """
    try:
        from ai_core.infra.pii_flags import get_pii_config  # lazy import
    except Exception:
        return event_dict

    try:
        cfg = get_pii_config() or {}
    except Exception:
        cfg = {}
    deterministic = bool(cfg.get("deterministic", False))
    hmac_secret = cfg.get("hmac_secret") if deterministic else None
    mode = str(cfg.get("mode", "industrial"))
    policy = str(cfg.get("policy", "balanced"))
    session_scope = cfg.get("session_scope")

    for key, raw_value in list(event_dict.items()):
        if not isinstance(raw_value, str):
            continue
        is_jsonish = "{" in raw_value and '":' in raw_value
        if not is_jsonish:
            continue
        masker = _resolve_mask_text()
        if masker is None:
            return event_dict
        try:
            masked = masker(
                raw_value,
                policy,
                deterministic,
                hmac_secret,
                mode=mode,
                name_detection=False,
                session_scope=session_scope,
                structured_max_length=_MAX_LOG_STRUCTURED_BYTES,
                json_dump_kwargs=_LOG_JSON_DUMP_KWARGS,
            )
        except Exception:
            continue
        if masked != raw_value:
            event_dict[key] = _collapse_placeholders(raw_value, masked)
    return event_dict


def _otel_trace_processor(
    _: structlog.typing.WrappedLogger,
    __: str,
    event_dict: MutableMapping[str, object],
) -> MutableMapping[str, object]:
    span = trace.get_current_span()
    span_context = span.get_span_context() if span else None

    def _format_hex(value: int, width: int) -> str:
        return f"{value:0{width}x}"

    if not span_context or not span_context.is_valid:
        event_dict["trace_id"] = None
        event_dict["span_id"] = None
        return event_dict

    trace_id = _format_hex(span_context.trace_id, 32)
    span_id = _format_hex(span_context.span_id, 16)

    event_dict["trace_id"] = trace_id
    event_dict["span_id"] = span_id

    gcp_project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if gcp_project:
        event_dict.setdefault(
            "logging.googleapis.com/trace",
            f"projects/{gcp_project}/traces/{trace_id}",
        )
        event_dict.setdefault("logging.googleapis.com/spanId", span_id)
    return event_dict


def _structlog_processors(
    redactor: Redactor,
    pii_processor: structlog.types.Processor | None,
) -> list[structlog.types.Processor]:
    processors: list[structlog.types.Processor] = [
        structlog.stdlib.filter_by_level,
        _service_processor,
        _context_processor,
        structlog.stdlib.add_log_level,
        _TIME_STAMPER,
        _otel_trace_processor,
        _ensure_trace_keys,
    ]
    if pii_processor is not None:
        processors.append(pii_processor)
    # Ensure JSON string fields get redacted even if the PII processor is unavailable
    processors.append(_jsonish_redaction_processor)
    processors.extend(
        [
            redactor,
            _stringify_ids_for_payload,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    )
    return processors


class _ContextAwareBoundLogger(structlog.stdlib.BoundLogger):
    """Bound logger that enriches events with the active log context."""

    def _process_event(
        self,
        method_name: str,
        event: str | None,
        event_kw: dict[str, object],
    ) -> tuple[Sequence[Any], MutableMapping[str, object]]:
        event_dict: Any = self._context.copy()
        event_dict.update(**event_kw)

        extra_payload = event_dict.pop("extra", None)
        if isinstance(extra_payload, Mapping):
            for key, value in extra_payload.items():
                event_dict.setdefault(key, value)

        context = get_log_context()
        if context:
            mask = _masking_enabled()
            for field in _CONTEXT_FIELDS:
                current_value = event_dict.get(field)
                if current_value not in (None, ""):
                    continue
                raw_value = context.get(field)
                if raw_value is None:
                    continue
                event_dict[field] = mask_value(raw_value) if mask else str(raw_value)

        if event is not None:
            event_dict["event"] = event

        config = structlog.get_config()
        config_processors = (
            config.get("processors") if isinstance(config, dict) else None
        )
        processors = config_processors or self._processors or ()

        for proc in processors:
            event_dict = proc(self._logger, method_name, event_dict)

        if isinstance(event_dict, (str, bytes, bytearray)):
            return (event_dict,), {}

        if isinstance(event_dict, tuple):
            return event_dict

        if isinstance(event_dict, dict):
            return (), event_dict

        msg = (
            "Last processor didn't return an appropriate value.  "
            "Valid return values are a dict, a tuple of (args, kwargs), bytes, or a str."
        )
        raise ValueError(msg)


def _configure_stdlib_logging(
    level: int,
    redactor: Redactor,
    target_stream: TextIO | None,
    pii_processor: structlog.types.Processor | None,
    file_stream: TextIO | None,
) -> None:
    handler: logging.StreamHandler | None = None
    root_logger = logging.getLogger()

    for existing in root_logger.handlers:
        if getattr(existing, "_noesis_file_handler", False):  # type: ignore[attr-defined]
            continue
        if isinstance(existing, logging.StreamHandler):
            handler = existing
            break

    stream_closed = bool(handler and getattr(handler.stream, "closed", False))

    if handler is None or stream_closed:
        handler = logging.StreamHandler(target_stream)
    else:
        try:
            handler.flush()
        except ValueError:
            handler = logging.StreamHandler(target_stream)
        else:
            handler.setStream(target_stream)

    handler.setLevel(level)
    foreign_pre_chain = [
        _service_processor,
        _context_processor,
        structlog.stdlib.add_log_level,
        _TIME_STAMPER,
        _otel_trace_processor,
        _ensure_trace_keys,
    ]
    if pii_processor is not None:
        foreign_pre_chain.append(pii_processor)
    # Also include JSON string redaction in stdlib pre-chain
    foreign_pre_chain.extend(
        [_jsonish_redaction_processor, redactor, _stringify_ids_for_payload]
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=_JSON_RENDERER,
        foreign_pre_chain=foreign_pre_chain,
    )

    handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [handler]

    global _FILE_HANDLER
    if file_stream is not None:
        file_handler = _FILE_HANDLER
        needs_new_handler = True
        if isinstance(file_handler, logging.StreamHandler):
            file_closed = bool(getattr(file_handler.stream, "closed", False))
            if not file_closed:
                try:
                    file_handler.flush()
                except ValueError:
                    file_handler = None
                else:
                    file_handler.setStream(file_stream)
                    needs_new_handler = False
        if file_handler is None or needs_new_handler:
            file_handler = logging.StreamHandler(file_stream)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=_JSON_RENDERER,
                foreign_pre_chain=foreign_pre_chain.copy(),
            )
        )
        setattr(file_handler, "_noesis_file_handler", True)
        handlers.append(file_handler)
        _FILE_HANDLER = file_handler
    else:
        if _FILE_HANDLER is not None:
            try:
                stream_obj = getattr(_FILE_HANDLER, "stream", None)
                if stream_obj not in (None, target_stream, sys.stderr):
                    stream_obj.close()
            except Exception:
                pass
        _FILE_HANDLER = None

    root_logger.handlers = handlers
    root_logger.setLevel(level)


def _instrument_logging() -> None:
    """Optionally enable OpenTelemetry logging instrumentation.

    Disabled by default to avoid background exporter overhead in dev/WSL.
    Enable by setting LOGGING_OTEL_INSTRUMENT=true.
    """

    if LoggingInstrumentor is None:
        return

    flag = os.getenv("LOGGING_OTEL_INSTRUMENT", "").strip().lower()
    enabled = flag in {"1", "true", "yes", "on"}
    if not enabled:
        return

    try:  # pragma: no cover - depends on optional instrumentation
        # LoggingInstrumentor().instrument(set_logging_format=False)
        pass
    except Exception:
        # Never fail app startup due to optional instrumentation
        pass


def _log_level_from_env() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _close_file_stream() -> None:
    """Close and reset the cached file stream if present."""

    global _FILE_STREAM, _FILE_STREAM_PATH

    stream = _FILE_STREAM
    _FILE_STREAM = None
    _FILE_STREAM_PATH = None

    if stream is None:
        return

    try:
        stream.close()
    except Exception:
        pass


def _stream_from_env() -> TextIO | None:
    """Optionally open or reuse a log file when APP_LOG_DIR or LOG_FILE_PATH is set."""

    global _FILE_STREAM, _FILE_STREAM_PATH

    path = os.getenv("LOG_FILE_PATH")
    if not path:
        app_log_dir = os.getenv("APP_LOG_DIR")
        if app_log_dir:
            path = os.path.join(app_log_dir, "noesis-app.log")
    if not path:
        _close_file_stream()
        return None

    cached = _FILE_STREAM
    if cached is not None:
        closed = bool(getattr(cached, "closed", False))
        if not closed and _FILE_STREAM_PATH == path:
            return cached
        _close_file_stream()

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # line-buffered text stream
        _FILE_STREAM = open(path, mode="a", encoding="utf-8", buffering=1)
        _FILE_STREAM_PATH = path
        return _FILE_STREAM
    except Exception:
        _close_file_stream()
        return None


def configure_logging(stream: TextIO | None = None) -> None:
    """Configure structlog and stdlib logging once."""

    global _CONFIGURED, _REDACTOR, _CONFIGURED_STREAM

    file_stream: TextIO | None = None
    if stream is None:
        file_stream = _stream_from_env()
    else:
        _close_file_stream()

    active_stream = stream or sys.stderr

    level = _log_level_from_env()

    pii_processor = _pii_redaction_processor_factory()

    if _CONFIGURED:
        if _REDACTOR is None:
            _REDACTOR = Redactor()

        file_handler_missing = file_stream is not None and _FILE_HANDLER is None
        file_handler_should_remove = file_stream is None and _FILE_HANDLER is not None

        if (
            (_CONFIGURED_STREAM is not active_stream)
            or file_handler_missing
            or file_handler_should_remove
        ) and _REDACTOR is not None:
            _configure_stdlib_logging(
                level,
                _REDACTOR,
                active_stream,
                pii_processor,
                file_stream,
            )
            _CONFIGURED_STREAM = active_stream
        return

    redactor = Redactor()
    _configure_stdlib_logging(
        level,
        redactor,
        active_stream,
        pii_processor,
        file_stream,
    )

    structlog.configure(
        processors=_structlog_processors(redactor, pii_processor),
        wrapper_class=_ContextAwareBoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _REDACTOR = redactor
    _instrument_logging()
    _CONFIGURED = True
    _CONFIGURED_STREAM = active_stream


def configure_django_logging(logging_settings: dict[str, object] | None) -> None:
    """Django `LOGGING_CONFIG` hook that preserves structlog configuration."""

    configure_logging()

    if not logging_settings:
        return

    config = copy.deepcopy(logging_settings)
    config.pop("root", None)

    extras_present = any(
        bool(config.get(key)) for key in ("handlers", "loggers", "filters")
    )
    if not extras_present:
        return

    config.setdefault("version", 1)
    config.setdefault("disable_existing_loggers", False)

    logging.config.dictConfig(config)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to service context."""

    if _CONFIGURED:
        stream = _CONFIGURED_STREAM
        if stream is not None and getattr(stream, "closed", False):
            configure_logging()
        else:
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if not isinstance(handler, logging.StreamHandler):
                    continue
                handler_stream = getattr(handler, "stream", None)
                if handler_stream is not None and getattr(
                    handler_stream, "closed", False
                ):
                    configure_logging()
                    break

    logger = structlog.get_logger(name) if name else structlog.get_logger()

    if isinstance(logger, structlog._config.BoundLoggerLazyProxy):  # type: ignore[attr-defined]
        logger = logger.bind()

    if not isinstance(logger, _ContextAwareBoundLogger):
        underlying = getattr(logger, "_logger", None)
        if underlying is None or not hasattr(underlying, "disabled"):
            # Use a stdlib logger as the wrapped logger to stay compatible with
            # `structlog.stdlib.*` processors (e.g. `filter_by_level`) even if
            # this module is imported before `configure_logging()` runs.
            underlying = logging.getLogger(name)
        processors = getattr(logger, "_processors", None)
        context = getattr(logger, "_context", {})
        logger = _ContextAwareBoundLogger(underlying, processors, context)

    bound = logger.bind(**_SERVICE_CONTEXT)

    default_fields = {
        "trace_id": None,
        "span_id": None,
        "case_id": None,
        "tenant_id": None,
        "workflow_id": None,
        "run_id": None,
        "ingestion_run_id": None,
        "collection_id": None,
        "document_id": None,
        "document_version_id": None,
    }

    if name and name.startswith("ai_core"):
        default_fields["_payload_logger"] = True

    return bound.bind(**default_fields)


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
