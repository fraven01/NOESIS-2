"""OpenTelemetry-backed observability helpers.

All functions in this module gracefully degrade to no-ops when the
OpenTelemetry SDK is not available or tracing is disabled via
configuration. This keeps local development lightweight while providing a
clear path to production-grade tracing.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import base64
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Optional, TypeVar, cast

LOGGER = logging.getLogger(__name__)

_OTEL_TRACE_SPEC = importlib.util.find_spec("opentelemetry.trace")
_OTEL_TRACE = (
    importlib.import_module("opentelemetry.trace")
    if _OTEL_TRACE_SPEC is not None
    else None
)

_OTEL_ROOT_CM: ContextVar[Any | None] = ContextVar("OTEL_ROOT_CM", default=None)
_OTEL_CONFIGURED: bool = False

F = TypeVar("F", bound=Callable[..., Any])


def _parse_key_value_pairs(raw: str) -> dict[str, str]:
    """Parse comma separated key=value pairs into a dictionary."""

    result: dict[str, str] = {}
    if not raw:
        return result
    for chunk in raw.split(","):
        if not chunk:
            continue
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        if not key:
            continue
        result[key] = value.strip()
    return result


def _ensure_tracer_provider_configured() -> None:
    """Initialise an OTLP trace exporter wired to Langfuse when possible."""

    global _OTEL_CONFIGURED

    if _OTEL_CONFIGURED:
        return
    if _OTEL_TRACE is None:
        return

    try:
        from opentelemetry import trace as otel_trace_api  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import (  # type: ignore
            BatchSpanProcessor,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.debug("OpenTelemetry SDK not available: %s", exc)
        return

    current_provider = otel_trace_api.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        _OTEL_CONFIGURED = True
        return

    endpoint = (os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or "").strip()
    if not endpoint:
        langfuse_host = (os.getenv("LANGFUSE_HOST") or "").strip()
        if langfuse_host:
            endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel/v1/traces"
    if not endpoint:
        LOGGER.debug("OTLP traces endpoint not configured; skipping tracer setup")
        return

    headers = _parse_key_value_pairs(os.getenv("OTEL_EXPORTER_OTLP_HEADERS") or "")
    public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    if public_key and "X-Langfuse-Public-Key" not in headers:
        headers["X-Langfuse-Public-Key"] = public_key
    if secret_key and "X-Langfuse-Secret-Key" not in headers:
        headers["X-Langfuse-Secret-Key"] = secret_key
    if (
        public_key
        and secret_key
        and "authorization" not in {key.lower(): key for key in headers}
    ):
        token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode(
            "ascii"
        )
        headers.setdefault("Authorization", f"Basic {token}")
    if not headers.get("X-Langfuse-Public-Key") or not headers.get(
        "X-Langfuse-Secret-Key"
    ):
        LOGGER.debug("Langfuse OTLP headers missing; skipping tracer setup")
        return

    resource_attrs = _parse_key_value_pairs(os.getenv("OTEL_RESOURCE_ATTRIBUTES") or "")
    resource_attrs.setdefault("service.name", os.getenv("SERVICE_NAME", "noesis2"))
    resource_attrs.setdefault(
        "service.version", os.getenv("SERVICE_VERSION", "unknown")
    )
    deployment_env = (
        os.getenv("DEPLOY_ENV") or os.getenv("DEPLOYMENT_ENVIRONMENT") or ""
    ).strip()
    if deployment_env:
        resource_attrs.setdefault("deployment.environment", deployment_env)

    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None)
        provider = TracerProvider(resource=Resource.create(resource_attrs))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace_api.set_tracer_provider(provider)
        _OTEL_CONFIGURED = True
        LOGGER.info(
            "configured_otlp_exporter",
            extra={
                "endpoint": endpoint,
                "service_name": resource_attrs.get("service.name"),
                "deployment_environment": resource_attrs.get("deployment.environment"),
            },
        )
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("Failed to configure Langfuse OTLP exporter")


def _get_tracer() -> Any | None:
    if _OTEL_TRACE is None:
        return None
    get_tracer = getattr(_OTEL_TRACE, "get_tracer", None)
    if not callable(get_tracer):
        return None
    try:
        return get_tracer("noesis2")
    except Exception:
        return None


def _get_current_span() -> Any | None:
    if _OTEL_TRACE is None:
        return None
    get_current_span = getattr(_OTEL_TRACE, "get_current_span", None)
    if not callable(get_current_span):
        return None
    try:
        span = get_current_span()
    except Exception:
        return None
    if span is None:
        return None
    is_recording = getattr(span, "is_recording", None)
    try:
        if callable(is_recording) and not is_recording():
            return None
    except Exception:
        return None
    return span


def tracing_enabled() -> bool:
    """Return ``True`` when tracing should attempt to emit spans."""

    if _OTEL_TRACE is None:
        return False
    sample = (os.getenv("LANGFUSE_SAMPLE_RATE") or "1").strip().lower()
    if sample in {"0", "0.0"}:
        return False
    exporter = (os.getenv("OTEL_TRACES_EXPORTER") or "").strip().lower()
    if exporter == "none":
        return False
    _ensure_tracer_provider_configured()
    if not _OTEL_CONFIGURED:
        return False
    return _get_tracer() is not None


def observe_span(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator that wraps the target in an OpenTelemetry span when available."""

    def decorator(func: F) -> F:
        span_name = name or getattr(func, "__name__", "observe")

        def wrapped(*args: Any, **kwargs: Any):  # type: ignore[misc]
            if not tracing_enabled():
                return func(*args, **kwargs)
            tracer = _get_tracer()
            if tracer is None:
                return func(*args, **kwargs)
            try:
                cm = tracer.start_as_current_span(span_name)
            except Exception:
                return func(*args, **kwargs)
            with cm:
                return func(*args, **kwargs)

        return cast(F, wrapped)

    return decorator


def _apply_attributes(span: Any, attributes: dict[str, Any]) -> None:
    setter = getattr(span, "set_attribute", None)
    if not callable(setter):
        return
    for key, value in attributes.items():
        try:
            setter(key, value)
        except Exception:
            continue


def _normalise_attribute_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        normalised = [
            item
            for item in (_normalise_attribute_value(v) for v in value)
            if item is not None
        ]
        return normalised
    try:
        return str(value)
    except Exception:
        return None


def _normalise_attributes(values: dict[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for key, value in values.items():
        normalised = _normalise_attribute_value(value)
        if normalised is not None:
            attributes[key] = normalised
    return attributes


def update_observation(**fields: Any) -> None:
    """Attach metadata to the current span when tracing is active."""

    span = _get_current_span()
    if span is None:
        return

    attributes: dict[str, Any] = {}
    user_id = fields.get("user_id")
    session_id = fields.get("session_id")
    tags = fields.get("tags")
    metadata = fields.get("metadata")

    if user_id is not None:
        attributes["user.id"] = str(user_id)
        attributes["user_id"] = str(user_id)
    if session_id is not None:
        attributes["session.id"] = str(session_id)
        attributes["session_id"] = str(session_id)
    if isinstance(tags, (list, tuple)):
        attributes["tags"] = [str(tag) for tag in tags if tag is not None]
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            attributes[f"meta.{key}"] = value

    attributes = _normalise_attributes(attributes)
    _apply_attributes(span, attributes)


def start_trace(
    *,
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Start a root trace using OpenTelemetry."""

    if not tracing_enabled():
        return
    tracer = _get_tracer()
    if tracer is None:
        return

    attributes = {
        "user.id": user_id or "",
        "session.id": session_id or "",
    }
    if metadata:
        for key, value in metadata.items():
            attributes[f"meta.{key}"] = value
    attributes = _normalise_attributes(attributes)

    try:
        cm = tracer.start_as_current_span(name, attributes=attributes)
    except Exception:
        return
    try:
        cm.__enter__()
    except Exception:
        return
    _OTEL_ROOT_CM.set(cm)


def end_trace() -> None:
    """End the current root trace started via :func:`start_trace`."""

    cm = _OTEL_ROOT_CM.get()
    if cm is None:
        return
    try:
        cm.__exit__(None, None, None)
    except Exception:
        pass
    finally:
        _OTEL_ROOT_CM.set(None)


def record_span(
    name: str,
    *,
    attributes: Optional[dict[str, Any]] = None,
    trace_id: str | None = None,
) -> None:
    """Record a standalone span with optional metadata."""

    if not tracing_enabled():
        return
    tracer = _get_tracer()
    if tracer is None:
        return

    span_attributes = _normalise_attributes(attributes or {})
    if trace_id is not None:
        span_attributes.setdefault("legacy.trace_id", str(trace_id))

    try:
        cm = tracer.start_as_current_span(name, attributes=span_attributes)
    except Exception:
        return
    try:
        cm.__enter__()
    except Exception:
        return
    try:
        span = _get_current_span()
        if span is not None and span_attributes:
            _apply_attributes(span, span_attributes)
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def emit_event(payload: dict[str, Any]) -> None:
    """Emit an observability event attached to the current span when possible."""

    if not isinstance(payload, dict):
        return
    event_name = str(payload.get("event") or "event")
    attributes = {k: v for k, v in payload.items() if k != "event"}
    span = _get_current_span()
    if span is not None:
        add_event = getattr(span, "add_event", None)
        if callable(add_event):
            try:
                add_event(event_name, attributes=_normalise_attributes(attributes))
                return
            except Exception:
                pass
    output_payload = {"event": event_name, **attributes}
    try:
        print(json.dumps(output_payload))
    except Exception:
        print(str(output_payload))


def get_langchain_callbacks() -> Iterable[Any]:
    """Return LangChain callbacks; fall back to empty tuple.

    Prefer the Langfuse callback handler when available. This integrates
    LangChain runs with the same Langfuse project configured via ENV
    (LANGFUSE_*). If the dependency is missing, return an empty iterable.
    """

    try:  # pragma: no cover - optional dependency
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler  # type: ignore

        handler = LangfuseCallbackHandler()
        return (handler,)
    except Exception:
        return ()
