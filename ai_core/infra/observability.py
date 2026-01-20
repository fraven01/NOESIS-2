"""OpenTelemetry-backed observability helpers.

All functions in this module gracefully degrade to no-ops when the
OpenTelemetry SDK is not available or tracing is disabled via
configuration. This keeps local development lightweight while providing a
clear path to production-grade tracing.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import base64
import functools
import inspect
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Optional, TypeVar, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from .usage import Usage

LOGGER = logging.getLogger(__name__)

# Suppress noisy OTEL exporter tracebacks on connection errors.
# These modules log full tracebacks when Langfuse is temporarily unavailable.
_OTEL_NOISY_LOGGERS = [
    "opentelemetry.sdk._shared_internal",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.exporter.otlp.proto.http",
    "urllib3.connectionpool",
]


def _configure_otel_logging() -> None:
    """Reduce OTEL exporter log verbosity to avoid traceback spam."""
    level_str = (os.getenv("OTEL_EXPORTER_LOG_LEVEL") or "WARNING").upper()
    level = getattr(logging, level_str, logging.WARNING)
    for logger_name in _OTEL_NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)


# Apply immediately on import
_configure_otel_logging()

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


def observe_span(
    name: Optional[str] = None,
    *,
    auto_annotate: bool = False,
) -> Callable[[F], F]:
    """Decorator that wraps the target in an OpenTelemetry span when available.

    When ``auto_annotate`` is ``True`` and the wrapped callable is a bound
    method whose first argument exposes ``_annotate_span``, the helper will be
    invoked after the wrapped function returns while the span context is still
    active. This allows callers to attach metadata to the span before it
    closes.
    """

    def decorator(func: F) -> F:
        span_name = name or getattr(func, "__name__", "observe")
        phase = _derive_phase(span_name, getattr(func, "__name__", None))

        if inspect.iscoroutinefunction(func):

            async def _run_async(*args: Any, **kwargs: Any):  # type: ignore[misc]
                result = await func(*args, **kwargs)
                if auto_annotate:
                    _maybe_annotate_span(phase, args, kwargs, result)
                return result

            @functools.wraps(func)
            async def wrapped(*args: Any, **kwargs: Any):  # type: ignore[misc]
                if not tracing_enabled():
                    return await _run_async(*args, **kwargs)
                tracer = _get_tracer()
                if tracer is None:
                    return await _run_async(*args, **kwargs)
                try:
                    cm = tracer.start_as_current_span(
                        span_name, kind=_resolve_span_kind(None)
                    )
                except Exception:
                    return await _run_async(*args, **kwargs)
                with cm:
                    return await _run_async(*args, **kwargs)

            return cast(F, wrapped)

        def _run(*args: Any, **kwargs: Any):  # type: ignore[misc]
            result = func(*args, **kwargs)
            if auto_annotate:
                _maybe_annotate_span(phase, args, kwargs, result)
            return result

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any):  # type: ignore[misc]
            if not tracing_enabled():
                return _run(*args, **kwargs)
            tracer = _get_tracer()
            if tracer is None:
                return _run(*args, **kwargs)
            try:
                cm = tracer.start_as_current_span(
                    span_name, kind=_resolve_span_kind(None)
                )
            except Exception:
                return _run(*args, **kwargs)
            with cm:
                return _run(*args, **kwargs)

        return cast(F, wrapped)

    return decorator


def _derive_phase(span_name: str, func_name: Optional[str]) -> str:
    if "." in span_name:
        return span_name.rsplit(".", 1)[-1]
    if func_name:
        if func_name.startswith("_node_"):
            return func_name[len("_node_") :]
        return func_name
    return span_name


def _maybe_annotate_span(
    phase: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> None:
    if not args:
        return
    target = args[0]
    annotate = getattr(target, "_annotate_span", None)
    if not callable(annotate):
        return
    state = None
    if len(args) >= 2:
        state = args[1]
    if state is None:
        state = kwargs.get("state")
    if state is None:
        return
    try:
        annotate(state, phase=phase, transition=result)
    except TypeError:
        try:
            annotate(state, phase=phase)
        except TypeError:
            annotate(state, phase=phase, transition=result)


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


def _resolve_span_kind(kind: str | None) -> Any | None:
    if not kind:
        try:
            from opentelemetry.trace import SpanKind

            return SpanKind.INTERNAL
        except Exception:
            return None
    try:
        from opentelemetry.trace import SpanKind

        kind_upper = kind.upper()
        if kind_upper == "SERVER":
            return SpanKind.SERVER
        if kind_upper == "CLIENT":
            return SpanKind.CLIENT
        if kind_upper == "PRODUCER":
            return SpanKind.PRODUCER
        if kind_upper == "CONSUMER":
            return SpanKind.CONSUMER
        if kind_upper == "INTERNAL":
            return SpanKind.INTERNAL
    except Exception:
        return None
    return None


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


def report_generation_usage(usage: "Usage", model: str | None = None) -> None:
    """Attach generation-related usage metrics to the active span."""
    span = _get_current_span()
    if span is None:
        return

    attributes = {
        "gen_ai.usage.input_tokens": usage.input_tokens,
        "gen_ai.usage.output_tokens": usage.output_tokens,
        "gen_ai.usage.total_tokens": usage.total_tokens,
    }
    if usage.cost_usd is not None:
        attributes["gen_ai.usage.cost"] = usage.cost_usd
    if model:
        attributes["gen_ai.request.model"] = model

    _apply_attributes(span, attributes)


def start_trace(
    *,
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    kind: str | None = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Any | None:
    """Start a root trace using OpenTelemetry.

    Supports manual context propagation via trace_id/parent_id.
    """

    if not tracing_enabled():
        return None
    tracer = _get_tracer()
    if tracer is None:
        return None

    # Merge explicitly provided attributes with metadata/user info
    final_attributes: dict[str, Any] = dict(attributes or {})
    if user_id:
        final_attributes["user.id"] = user_id
    if session_id:
        final_attributes["session.id"] = session_id
    if metadata:
        for key, value in metadata.items():
            final_attributes[f"meta.{key}"] = value

    final_attributes = _normalise_attributes(final_attributes)

    # Resolve Context (Parent/Distributed)
    context = None
    if trace_id:
        try:
            from opentelemetry.trace import (
                SpanContext,
                TraceFlags,
                set_span_in_context,
                NonRecordingSpan,
            )

            # We need a 128-bit hex trace_id (32 chars) and 64-bit hex span_id (16 chars)
            # UUIDs are 32 chars hex, so they map directly to trace_id.
            t_id = int(trace_id, 16)
            s_id = int(parent_id, 16) if parent_id else 0

            # If s_id is 0, we can't create a valid parent context for strict tree structures.
            # However, if we simply want to force the trace_id, we need a parent with that trace_id.
            # We'll treat it as a remote parent.
            if t_id:
                # If no parent ID provided, use a dummy or random one?
                # Or 0? OTel might reject 0 span_id.
                # Let's hope parent_id is passed or handle gracefully.
                if s_id == 0:
                    # Generate a random span ID for the "virtual" parent to anchor the trace?
                    import random

                    s_id = random.getrandbits(64)

                span_context = SpanContext(
                    trace_id=t_id,
                    span_id=s_id,
                    is_remote=True,
                    trace_flags=TraceFlags.SAMPLED,
                )
                # Wrap in NonRecordingSpan to satisfy set_span_in_context
                parent_span = NonRecordingSpan(span_context)
                context = set_span_in_context(parent_span)
        except Exception:
            # Fallback to standard new trace if ID parsing fails
            pass

    # Resolve SpanKind
    otel_kind = _resolve_span_kind(kind)

    try:
        cm = tracer.start_as_current_span(
            name, context=context, kind=otel_kind, attributes=final_attributes
        )
    except Exception:
        return None

    try:
        span = cm.__enter__()
    except Exception:
        return None

    _OTEL_ROOT_CM.set(cm)
    return span


def end_trace(span: Any | None = None) -> None:
    """End the current root trace started via :func:`start_trace`."""

    cm = _OTEL_ROOT_CM.get()
    if cm is None:
        return
    try:
        # We ignore the specific 'span' argument for now and rely on the ContextVar
        # Stack-like behavior isn't fully implemented here, assuming single root per task context.
        cm.__exit__(None, None, None)
    except Exception:
        pass
    finally:
        _OTEL_ROOT_CM.set(None)


def record_exception(span: Any, exc: BaseException) -> None:
    """Record an exception on the given span."""
    if span is None:
        return

    recorder = getattr(span, "record_exception", None)
    if callable(recorder):
        try:
            recorder(exc)
        except Exception:
            pass


def record_span(
    name: str,
    *,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record a standalone span with optional metadata."""

    if not tracing_enabled():
        return
    tracer = _get_tracer()
    if tracer is None:
        return

    span_attributes = _normalise_attributes(attributes or {})

    try:
        cm = tracer.start_as_current_span(
            name,
            attributes=span_attributes,
            kind=_resolve_span_kind(None),
        )
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


def emit_event(event: Any, attributes: Optional[dict[str, Any]] = None) -> None:
    """Emit an observability event attached to the current span when possible."""

    if isinstance(event, dict):
        event_name = str(event.get("event") or "event")
        attrs = {k: v for k, v in event.items() if k != "event"}
    else:
        event_name = str(event or "event")
        attrs = dict(attributes or {})
    span = _get_current_span()
    if span is not None:
        add_event = getattr(span, "add_event", None)
        if callable(add_event):
            try:
                add_event(event_name, attributes=_normalise_attributes(attrs))
                return
            except Exception:
                pass
    output_payload = {"event": event_name, **attrs}
    normalised = _normalise_attributes(output_payload)
    event_label = normalised.pop("event", event_name)
    LOGGER.info("observability.event", extra={"event": event_label, **normalised})


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
