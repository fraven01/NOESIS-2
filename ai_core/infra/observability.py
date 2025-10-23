"""Lightweight observability adapters (Langfuse SDK if available).

This module avoids hard dependencies: if the `langfuse` package is not
installed, all helpers degrade to no-ops so local dev remains smooth.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Optional
import os

try:  # Optional Langfuse SDK (v2; v3 removes decorators)
    from langfuse.decorators import observe as _lf_observe  # type: ignore
    from langfuse.decorators import langfuse_context as _lf_context  # type: ignore
except Exception:  # pragma: no cover - compatibility for newer SDKs
    try:
        from langfuse import observe as _lf_observe  # type: ignore
    except Exception:
        _lf_observe = None
    # v3 no longer exposes langfuse_context
    _lf_context = None  # type: ignore

try:  # v3 SDK helper to inspect active client
    from langfuse import get_client as _lf_get_client  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _lf_get_client = None  # type: ignore

try:  # v3 SDK client class for explicit initialisation
    from langfuse import Langfuse as _LfClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _LfClient = None  # type: ignore

try:  # Optional OpenTelemetry for root spans in v3
    from opentelemetry.trace import (
        get_tracer as _otel_get_tracer,  # type: ignore
        get_current_span as _otel_get_current_span,  # type: ignore
    )
except Exception:  # pragma: no cover - optional dependency
    _otel_get_tracer = None  # type: ignore
    _otel_get_current_span = None  # type: ignore


class _NoopContext(ContextDecorator):
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False


# Context-local handles for root traces/spans so concurrent requests are safe
_LF_ROOT_TRACE: ContextVar[Any | None] = ContextVar("LF_ROOT_TRACE", default=None)
_OTEL_ROOT_CM: ContextVar[Any | None] = ContextVar("OTEL_ROOT_CM", default=None)


def _ensure_client_initialised() -> None:
    """Initialise a Langfuse v3 client from ENV when none is active.

    Best-effort and safe to call multiple times. No-ops when SDK is absent
    or credentials are missing.
    """
    # If OTel exporter is in use, prefer not to create a client at all to
    # avoid synchronous SDK initialisation or background media threads.
    disable_flag = (os.getenv("LANGFUSE_DISABLE_CLIENT_INIT") or "").strip().lower()
    if disable_flag in {"1", "true", "yes", "on"}:
        return
    otel_exporter = (os.getenv("OTEL_TRACES_EXPORTER") or "").strip().lower()
    force_client = (os.getenv("LANGFUSE_FORCE_CLIENT") or "").strip()
    if otel_exporter == "otlp" and not force_client:
        return

    if _lf_get_client is None or _LfClient is None:
        return
    try:
        existing = _lf_get_client()  # type: ignore[misc]
    except Exception:
        existing = None
    if existing is not None:
        return

    public = os.getenv("LANGFUSE_PUBLIC_KEY") or ""
    secret = os.getenv("LANGFUSE_SECRET_KEY") or ""
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or ""
    if not (public and secret):
        return
    try:  # pragma: no cover - optional dependency
        _LfClient(public_key=public, secret_key=secret, host=(host or None))
    except Exception:
        return


def observe_span(
    name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records a span when tracing is enabled.

    Preference order:
    - Use OpenTelemetry span when available (v3 recommended path).
    - Use Langfuse SDK decorator only when explicitly allowed via LANGFUSE_USE_SDK_DECORATOR.
    - Otherwise no-op.
    """

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # If tracing is disabled, return the original function
        try:
            enabled = tracing_enabled()
        except Exception:
            enabled = False
        if not enabled:
            return fn

        # Prefer OTel span if available
        if _otel_get_tracer is not None:

            def _wrapped(*args: Any, **kwargs: Any):  # type: ignore[misc]
                try:
                    tracer = _otel_get_tracer("noesis2")  # type: ignore[misc]
                    span_name = name or getattr(fn, "__name__", "observe")
                    with tracer.start_as_current_span(span_name):
                        return fn(*args, **kwargs)
                except Exception:
                    return fn(*args, **kwargs)

            return _wrapped  # type: ignore[return-value]

        # Optionally fall back to SDK decorator when explicitly requested
        use_sdk = (os.getenv("LANGFUSE_USE_SDK_DECORATOR") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_sdk and _lf_observe is not None:
            if name:
                return _lf_observe(name=name)(fn)  # type: ignore[misc]
            return _lf_observe()(fn)  # type: ignore[misc]

        # No tracing backend available
        return fn

    return _decorator


def update_observation(**fields: Any) -> None:
    """Attach metadata/tags to the current observation if SDK is present.

    Common fields: user_id, session_id, tags (list[str]), metadata (dict).
    """

    # 1) v2 decorator context
    if _lf_context is not None:  # pragma: no cover - optional dependency
        try:
            _lf_context.update_current_observation(**fields)  # type: ignore[attr-defined]
            return
        except Exception:
            # Fall through to v3/OTel attempt
            pass

    # 2) v3 client trace handle
    trace = _LF_ROOT_TRACE.get()
    if trace is not None:
        try:
            # Prefer a unified update method when available
            update_fn = getattr(trace, "update", None)
            if callable(update_fn):
                # Build a conservative kwargs payload
                payload: dict[str, Any] = {}
                for key in ("user_id", "session_id", "metadata", "tags"):
                    if key in fields and fields[key] is not None:
                        payload[key] = fields[key]
                # Try to update; ignore unknown-arg errors
                try:
                    update_fn(**payload)  # type: ignore[misc]
                except Exception:
                    pass
            else:
                # Fallback: try to set common attributes directly
                if "user_id" in fields and fields["user_id"] is not None:
                    setattr(trace, "user_id", fields["user_id"])  # type: ignore[attr-defined]
                if "session_id" in fields and fields["session_id"] is not None:
                    setattr(trace, "session_id", fields["session_id"])  # type: ignore[attr-defined]
                if "tags" in fields and fields["tags"] is not None:
                    try:
                        # Merge with existing tags when possible
                        existing = getattr(trace, "tags", None)
                        if isinstance(existing, list):
                            merged = list({*existing, *list(fields["tags"] or [])})
                            setattr(trace, "tags", merged)
                        else:
                            setattr(trace, "tags", fields["tags"])  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if "metadata" in fields and isinstance(fields["metadata"], dict):
                    try:
                        meta_obj = getattr(trace, "metadata", None) or {}
                        if isinstance(meta_obj, dict):
                            meta_obj.update(fields["metadata"])  # type: ignore[arg-type]
                            setattr(trace, "metadata", meta_obj)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            # Do not fail app flow due to observability enrichments
            pass

    # 3) OTel current span enrichment
    if _otel_get_current_span is not None:
        try:
            span = _otel_get_current_span()  # type: ignore[misc]
            if span is not None:
                set_attr = getattr(span, "set_attribute", None)
                if callable(set_attr):
                    user_id = fields.get("user_id")
                    session_id = fields.get("session_id")
                    tags = fields.get("tags")
                    metadata = fields.get("metadata")
                    if user_id is not None:
                        try:
                            set_attr("user.id", str(user_id))
                            set_attr("user_id", str(user_id))
                        except Exception:
                            pass
                    if session_id is not None:
                        try:
                            set_attr("session.id", str(session_id))
                            set_attr("session_id", str(session_id))
                        except Exception:
                            pass
                    if isinstance(tags, (list, tuple)):
                        try:
                            set_attr("tags", list(tags))
                        except Exception:
                            pass
                    if isinstance(metadata, dict):
                        for k, v in metadata.items():
                            try:
                                set_attr(f"meta.{k}", v)
                            except Exception:
                                pass
        except Exception:
            # Best-effort only
            pass
    return


def get_langchain_callbacks() -> Iterable[Any]:
    """Return LangChain callbacks for Langfuse if available.

    Usage (when adopting LangChain):
        llm = ChatOpenAI(..., callbacks=list(get_langchain_callbacks()))
    """

    try:
        # Langfuse provides a LangChain callback handler
        from langfuse.callback import CallbackHandler  # type: ignore

        return (CallbackHandler(),)
    except Exception:  # pragma: no cover - optional dependency
        return ()


# -------- Root trace helpers (optional SDK) --------
def start_trace(
    *,
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Start a root trace compatible with Langfuse v3.

    Preference order:
    - Use `langfuse.get_client().trace(...)` when available (v3).
    - Otherwise, create an OpenTelemetry root span if OTel is present.
    - As a last resort, attempt v2 `langfuse_context.start_trace` when present.

    Always safe to call: falls back to no-op if neither backend is configured.
    """

    _ensure_client_initialised()

    # Prefer OTel root span for v3 integrations to avoid synchronous
    # network calls when creating client-side traces. Langfuse v3 ingests
    # OTel spans via the exporter.
    if _otel_get_tracer is not None:  # pragma: no cover - optional dependency
        try:
            tracer = _otel_get_tracer("noesis2")  # type: ignore[misc]
            cm = tracer.start_as_current_span(
                name,
                attributes={
                    "user_id": user_id or "",
                    "session_id": session_id or "",
                    **(metadata or {}),
                },
            )
            cm.__enter__()
            _OTEL_ROOT_CM.set(cm)
            return
        except Exception:
            # Fall through to client/v2 when OTel is unavailable
            pass

    # As a secondary option, create a v3 client trace handle
    if _lf_get_client is not None:  # pragma: no cover - optional dependency
        try:
            client = _lf_get_client()  # type: ignore[misc]
        except Exception:
            client = None
        if client is not None:
            try:
                trace = client.trace(  # type: ignore[attr-defined]
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata or {},
                )
                _LF_ROOT_TRACE.set(trace)
                return
            except Exception:
                pass

    # v2 fallback (rare; retained for compatibility in mixed environments)
    if _lf_context is not None:  # pragma: no cover - optional dependency
        try:
            _lf_context.start_trace(  # type: ignore[attr-defined]
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
        except Exception:
            return


def end_trace() -> None:
    """End the current root trace/span across supported backends.

    - Ends v3 `client.trace(...)` via `.end()` and flushes the client if possible.
    - Ends the OpenTelemetry span context manager begun in `start_trace()`.
    - Falls back to v2 `langfuse_context.end_trace()` if still present.
    """

    # Prefer v3 trace handle when available
    trace = _LF_ROOT_TRACE.get()
    if trace is not None:
        try:
            # End the trace on the object
            end = getattr(trace, "end", None)
            if callable(end):
                end()
        except Exception:
            pass
        finally:
            _LF_ROOT_TRACE.set(None)
        # Best-effort, non-blocking flush on the client
        # In some environments the synchronous flush may block when Langfuse is
        # slow or unreachable. To avoid impacting request latency, run flush in
        # a short-lived background thread and proceed if it doesn't finish
        # quickly.
        if _lf_get_client is not None:
            try:
                client = _lf_get_client()  # type: ignore[misc]
                flush = getattr(client, "flush", None)
                if callable(flush):
                    import threading

                    timeout_ms_raw = os.getenv("LANGFUSE_FLUSH_TIMEOUT_MS", "200")
                    try:
                        timeout_ms = max(0, int(timeout_ms_raw))
                    except Exception:
                        timeout_ms = 200

                    def _do_flush():  # pragma: no cover - timing dependent
                        try:
                            flush()
                        except Exception:
                            pass

                    t = threading.Thread(
                        target=_do_flush, name="langfuse-flush", daemon=True
                    )
                    t.start()
                    # Wait briefly, then continue even if flush is still ongoing
                    t.join(timeout=timeout_ms / 1000.0)
            except Exception:
                pass
        return

    # Close OTel span if we started one
    cm = _OTEL_ROOT_CM.get()
    if cm is not None:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
        finally:
            _OTEL_ROOT_CM.set(None)
        return

    # v2 fallback
    if _lf_context is not None:  # pragma: no cover - optional dependency
        try:
            _lf_context.end_trace()  # type: ignore[attr-defined]
        except Exception:
            return


def sdk_active() -> bool:
    """Return True if a Langfuse SDK client appears to be initialised.

    Works with v3 via ``get_client()``, otherwise falls back to decorator presence.
    """

    if _lf_get_client is not None:  # pragma: no cover - optional
        try:
            return _lf_get_client() is not None  # type: ignore[misc]
        except Exception:
            return False
    return _lf_observe is not None


def root_span(name: str, *, attributes: Optional[dict[str, Any]] = None):
    """Return a context manager that starts an OpenTelemetry span if available.

    Integrates with Langfuse v3 (OTel exporter). Falls back to a no-op when OTel
    is not present.
    """

    if _otel_get_tracer is None:  # pragma: no cover - optional dependency
        return _NoopContext()
    try:
        tracer = _otel_get_tracer("noesis2")  # type: ignore[misc]
        return tracer.start_as_current_span(name, attributes=attributes or {})
    except Exception:
        return _NoopContext()


# -------- Enable/disable helpers --------
def tracing_enabled() -> bool:
    """Return True if Langfuse/OTel tracing should be attempted.

    Heuristics:
    - Disabled when LANGFUSE_SAMPLE_RATE is 0
    - Disabled when OTEL_TRACES_EXPORTER=none
    - Disabled when LANGFUSE credentials are missing
    """

    sample = (os.getenv("LANGFUSE_SAMPLE_RATE") or "1").strip().lower()
    if sample in {"0", "0.0"}:
        return False
    otel_exporter = (os.getenv("OTEL_TRACES_EXPORTER") or "").strip().lower()
    if otel_exporter == "none":
        return False
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    return True
