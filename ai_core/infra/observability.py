"""Lightweight observability adapters (Langfuse SDK if available).

This module avoids hard dependencies: if the `langfuse` package is not
installed, all helpers degrade to no-ops so local dev remains smooth.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Any, Callable, Iterable, Optional

try:  # Optional Langfuse SDK
    from langfuse.decorators import observe as _lf_observe  # type: ignore
    from langfuse.decorators import langfuse_context as _lf_context  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _lf_observe = None
    _lf_context = None


class _NoopContext(ContextDecorator):
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        return False


def observe_span(
    name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that creates a Langfuse span when SDK is present.

    If the Langfuse SDK is not installed/configured, this is a no-op.
    """

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if _lf_observe is None:  # no SDK → noop
            return fn
        # Use built-in decorator, pass through name if provided
        if name:
            return _lf_observe(name=name)(fn)  # type: ignore[misc]
        return _lf_observe()(fn)  # type: ignore[misc]

    return _decorator


def update_observation(**fields: Any) -> None:
    """Attach metadata/tags to the current observation if SDK is present.

    Common fields: user_id, session_id, tags (list[str]), metadata (dict).
    """

    if _lf_context is None:  # pragma: no cover - optional dependency
        return
    try:
        _lf_context.update_current_observation(**fields)  # type: ignore[attr-defined]
    except Exception:
        # Never break application flow due to observability issues
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
