"""Tracing decorator forwarding metadata to the logging stub.

Each orchestrator node is wrapped so that a log entry is emitted for every
invocation.  The log payload mirrors the structure expected by Langfuse and
contains the tenant/case identifiers together with trace and prompt metadata.

The wrapped function must accept a ``meta`` keyword argument containing the
required information.  Missing keys are tolerated to keep the stub lightweight.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict

from ...infra import logging


def trace(node: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that logs invocation metadata for ``node``."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            meta: Dict[str, Any] = kwargs.get("meta", {}) or {}
            result = func(*args, **kwargs)
            prompt_version = meta.get("prompt_version")
            if isinstance(result, dict):
                prompt_version = prompt_version or result.get("prompt_version")
            payload = {
                "tenant": meta.get("tenant"),
                "case": meta.get("case"),
                "trace": meta.get("trace"),
                "prompt_version": prompt_version,
                "node": node,
            }
            logging.event("trace", payload)
            return result

        return wrapper

    return decorator
