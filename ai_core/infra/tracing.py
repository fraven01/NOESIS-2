"""Node-level tracing helpers."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar, cast

import requests

from common.logging import get_log_context, get_logger

F = TypeVar("F", bound=Callable[..., Any])


logger = get_logger(__name__)


def _normalise_collection_value(value: object) -> str | None:
    """Return a trimmed collection identifier or ``None`` when absent."""

    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
    else:
        try:
            candidate = str(value).strip()
        except Exception:
            return None
    return candidate or None


def trace_meta(meta: dict[str, Any], prompt_version: str | None) -> dict[str, Any]:
    """Return metadata enriched with ``prompt_version`` when provided."""

    enriched = dict(meta)
    if prompt_version is not None:
        enriched["prompt_version"] = prompt_version
    return enriched


def _log(payload: dict[str, Any]) -> None:
    """Emit a JSON log line to stdout."""

    print(json.dumps(payload))


def emit_event(payload: dict[str, Any]) -> None:
    """Public helper to emit structured tracing events.

    Tests and instrumentation code should rely on this function rather than
    patching the private :func:`_log` helper so refactors remain encapsulated.
    """

    _log(payload)


_LANGFUSE_EXECUTOR: ThreadPoolExecutor | None = None
_LANGFUSE_EXECUTOR_LOCK = threading.Lock()


def _shutdown_langfuse_executor() -> None:
    global _LANGFUSE_EXECUTOR
    executor = _LANGFUSE_EXECUTOR
    if executor is not None:
        executor.shutdown(wait=False)
        _LANGFUSE_EXECUTOR = None


atexit.register(_shutdown_langfuse_executor)


def _get_langfuse_executor() -> ThreadPoolExecutor:
    """Return (and lazily create) the shared Langfuse executor."""

    global _LANGFUSE_EXECUTOR
    if _LANGFUSE_EXECUTOR is None:
        with _LANGFUSE_EXECUTOR_LOCK:
            if _LANGFUSE_EXECUTOR is None:
                max_workers_raw = os.getenv("LANGFUSE_MAX_WORKERS", "2")
                try:
                    max_workers = max(1, int(max_workers_raw))
                except ValueError:
                    logger.warning(
                        "invalid LANGFUSE_MAX_WORKERS %r, defaulting to 2",
                        max_workers_raw,
                    )
                    max_workers = 2

                _LANGFUSE_EXECUTOR = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="langfuse",
                )

    return cast(ThreadPoolExecutor, _LANGFUSE_EXECUTOR)


def _dispatch_langfuse(trace_id: str, node_name: str, metadata: dict[str, Any]) -> None:
    """Send a tracing event to Langfuse in the background if credentials exist."""

    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        return

    url = os.getenv(
        "LANGFUSE_BASE_URL",
        "https://cloud.langfuse.com/api/public/ingest",
    )

    payload = {"traceId": trace_id, "name": node_name, "metadata": metadata}
    headers = {"X-Langfuse-Public-Key": public, "X-Langfuse-Secret-Key": secret}

    def _send() -> None:
        try:
            requests.post(url, json=payload, headers=headers, timeout=2)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("langfuse dispatch failed: %s", exc)

    executor = _get_langfuse_executor()
    executor.submit(_send)


def emit_span(trace_id: str, node_name: str, metadata: dict[str, Any]) -> None:
    """Public helper to dispatch Langfuse spans.

    Exposes a stable interface for tests that want to assert emitted spans
    without reaching into the module's private internals.
    """

    metadata_payload = dict(metadata)
    collection_value = _normalise_collection_value(
        metadata_payload.get("collection_id")
    )
    if collection_value is None:
        context_collection = _normalise_collection_value(
            get_log_context().get("collection_id")
        )
        metadata_payload["collection_id"] = context_collection or ""
    else:
        metadata_payload["collection_id"] = collection_value

    _dispatch_langfuse(
        trace_id=trace_id,
        node_name=node_name,
        metadata=metadata_payload,
    )


def trace(node_name: str) -> Callable[[F], F]:
    """Decorator emitting start/end logs and optional Langfuse events."""

    def decorator(func: F) -> F:
        def wrapped(*args: Any, **kwargs: Any):  # type: ignore[misc]
            meta = kwargs.get("meta")
            if meta is None and len(args) > 1:
                meta = args[1]
            if not isinstance(meta, dict):
                meta = {}

            meta_enriched = trace_meta(meta, meta.get("prompt_version"))

            start_ts = time.time()
            tenant_value = meta_enriched.get("tenant_id")
            case_value = meta_enriched.get("case_id")
            raw_collection_value = meta_enriched.get("collection_id")
            collection_value = _normalise_collection_value(raw_collection_value)
            collection_payload = collection_value or ""

            start_payload = {
                "event": "node.start",
                "node": node_name,
                "tenant_id": tenant_value,
                "case_id": case_value,
                "trace_id": meta_enriched.get("trace_id"),
                "prompt_version": meta_enriched.get("prompt_version"),
                "ts": start_ts,
            }
            start_payload["collection_id"] = collection_payload
            emit_event(start_payload)

            try:
                return func(*args, **kwargs)
            finally:
                end_ts = time.time()
                end_payload = {
                    "event": "node.end",
                    "node": node_name,
                    "tenant_id": tenant_value,
                    "case_id": case_value,
                    "trace_id": meta_enriched.get("trace_id"),
                    "prompt_version": meta_enriched.get("prompt_version"),
                    "ts": end_ts,
                    "duration_ms": int((end_ts - start_ts) * 1000),
                }
                end_payload["collection_id"] = collection_payload
                emit_event(end_payload)

                trace_id = meta_enriched.get("trace_id")
                if isinstance(trace_id, str):
                    trace_id = trace_id.strip()

                if trace_id:
                    emit_span(
                        trace_id=str(trace_id),
                        node_name=node_name,
                        metadata={
                            "tenant_id": tenant_value,
                            "case_id": case_value,
                            "prompt_version": meta_enriched.get("prompt_version"),
                            "collection_id": collection_payload,
                        },
                    )

        return cast(F, wrapped)

    return decorator
