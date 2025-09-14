"""Node-level tracing helpers."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, TypeVar, cast

import requests

F = TypeVar("F", bound=Callable[..., Any])


def trace_meta(meta: dict[str, Any], prompt_version: str | None) -> dict[str, Any]:
    """Return metadata enriched with ``prompt_version`` when provided."""

    enriched = dict(meta)
    if prompt_version is not None:
        enriched["prompt_version"] = prompt_version
    return enriched


def _log(payload: dict[str, Any]) -> None:
    """Emit a JSON log line to stdout."""

    print(json.dumps(payload))


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
            logging.getLogger(__name__).warning("langfuse dispatch failed: %s", exc)

    threading.Thread(target=_send, daemon=True).start()


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
            start_payload = {
                "event": "node.start",
                "node": node_name,
                "tenant": meta_enriched.get("tenant"),
                "case": meta_enriched.get("case"),
                "trace_id": meta_enriched.get("trace_id"),
                "prompt_version": meta_enriched.get("prompt_version"),
                "ts": start_ts,
            }
            _log(start_payload)

            try:
                return func(*args, **kwargs)
            finally:
                end_ts = time.time()
                end_payload = {
                    "event": "node.end",
                    "node": node_name,
                    "tenant": meta_enriched.get("tenant"),
                    "case": meta_enriched.get("case"),
                    "trace_id": meta_enriched.get("trace_id"),
                    "prompt_version": meta_enriched.get("prompt_version"),
                    "ts": end_ts,
                    "duration_ms": int((end_ts - start_ts) * 1000),
                }
                _log(end_payload)

                _dispatch_langfuse(
                    trace_id=str(meta_enriched.get("trace_id")),
                    node_name=node_name,
                    metadata={
                        "tenant": meta_enriched.get("tenant"),
                        "case": meta_enriched.get("case"),
                        "prompt_version": meta_enriched.get("prompt_version"),
                    },
                )

        return cast(F, wrapped)

    return decorator
