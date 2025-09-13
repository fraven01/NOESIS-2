"""Helpers for setting common response headers."""

from __future__ import annotations

from fastapi import Response


def apply_std_headers(response: Response, prompt_version: str, trace_id: str) -> None:
    """Attach tracing metadata to a response object."""
    response.headers["x-trace-id"] = trace_id
    response.headers["x-prompt-version"] = prompt_version
