"""Middleware utilities for API versioning."""

from __future__ import annotations

from typing import Callable

from django.http import HttpRequest, HttpResponse


class DeprecationHeaderMiddleware:
    """Ensure responses include deprecation metadata when configured."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if not hasattr(request, "_api_deprecation_headers"):
            request._api_deprecation_headers = {}
        response = self.get_response(request)
        headers = getattr(request, "_api_deprecation_headers", None)
        if isinstance(headers, dict):
            for name, value in headers.items():
                if value and name not in response:
                    response[name] = value
        return response


__all__ = ["DeprecationHeaderMiddleware"]
