"""Helpers for API versioning and deprecation headers."""

from __future__ import annotations

from typing import Dict, Mapping
from django.conf import settings
from django.http import HttpRequest, HttpResponse

_DEFAULT_DEPRECATION_VALUE = "true"


def _normalise_config(config: Mapping[str, Mapping[str, str]] | None) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(config, Mapping):
        return {}
    normalised: Dict[str, Mapping[str, str]] = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            normalised[str(key)] = value
    return normalised


def _resolve_deprecation_config(config_key: str | None) -> Mapping[str, str]:
    if not config_key:
        return {}
    config = _normalise_config(getattr(settings, "API_DEPRECATIONS", None))
    value = config.get(config_key)
    if isinstance(value, Mapping):
        return value
    return {}


def build_deprecation_headers(
    *,
    deprecated: bool,
    config_key: str | None = None,
    override_deprecation: str | None = None,
    override_sunset: str | None = None,
) -> Dict[str, str]:
    """Return HTTP headers describing an endpoint's deprecation policy."""

    if not deprecated:
        return {}

    config = _resolve_deprecation_config(config_key)
    deprecation_value = override_deprecation or config.get("deprecation")
    if not deprecation_value:
        deprecation_value = _DEFAULT_DEPRECATION_VALUE

    headers: Dict[str, str] = {"Deprecation": deprecation_value}

    sunset_value = override_sunset or config.get("sunset")
    if sunset_value:
        headers["Sunset"] = sunset_value

    return headers


def _store_headers(request: HttpRequest, headers: Mapping[str, str]) -> None:
    if not headers:
        return
    existing = getattr(request, "_api_deprecation_headers", None)
    if not isinstance(existing, dict):
        existing = {}
    for name, value in headers.items():
        if value:
            existing[name] = value
    request._api_deprecation_headers = existing


def mark_deprecated_response(
    request: HttpRequest,
    *,
    config_key: str | None = None,
    deprecation: str | None = None,
    sunset: str | None = None,
) -> Dict[str, str]:
    """Record deprecation headers for function-based views."""

    headers = build_deprecation_headers(
        deprecated=True,
        config_key=config_key,
        override_deprecation=deprecation,
        override_sunset=sunset,
    )
    _store_headers(request, headers)
    return headers


class DeprecationHeadersMixin:
    """Inject Deprecation/Sunset headers for deprecated API responses."""

    api_deprecated: bool = False
    api_deprecation_id: str | None = None
    api_deprecation_value: str | None = None
    api_sunset_value: str | None = None

    def get_deprecation_headers(self) -> Dict[str, str]:
        return build_deprecation_headers(
            deprecated=self.api_deprecated,
            config_key=self.api_deprecation_id,
            override_deprecation=self.api_deprecation_value,
            override_sunset=self.api_sunset_value,
        )

    def finalize_response(  # type: ignore[override]
        self,
        request: HttpRequest,
        response: HttpResponse,
        *args,
        **kwargs,
    ) -> HttpResponse:
        response = super().finalize_response(request, response, *args, **kwargs)
        headers = self.get_deprecation_headers()
        if headers:
            _store_headers(request, headers)
            for name, value in headers.items():
                if value:
                    response[name] = value
        return response


__all__ = [
    "DeprecationHeadersMixin",
    "build_deprecation_headers",
    "mark_deprecated_response",
]
