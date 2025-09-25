"""Branded documentation views for drf-spectacular."""

from __future__ import annotations

from django.conf import settings
from django.http import Http404

from drf_spectacular.views import SpectacularRedocView, SpectacularSwaggerView


def _build_docs_title() -> str:
    """Compose the UI title including the optional deployment version label."""

    base_title = getattr(
        settings, "API_DOCS_TITLE", settings.SPECTACULAR_SETTINGS["TITLE"]
    )
    version_label = getattr(settings, "API_DOCS_VERSION_LABEL", "").strip()
    if version_label:
        return f"{base_title} ({version_label})"
    return base_title


class _DocsToggleMixin:
    """Return a 404 when API documentation is disabled for the environment."""

    @staticmethod
    def _ensure_docs_enabled() -> None:
        if not getattr(settings, "ENABLE_API_DOCS", False):
            raise Http404("API documentation is disabled.")


class BrandedSpectacularSwaggerView(_DocsToggleMixin, SpectacularSwaggerView):
    """Swagger UI view that honours the docs toggle and custom branding."""

    def get(self, request, *args, **kwargs):  # noqa: D401 - inherited behaviour
        self._ensure_docs_enabled()
        self.title = _build_docs_title()
        return super().get(request, *args, **kwargs)


class BrandedSpectacularRedocView(_DocsToggleMixin, SpectacularRedocView):
    """Redoc view mirroring the Swagger UI toggle and branding."""

    def get(self, request, *args, **kwargs):  # noqa: D401 - inherited behaviour
        self._ensure_docs_enabled()
        self.title = _build_docs_title()
        return super().get(request, *args, **kwargs)
