import io
import socket
from typing import Any, Dict
from unittest import mock

import pytest
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.test import override_settings
from django.urls import path
from rest_framework.response import Response
from rest_framework.test import APIClient
from rest_framework.views import APIView

from common.constants import X_TENANT_ID_HEADER
from noesis2 import healthsrv
from noesis2.api import versioning
from noesis2.api.versioning import (
    DeprecationHeadersMixin,
    build_deprecation_headers,
    mark_deprecated_response,
)
from noesis2.tests.test_api_schema import tenant  # noqa: F401 re-export fixture


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


class VersionedContractView(APIView):
    """Simple endpoint that validates tenant/version headers for tests."""

    def get(self, request):  # type: ignore[override]
        tenant = request.headers.get(X_TENANT_ID_HEADER)
        if not tenant:
            return Response(
                {"detail": "Tenant header is required for versioned endpoints."},
                status=400,
            )

        version = request.headers.get("X-API-Version")
        if not version:
            return Response(
                {"detail": "X-API-Version header is required."},
                status=400,
            )
        if version not in {"legacy", "2024-01-01"}:
            return Response(
                {
                    "detail": "Unsupported API version requested.",
                    "supported": ["legacy", "2024-01-01"],
                },
                status=406,
            )

        if version == "legacy":
            headers = mark_deprecated_response(
                request,
                config_key="ai-core-legacy",
            )
            response = Response({"ok": True, "version": version})
            for name, value in headers.items():
                response[name] = value
            return response

        return Response({"ok": True, "version": version})


class DeprecatedMixinView(DeprecationHeadersMixin, APIView):
    """View that relies on the mixin to emit configured headers."""

    api_deprecated = True
    api_deprecation_id = "ai-core-legacy"

    def get(self, request):  # type: ignore[override]
        return Response({"ok": True})


urlpatterns = [
    path("test/version/", VersionedContractView.as_view(), name="test-version"),
    path("test/mixin/", DeprecatedMixinView.as_view(), name="test-mixin"),
]


@override_settings(ROOT_URLCONF="noesis2.tests.test_health_and_versioning")
@pytest.mark.django_db
def test_version_endpoint_requires_tenant(api_client, tenant):
    response = api_client.get("/test/version/")

    assert response.status_code == 400
    assert response.json()["detail"] == "Tenant header is required for versioned endpoints."


@override_settings(ROOT_URLCONF="noesis2.tests.test_health_and_versioning")
@pytest.mark.django_db
def test_version_endpoint_requires_version_header(api_client, tenant):
    response = api_client.get(
        "/test/version/",
        HTTP_X_TENANT_ID="tenant-x",
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "X-API-Version header is required."


@override_settings(ROOT_URLCONF="noesis2.tests.test_health_and_versioning")
@pytest.mark.django_db
def test_version_endpoint_rejects_unsupported_version(api_client, tenant):
    response = api_client.get(
        "/test/version/",
        HTTP_X_TENANT_ID="tenant-x",
        HTTP_X_API_VERSION="v2",
    )

    assert response.status_code == 406
    payload = response.json()
    assert payload["detail"] == "Unsupported API version requested."
    assert payload["supported"] == ["legacy", "2024-01-01"]


@override_settings(ROOT_URLCONF="noesis2.tests.test_health_and_versioning")
@pytest.mark.django_db
def test_version_endpoint_legacy_version_marks_deprecated(api_client, tenant):
    response = api_client.get(
        "/test/version/",
        HTTP_X_TENANT_ID="tenant-x",
        HTTP_X_API_VERSION="legacy",
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "version": "legacy"}
    assert response["Deprecation"] == settings.API_DEPRECATIONS["ai-core-legacy"]["deprecation"]
    assert response["Sunset"] == settings.API_DEPRECATIONS["ai-core-legacy"]["sunset"]


@override_settings(ROOT_URLCONF="noesis2.tests.test_health_and_versioning")
@pytest.mark.django_db
def test_version_endpoint_modern_version_has_no_deprecation_headers(api_client, tenant):
    response = api_client.get(
        "/test/version/",
        HTTP_X_TENANT_ID="tenant-x",
        HTTP_X_API_VERSION="2024-01-01",
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "version": "2024-01-01"}
    assert "Deprecation" not in response
    assert "Sunset" not in response


@override_settings(ROOT_URLCONF="noesis2.tests.test_health_and_versioning")
@pytest.mark.django_db
def test_mixin_adds_deprecation_headers(api_client, tenant):
    response = api_client.get("/test/mixin/")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response["Deprecation"] == settings.API_DEPRECATIONS["ai-core-legacy"]["deprecation"]
    assert response["Sunset"] == settings.API_DEPRECATIONS["ai-core-legacy"]["sunset"]


def test_health_handler_writes_plain_response():
    handler = healthsrv._Handler.__new__(healthsrv._Handler)
    output = io.BytesIO()
    status: Dict[str, Any] = {}

    def _send_response(code: int) -> None:
        status["code"] = code

    sent_headers: Dict[str, str] = {}

    def _send_header(name: str, value: str) -> None:
        sent_headers[name] = value

    handler.wfile = output
    handler.send_response = _send_response  # type: ignore[assignment]
    handler.send_header = _send_header  # type: ignore[assignment]
    handler.end_headers = lambda: status.setdefault("ended", True)  # type: ignore[assignment]
    handler.request_version = "HTTP/1.1"
    handler.command = "GET"
    handler.path = "/health"
    handler.requestline = "GET /health HTTP/1.1"
    handler.headers = {}

    handler.do_GET()
    handler.log_message("ignored")

    assert status["code"] == 200
    assert sent_headers["Content-Type"] == "text/plain; charset=utf-8"
    assert output.getvalue() == b"ok"


def test_health_main_uses_configured_port(monkeypatch):
    observed: Dict[str, Any] = {}

    class DummyServer:
        def __init__(self, addr, handler):
            observed["addr"] = addr
            observed["handler"] = handler
            self.socket = mock.Mock()
            observed["server"] = self

        def serve_forever(self):
            observed["served"] = True
            raise KeyboardInterrupt

        def server_close(self):
            observed["closed"] = True

    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setattr(healthsrv, "HTTPServer", DummyServer)

    healthsrv.main()

    assert observed["addr"] == ("0.0.0.0", 9090)
    assert observed["handler"] is healthsrv._Handler
    assert observed["served"] is True
    assert observed["closed"] is True
    observed["server"].socket.setsockopt.assert_called_once_with(
        socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
    )


def test_health_main_sets_socket_reuse_and_falls_back(monkeypatch, capsys):
    observed: Dict[str, Any] = {}

    class DummyServer:
        def __init__(self, addr, handler):
            observed["addr"] = addr
            observed["handler"] = handler
            self.socket = mock.Mock()
            observed["server"] = self

        def serve_forever(self):
            observed["served"] = True
            raise KeyboardInterrupt

        def server_close(self):
            observed["closed"] = True

    monkeypatch.setenv("PORT", "invalid")
    monkeypatch.setattr(healthsrv, "HTTPServer", DummyServer)

    healthsrv.main()

    captured = capsys.readouterr()
    assert "Invalid PORT value: invalid" in captured.err
    assert observed["addr"] == ("0.0.0.0", 8080)
    assert observed["served"] is True
    assert observed["closed"] is True
    observed["server"].socket.setsockopt.assert_called_once_with(
        socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
    )


def test_normalise_config_filters_invalid_entries():
    config = versioning._normalise_config({"valid": {"deprecation": "soon"}, "bad": object()})

    assert config == {"valid": {"deprecation": "soon"}}


def test_normalise_config_with_non_mapping_returns_empty():
    assert versioning._normalise_config(None) == {}
    assert versioning._normalise_config([1, 2, 3]) == {}


@override_settings(API_DEPRECATIONS={"demo": {"sunset": "2025"}})
def test_build_deprecation_headers_uses_defaults():
    headers = build_deprecation_headers(deprecated=True, config_key="demo")

    assert headers["Deprecation"] == "true"
    assert headers["Sunset"] == "2025"


def test_build_deprecation_headers_allows_overrides():
    headers = build_deprecation_headers(
        deprecated=True,
        override_deprecation="Mon, 01 Jan 2024 00:00:00 GMT",
        override_sunset="Tue, 01 Jul 2024 00:00:00 GMT",
    )

    assert headers == {
        "Deprecation": "Mon, 01 Jan 2024 00:00:00 GMT",
        "Sunset": "Tue, 01 Jul 2024 00:00:00 GMT",
    }


def test_build_deprecation_headers_skips_when_not_deprecated():
    assert build_deprecation_headers(deprecated=False) == {}


@override_settings(API_DEPRECATIONS={"demo": {"deprecation": "soon"}})
def test_resolve_missing_sunset_is_ignored():
    request = HttpRequest()
    headers = mark_deprecated_response(request, config_key="demo")

    assert headers == {"Deprecation": "soon"}
    assert request._api_deprecation_headers == headers


def test_store_headers_ignores_falsey_values():
    request = HttpRequest()
    request._api_deprecation_headers = {"Existing": "keep"}
    versioning._store_headers(request, {"New": "value", "Empty": ""})

    assert request._api_deprecation_headers == {"Existing": "keep", "New": "value"}


def test_store_headers_initialises_dict_for_invalid_existing():
    request = HttpRequest()
    request._api_deprecation_headers = None  # type: ignore[attr-defined]
    versioning._store_headers(request, {"New": "value"})

    assert request._api_deprecation_headers == {"New": "value"}


class _DummyBase:
    def finalize_response(self, request, response, *args, **kwargs):
        response["X-Base"] = "called"
        return response


class _DummyView(DeprecationHeadersMixin, _DummyBase):
    api_deprecated = True
    api_deprecation_id = "ai-core-legacy"


def test_mixin_finalize_response_adds_headers():
    request = HttpRequest()
    response = HttpResponse()

    view = _DummyView()
    result = view.finalize_response(request, response)

    assert result["X-Base"] == "called"
    assert result["Deprecation"] == settings.API_DEPRECATIONS["ai-core-legacy"]["deprecation"]
    assert result["Sunset"] == settings.API_DEPRECATIONS["ai-core-legacy"]["sunset"]
    assert request._api_deprecation_headers["Deprecation"] == settings.API_DEPRECATIONS["ai-core-legacy"]["deprecation"]


def test_build_deprecation_headers_defaults_without_config():
    headers = build_deprecation_headers(deprecated=True, config_key="missing")

    assert headers == {"Deprecation": "true"}


def test_mark_deprecated_response_accepts_manual_values():
    request = HttpRequest()
    headers = mark_deprecated_response(
        request,
        config_key=None,
        deprecation="Fri, 01 Mar 2024 00:00:00 GMT",
        sunset="Sat, 01 Jun 2024 00:00:00 GMT",
    )

    assert headers == {
        "Deprecation": "Fri, 01 Mar 2024 00:00:00 GMT",
        "Sunset": "Sat, 01 Jun 2024 00:00:00 GMT",
    }
    assert request._api_deprecation_headers == headers
