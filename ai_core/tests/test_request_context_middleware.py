from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest
from django.conf import settings
from django.db import connection
from django.http import HttpResponse
from django.test import RequestFactory
from structlog.contextvars import clear_contextvars, get_contextvars

from ai_core.contracts.scope import ScopeContext
from ai_core.middleware import RequestContextMiddleware
from common.constants import IDEMPOTENCY_KEY_HEADER
from customers.models import Tenant
from customers.tenant_context import TenantRequiredError


@pytest.fixture(autouse=True)
def _clear_structlog_context():
    clear_contextvars()
    yield
    clear_contextvars()


def _build_request(
    factory: RequestFactory,
    method: str,
    path: str,
    **headers: Any,
):
    request_method = getattr(factory, method.lower())
    request = request_method(path, **headers)
    return request


def test_middleware_binds_headers_and_sets_response_metadata():
    factory = RequestFactory()
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    request = _build_request(
        factory,
        "get",
        "/ai/ping/",
        HTTP_TRACEPARENT=traceparent,
        HTTP_X_TRACE_ID="trace-ignore",
        HTTP_X_CASE_ID="case-456",
        HTTP_X_TENANT_ID="tenant-spoof",
        HTTP_X_KEY_ALIAS="alias-001",
        HTTP_X_FORWARDED_FOR="203.0.113.1, 10.0.0.1",
        HTTP_IDEMPOTENCY_KEY="idem-789",
    )
    request.tenant = Tenant(schema_name="tenant-789", name="Tenant 789")
    request.resolver_match = SimpleNamespace(route="api:ping")

    captured: dict[str, str] = {}
    scope: ScopeContext | None = None

    def _view(inner_request):
        nonlocal captured
        nonlocal scope
        captured = get_contextvars()
        scope = getattr(inner_request, "scope_context", None)
        return HttpResponse("ok")

    middleware = RequestContextMiddleware(_view)
    response = middleware(request)

    assert response.status_code == 200
    assert response["X-Trace-Id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert response["X-Tenant-Id"] == "tenant-789"
    assert response["X-Case-Id"] == "case-456"
    assert response["X-Key-Alias"] == "alias-001"
    assert response["traceparent"] == traceparent
    assert response[IDEMPOTENCY_KEY_HEADER] == "idem-789"

    assert captured["trace.id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert captured["span.id"] == "00f067aa0ba902b7"
    assert captured["tenant.id"] == "tenant-789"
    assert captured["case.id"] == "case-456"
    assert captured["key.alias"] == "alias-001"
    assert captured["idempotency.key"] == "idem-789"
    assert captured["http.method"] == "GET"
    assert captured["http.route"] == "api:ping"
    assert captured["client.ip"] == "203.0.113.1"

    assert isinstance(scope, ScopeContext)
    assert scope.trace_id == captured["trace.id"]
    assert scope.run_id == scope.trace_id
    assert scope.ingestion_run_id is None
    assert scope.invocation_id
    assert scope.case_id == "case-456"
    assert scope.idempotency_key == "idem-789"

    assert get_contextvars() == {}


def test_middleware_generates_trace_ids_when_headers_missing():
    factory = RequestFactory()
    request = _build_request(factory, "post", "/ai/intake/")
    request.tenant = Tenant(schema_name="trace-tenant", name="Trace Tenant")

    captured: dict[str, str] = {}
    scope: ScopeContext | None = None

    def _view(inner_request):
        nonlocal captured
        nonlocal scope
        captured = get_contextvars()
        scope = getattr(inner_request, "scope_context", None)
        return HttpResponse(status=201)

    middleware = RequestContextMiddleware(_view)
    response = middleware(request)

    assert response.status_code == 201
    trace_id = response["X-Trace-Id"]
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)
    assert response["X-Tenant-Id"] == "trace-tenant"
    assert "traceparent" not in response
    assert "X-Case-Id" not in response
    assert "X-Key-Alias" not in response

    assert captured["trace.id"] == trace_id
    assert re.fullmatch(r"[0-9a-f]{16}", captured["span.id"])
    assert captured["tenant.id"] == "trace-tenant"
    assert captured["http.method"] == "POST"
    assert captured["http.route"] == "/ai/intake/"
    assert captured["client.ip"] == "127.0.0.1"
    assert "case.id" not in captured
    assert "key.alias" not in captured

    assert isinstance(scope, ScopeContext)
    assert scope.trace_id == trace_id
    assert scope.run_id == trace_id
    assert scope.invocation_id
    assert scope.ingestion_run_id is None
    assert scope.case_id is None
    assert scope.idempotency_key is None

    assert get_contextvars() == {}


def test_middleware_rejects_missing_tenant(monkeypatch):
    monkeypatch.setattr(
        "ai_core.middleware.context.TenantContext.from_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TenantRequiredError("Tenant could not be resolved from request")
        ),
    )
    factory = RequestFactory()
    request = _build_request(
        factory,
        "get",
        "/ai/ping/",
        HTTP_X_TENANT_ID="spoof-only",
    )

    called = False

    def _view(inner_request):
        nonlocal called
        called = True
        return HttpResponse("ok")

    middleware = RequestContextMiddleware(_view)
    response = middleware(request)

    assert response.status_code == 403
    assert json.loads(response.content.decode()) == {
        "detail": "Tenant could not be resolved from request"
    }
    assert called is False
    assert get_contextvars() == {}


def test_middleware_rejects_header_only_when_headers_disallowed(monkeypatch):
    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)

    factory = RequestFactory()
    request = _build_request(
        factory,
        "get",
        "/ai/ping/",
        HTTP_X_TENANT_ID="header-only",
    )

    called = False

    def _view(inner_request):
        nonlocal called
        called = True
        return HttpResponse("ok")

    middleware = RequestContextMiddleware(_view)
    response = middleware(request)

    assert response.status_code == 403
    assert json.loads(response.content.decode()) == {
        "detail": "Tenant could not be resolved from request",
    }
    assert called is False
    assert get_contextvars() == {}


def test_middleware_uses_resolved_tenant_over_header():
    factory = RequestFactory()
    request = _build_request(
        factory,
        "get",
        "/ai/ping/",
        HTTP_X_TENANT_ID="spoofed-tenant",
    )
    request.tenant = Tenant(schema_name="canonical-tenant", name="Canonical")

    captured: dict[str, str] = {}

    def _view(inner_request):
        nonlocal captured
        captured = get_contextvars()
        return HttpResponse("ok")

    middleware = RequestContextMiddleware(_view)
    response = middleware(request)

    assert response["X-Tenant-Id"] == "canonical-tenant"
    assert captured["tenant.id"] == "canonical-tenant"


def test_middleware_builds_scope_context_from_ingestion_headers():
    factory = RequestFactory()
    request = _build_request(
        factory,
        "post",
        "/ai/intake/",
        HTTP_X_TENANT_ID="tenant-scope",
        HTTP_X_CASE_ID="case-scope",
        HTTP_X_TRACE_ID="trace-scope",
        HTTP_X_INGESTION_RUN_ID="ingest-42",
        HTTP_X_INVOCATION_ID="invoke-99",
    )
    request.tenant = Tenant(schema_name="tenant-scope", name="Tenant Scope")

    captured_scope: ScopeContext | None = None

    def _view(inner_request):
        nonlocal captured_scope
        captured_scope = getattr(inner_request, "scope_context", None)
        return HttpResponse("ok")

    middleware = RequestContextMiddleware(_view)
    response = middleware(request)

    assert response.status_code == 200
    assert isinstance(captured_scope, ScopeContext)
    assert captured_scope.ingestion_run_id == "ingest-42"
    assert captured_scope.run_id is None
    assert captured_scope.invocation_id == "invoke-99"
    assert captured_scope.trace_id == "trace-scope"
    assert captured_scope.case_id == "case-scope"
    assert captured_scope.tenant_id == "tenant-scope"


def test_middleware_rejects_conflicting_run_scope():
    factory = RequestFactory()
    request = _build_request(
        factory,
        "get",
        "/ai/ping/",
        HTTP_X_TENANT_ID="tenant-conflict",
        HTTP_X_TRACE_ID="trace-conflict",
        HTTP_X_RUN_ID="graph-1",
        HTTP_X_INGESTION_RUN_ID="ingest-1",
    )
    request.tenant = Tenant(schema_name="tenant-conflict", name="Tenant Conflict")

    called = False

    def _view(inner_request):
        nonlocal called
        called = True
        return HttpResponse("ok")

    middleware = RequestContextMiddleware(_view)
    response = middleware(request)

    assert response.status_code == 400
    assert json.loads(response.content.decode()) == {
        "detail": "Exactly one of run_id or ingestion_run_id must be provided",
    }
    assert called is False


def test_middleware_allows_value_error_from_view_to_propagate():
    factory = RequestFactory()
    request = _build_request(
        factory,
        "get",
        "/ai/ping/",
        HTTP_X_TENANT_ID="tenant-error",
    )
    request.tenant = Tenant(schema_name="tenant-error", name="Tenant Error")

    def _view(_inner_request):
        raise ValueError("view raised")

    middleware = RequestContextMiddleware(_view)

    with pytest.raises(ValueError, match="view raised"):
        middleware(request)

    assert get_contextvars() == {}
