import pytest
from django.conf import settings
from django.db import connection
from django.test import RequestFactory

from customers.tenant_context import TenantContext, TenantRequiredError


pytestmark = [
    pytest.mark.slow,
    pytest.mark.django_db,
    pytest.mark.xdist_group("tenant_ops"),
]


def test_resolve_identifier_prefers_schema_name(tenant_pool):
    tenant = tenant_pool["alpha"]

    resolved = TenantContext.resolve_identifier(tenant.schema_name)

    assert resolved == tenant


def test_resolve_identifier_allows_pk_when_opted_in(tenant_pool):
    tenant = tenant_pool["beta"]

    assert TenantContext.resolve_identifier(str(tenant.pk)) is None
    assert TenantContext.resolve_identifier(str(tenant.pk), allow_pk=True) == tenant


def test_from_request_prefers_request_tenant_and_caches(monkeypatch, tenant_pool):
    tenant = tenant_pool["gamma"]
    request = RequestFactory().get("/")
    request.tenant = tenant

    first_resolution = TenantContext.from_request(request)
    assert first_resolution == tenant

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)
    request.tenant = None

    cached_resolution = TenantContext.from_request(
        request, allow_headers=False, require=False
    )

    assert cached_resolution == tenant
    assert request._tenant_context_cache is tenant


def test_from_request_resolves_connection_schema(monkeypatch, tenant_pool):
    tenant = tenant_pool["delta"]
    request = RequestFactory().get("/")

    monkeypatch.setattr(connection, "schema_name", tenant.schema_name, raising=False)

    resolved = TenantContext.from_request(request)

    assert resolved == tenant


def test_from_request_resolves_connection_schema_pk(monkeypatch, tenant_pool):
    tenant = tenant_pool["alpha"]
    request = RequestFactory().get("/")

    monkeypatch.setattr(connection, "schema_name", str(tenant.pk), raising=False)

    with pytest.raises(TenantRequiredError):
        TenantContext.from_request(request)

    resolved_optional = TenantContext.from_request(request, require=False)

    assert resolved_optional is None


def test_from_request_resolves_connection_schema_pk_when_allowed(
    monkeypatch, tenant_pool
):
    tenant = tenant_pool["beta"]
    request = RequestFactory().get("/")

    monkeypatch.setattr(connection, "schema_name", str(tenant.pk), raising=False)

    resolved = TenantContext.from_request(request, allow_pk=True)

    assert resolved == tenant


def test_from_request_resolves_connection_schema_pk_in_cli_context(
    monkeypatch, tenant_pool
):
    tenant = tenant_pool["gamma"]
    from types import SimpleNamespace

    request = SimpleNamespace()

    monkeypatch.setattr(connection, "schema_name", str(tenant.pk), raising=False)

    resolved = TenantContext.from_request(request, allow_headers=False, allow_pk=True)

    assert resolved == tenant


def test_from_request_resolves_headers_when_allowed(monkeypatch, tenant_pool):
    tenant = tenant_pool["delta"]
    request = RequestFactory().get("/", HTTP_X_TENANT_ID=tenant.schema_name)

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)

    resolved = TenantContext.from_request(request)

    assert resolved == tenant


def test_from_request_resolves_header_pk(monkeypatch, tenant_pool):
    tenant = tenant_pool["alpha"]
    request = RequestFactory().get("/", HTTP_X_TENANT_ID=str(tenant.pk))

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)

    resolved = TenantContext.from_request(request)

    assert resolved == tenant


def test_from_request_raises_when_required_and_missing(monkeypatch):
    request = RequestFactory().get("/")

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)

    with pytest.raises(TenantRequiredError):
        TenantContext.from_request(request)
