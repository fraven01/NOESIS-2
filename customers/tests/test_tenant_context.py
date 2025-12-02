import pytest
from django.conf import settings
from django.db import connection
from django.test import RequestFactory

from customers.tenant_context import TenantContext, TenantRequiredError
from customers.tests.factories import TenantFactory


@pytest.mark.django_db
def test_resolve_identifier_prefers_schema_name():
    tenant = TenantFactory(schema_name="alpha")

    resolved = TenantContext.resolve_identifier("alpha")

    assert resolved == tenant


@pytest.mark.django_db
def test_resolve_identifier_allows_pk_when_opted_in():
    tenant = TenantFactory()

    assert TenantContext.resolve_identifier(str(tenant.pk)) is None
    assert TenantContext.resolve_identifier(str(tenant.pk), allow_pk=True) == tenant


@pytest.mark.django_db
def test_from_request_prefers_request_tenant_and_caches(monkeypatch):
    tenant = TenantFactory(schema_name="cached")
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


@pytest.mark.django_db
def test_from_request_resolves_connection_schema(monkeypatch):
    tenant = TenantFactory(schema_name="schema-source")
    request = RequestFactory().get("/")

    monkeypatch.setattr(connection, "schema_name", tenant.schema_name, raising=False)

    resolved = TenantContext.from_request(request)

    assert resolved == tenant


@pytest.mark.django_db
def test_from_request_resolves_connection_schema_pk(monkeypatch):
    tenant = TenantFactory()
    request = RequestFactory().get("/")

    monkeypatch.setattr(connection, "schema_name", str(tenant.pk), raising=False)

    with pytest.raises(TenantRequiredError):
        TenantContext.from_request(request)

    resolved_optional = TenantContext.from_request(request, require=False)

    assert resolved_optional is None


@pytest.mark.django_db
def test_from_request_resolves_connection_schema_pk_when_allowed(monkeypatch):
    tenant = TenantFactory()
    request = RequestFactory().get("/")

    monkeypatch.setattr(connection, "schema_name", str(tenant.pk), raising=False)

    resolved = TenantContext.from_request(request, allow_pk=True)

    assert resolved == tenant


@pytest.mark.django_db
def test_from_request_resolves_connection_schema_pk_in_cli_context(monkeypatch):
    tenant = TenantFactory()
    from types import SimpleNamespace

    request = SimpleNamespace()

    monkeypatch.setattr(connection, "schema_name", str(tenant.pk), raising=False)

    resolved = TenantContext.from_request(request, allow_headers=False, allow_pk=True)

    assert resolved == tenant


@pytest.mark.django_db
def test_from_request_resolves_headers_when_allowed(monkeypatch):
    tenant = TenantFactory(schema_name="header-tenant")
    request = RequestFactory().get("/", HTTP_X_TENANT_ID=tenant.schema_name)

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)

    resolved = TenantContext.from_request(request)

    assert resolved == tenant


@pytest.mark.django_db
def test_from_request_resolves_header_pk(monkeypatch):
    tenant = TenantFactory()
    request = RequestFactory().get("/", HTTP_X_TENANT_ID=str(tenant.pk))

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)

    resolved = TenantContext.from_request(request)

    assert resolved == tenant


@pytest.mark.django_db
def test_from_request_raises_when_required_and_missing(monkeypatch):
    request = RequestFactory().get("/")

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    monkeypatch.setattr(connection, "schema_name", public_schema, raising=False)

    with pytest.raises(TenantRequiredError):
        TenantContext.from_request(request)
