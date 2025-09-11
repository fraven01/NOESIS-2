import pytest
from django_tenants.utils import schema_context

from customers.tests.factories import DomainFactory, TenantFactory
from customers.models import Domain
from common.tenants import get_current_tenant


@pytest.mark.django_db
def test_demo_view_requires_tenant_header(client):
    tenant = TenantFactory(schema_name="alpha")
    tenant.create_schema(check_if_exists=True)
    Domain.objects.update_or_create(
        domain="testserver", defaults={"tenant": tenant, "is_primary": True}
    )
    with schema_context(tenant.schema_name):
        response = client.get("/tenant-demo/")
    assert response.status_code == 403


@pytest.mark.django_db
def test_demo_view_with_valid_header(client):
    tenant = TenantFactory(schema_name="beta")
    tenant.create_schema(check_if_exists=True)
    Domain.objects.update_or_create(
        domain="testserver", defaults={"tenant": tenant, "is_primary": True}
    )
    with schema_context(tenant.schema_name):
        response = client.get("/tenant-demo/", HTTP_X_TENANT_SCHEMA=tenant.schema_name)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.django_db
def test_get_current_tenant_returns_active_tenant():
    tenant = TenantFactory(schema_name="gamma")
    with schema_context(tenant.schema_name):
        assert get_current_tenant() == tenant
