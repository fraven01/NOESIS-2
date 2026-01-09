import pytest
from django_tenants.utils import schema_context

from common.tenants import TenantSchemaRequiredMixin, get_current_tenant
from common.views import DemoView
from rest_framework.views import APIView
from testsupport.tenant_fixtures import ensure_tenant_domain


pytestmark = [
    pytest.mark.slow,
    pytest.mark.django_db,
    pytest.mark.xdist_group("tenant_ops"),
]


def test_demo_view_requires_tenant_header(client, tenant_pool):
    tenant = tenant_pool["alpha"]
    ensure_tenant_domain(tenant, domain="testserver")
    with schema_context(tenant.schema_name):
        response = client.get("/tenant-demo/")
    assert response.status_code == 403
    assert response.content.decode() == "Tenant schema header missing"


def test_demo_view_mro_places_tenant_mixin_before_apiview():
    mro = DemoView.mro()
    assert mro.index(TenantSchemaRequiredMixin) < mro.index(APIView)


def test_demo_view_with_valid_header(client, tenant_pool):
    tenant = tenant_pool["beta"]
    ensure_tenant_domain(tenant, domain="testserver")
    with schema_context(tenant.schema_name):
        response = client.get("/tenant-demo/", HTTP_X_TENANT_SCHEMA=tenant.schema_name)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_view_with_mismatched_header(client, tenant_pool):
    tenant = tenant_pool["gamma"]
    ensure_tenant_domain(tenant, domain="testserver")
    with schema_context(tenant.schema_name):
        response = client.get("/tenant-demo/", HTTP_X_TENANT_SCHEMA="unexpected")
    assert response.status_code == 403
    assert response.content.decode() == "Tenant schema does not match resolved tenant"


def test_get_current_tenant_returns_active_tenant(tenant_pool):
    tenant = tenant_pool["delta"]
    with schema_context(tenant.schema_name):
        assert get_current_tenant() == tenant
