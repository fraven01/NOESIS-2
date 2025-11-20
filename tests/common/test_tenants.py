import pytest

from django.conf import settings
from django.test import RequestFactory
from django_tenants.utils import schema_context

from common.tenants import _is_valid_tenant_request

pytestmark = pytest.mark.django_db


def _build_request(schema_name: str):
    request = RequestFactory().get("/", HTTP_X_TENANT_SCHEMA=schema_name)
    request.tenant_schema = schema_name
    return request


def test_tenant_request_requires_active_schema(test_tenant_schema_name):
    request = _build_request(test_tenant_schema_name)

    tenant, error = _is_valid_tenant_request(request)

    assert error is None
    assert tenant is not None
    assert tenant.schema_name == test_tenant_schema_name


def test_tenant_request_rejected_when_connection_is_public(test_tenant_schema_name):
    with schema_context(settings.PUBLIC_SCHEMA_NAME):
        request = _build_request(test_tenant_schema_name)

        tenant, error = _is_valid_tenant_request(request)

    assert tenant is None
    assert error == "Tenant context is not active for this request"
