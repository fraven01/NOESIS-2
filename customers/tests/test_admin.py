import pytest
from unittest.mock import call

from customers.admin import migrate_selected_tenants
from customers.models import Tenant
from .factories import TenantFactory


@pytest.mark.django_db
def test_migrate_selected_tenants_passes_schema(mocker):
    t1 = TenantFactory(schema_name="alpha")
    t2 = TenantFactory(schema_name="beta")
    call_mock = mocker.patch("customers.admin.call_command")

    queryset = Tenant.objects.filter(id__in=[t1.id, t2.id])
    migrate_selected_tenants(None, None, queryset)

    call_mock.assert_has_calls(
        [
            call("migrate_schemas", schema="alpha"),
            call("migrate_schemas", schema="beta"),
        ]
    )
