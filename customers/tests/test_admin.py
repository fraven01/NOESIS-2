import pytest
from unittest.mock import call

from customers.admin import migrate_selected_tenants
from customers.models import Tenant


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_migrate_selected_tenants_passes_schema(mocker, tenant_pool):
    t1 = tenant_pool["alpha"]
    t2 = tenant_pool["beta"]
    call_mock = mocker.patch("customers.admin.call_command")

    queryset = Tenant.objects.filter(id__in=[t1.id, t2.id])
    migrate_selected_tenants(None, None, queryset)

    call_mock.assert_has_calls(
        [
            call("migrate_schemas", schema=t1.schema_name),
            call("migrate_schemas", schema=t2.schema_name),
        ]
    )
