import pytest
from django.conf import settings
from django.core.management import call_command

from customers.models import Domain, Tenant


pytestmark = pytest.mark.django_db


def test_bootstrap_public_tenant_idempotent():
    call_command("bootstrap_public_tenant", domain="public.localhost")
    call_command("bootstrap_public_tenant", domain="public.localhost")
    assert Tenant.objects.filter(schema_name=settings.PUBLIC_SCHEMA_NAME).count() == 1
    assert Domain.objects.filter(domain="public.localhost").count() == 1
