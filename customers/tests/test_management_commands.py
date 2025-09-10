import pytest
from django.core.management import CommandError, call_command

from customers.models import Domain, Tenant
from .factories import DomainFactory, TenantFactory


@pytest.mark.django_db
def test_create_tenant_command():
    call_command(
        "create_tenant", schema="testschema", name="Test", domain="test.example.com"
    )
    tenant = Tenant.objects.get(schema_name="testschema")
    assert tenant.name == "Test"
    assert Domain.objects.filter(tenant=tenant, domain="test.example.com").exists()


@pytest.mark.django_db
def test_create_tenant_disallows_public_schema():
    with pytest.raises(CommandError):
        call_command(
            "create_tenant",
            schema="public",
            name="Public",
            domain="public.example.com",
        )


@pytest.mark.django_db
def test_create_tenant_domain_collision():
    DomainFactory(domain="collision.example.com")
    with pytest.raises(CommandError):
        call_command(
            "create_tenant",
            schema="collision",
            name="Collision",
            domain="collision.example.com",
        )


@pytest.mark.django_db
def test_list_tenants_command(capsys):
    tenant = TenantFactory(schema_name="alpha")
    call_command("list_tenants")
    captured = capsys.readouterr()
    assert "alpha" in captured.out
    assert tenant.name in captured.out
