import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError

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
def test_create_tenant_command_public_schema():
    with pytest.raises(CommandError):
        call_command(
            "create_tenant",
            schema=settings.PUBLIC_SCHEMA_NAME,
            name="Test",
            domain="test.example.com",
        )


@pytest.mark.django_db
def test_create_tenant_command_duplicate_schema():
    TenantFactory(schema_name="dup")
    with pytest.raises(CommandError):
        call_command(
            "create_tenant", schema="dup", name="Test", domain="test2.example.com"
        )


@pytest.mark.django_db
def test_create_tenant_command_duplicate_domain():
    DomainFactory(domain="dup.example.com")
    with pytest.raises(CommandError):
        call_command(
            "create_tenant", schema="test2", name="Test", domain="dup.example.com"
        )


@pytest.mark.django_db
def test_create_tenant_command_is_atomic(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(Domain.objects, "create", _raise)
    with pytest.raises(RuntimeError):
        call_command(
            "create_tenant", schema="atomic", name="Atomic", domain="atomic.example"
        )
    assert not Tenant.objects.filter(schema_name="atomic").exists()


@pytest.mark.django_db
def test_list_tenants_command(capsys):
    tenant = TenantFactory(schema_name="alpha")
    call_command("list_tenants")
    captured = capsys.readouterr()
    assert "alpha" in captured.out
    assert tenant.name in captured.out
