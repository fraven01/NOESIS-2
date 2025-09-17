import pytest
from django.core.management import call_command, CommandError
from django.db import connection, OperationalError
from django_tenants.utils import get_public_schema_name, schema_context

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
def test_create_tenant_creates_schema_when_auto_creation_disabled(monkeypatch):
    monkeypatch.setattr("customers.models.Tenant.auto_create_schema", False)

    call_command(
        "create_tenant", schema="noschemaauto", name="Schema", domain="schema.example"
    )
    tenant = Tenant.objects.get(schema_name="noschemaauto")

    with schema_context(tenant.schema_name):
        tables = connection.introspection.table_names()

    assert "customers_domain" in tables


@pytest.mark.django_db
def test_create_tenant_disallows_public_schema():
    with pytest.raises(CommandError):
        call_command(
            "create_tenant",
            schema=get_public_schema_name(),
            name="Public",
            domain="public.example.com",
        )


@pytest.mark.django_db
def test_create_tenant_duplicate_schema():
    TenantFactory(schema_name="dup")
    with pytest.raises(CommandError):
        call_command(
            "create_tenant", schema="dup", name="Test", domain="test2.example.com"
        )


@pytest.mark.django_db
def test_create_tenant_duplicate_domain():
    DomainFactory(domain="dup.example.com")
    with pytest.raises(CommandError):
        call_command(
            "create_tenant", schema="test2", name="Test", domain="dup.example.com"
        )


@pytest.mark.django_db
def test_create_tenant_is_atomic(monkeypatch):
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


@pytest.mark.django_db
def test_create_tenant_superuser_missing_tables(monkeypatch):
    tenant = TenantFactory(schema_name="missing")

    recorded_calls = []

    def fake_call_command(name, *args, **kwargs):
        recorded_calls.append((name, args, kwargs))

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.call_command",
        fake_call_command,
    )

    class DummyQuerySet:
        def exists(self):
            return False

    class DummyManager:
        def filter(self, *args, **kwargs):
            return DummyQuerySet()

        def create_superuser(self, *args, **kwargs):
            raise OperationalError("relation does not exist")

    class DummyUser:
        objects = DummyManager()

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.get_user_model",
        lambda: DummyUser,
    )

    with pytest.raises(CommandError) as excinfo:
        call_command(
            "create_tenant_superuser",
            schema=tenant.schema_name,
            username="admin",
            password="secret",
        )

    assert "migrate" in str(excinfo.value).lower()
    assert recorded_calls == [("migrate_schemas", (), {"schema": tenant.schema_name})]
