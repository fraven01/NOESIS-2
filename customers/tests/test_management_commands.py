from io import StringIO

import pytest
from django.core.management import call_command, CommandError
from django.db import connection, OperationalError
from django_tenants.utils import get_public_schema_name, schema_context

from customers.models import Domain, Tenant
from users.models import User
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


@pytest.mark.django_db
def test_create_tenant_superuser_creates_user(monkeypatch):
    tenant = TenantFactory(schema_name="createuser")
    tenant.create_schema(check_if_exists=True)

    recorded_calls = []

    def fake_call_command(name, *args, **kwargs):
        recorded_calls.append((name, args, kwargs))
        return None

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.call_command",
        fake_call_command,
    )

    stdout = StringIO()

    call_command(
        "create_tenant_superuser",
        schema=tenant.schema_name,
        username="demo-admin",
        email="demo@example.com",
        password="super-secret",
        stdout=stdout,
    )

    assert recorded_calls == [("migrate_schemas", (), {"schema": tenant.schema_name})]
    assert "created" in stdout.getvalue()

    with schema_context(tenant.schema_name):
        user = User.objects.get(username="demo-admin")
        assert user.email == "demo@example.com"
        assert user.is_staff is True
        assert user.is_superuser is True
        assert user.check_password("super-secret")


@pytest.mark.django_db
def test_create_tenant_superuser_updates_existing_user(monkeypatch):
    tenant = TenantFactory(schema_name="updateuser")
    tenant.create_schema(check_if_exists=True)

    with schema_context(tenant.schema_name):
        user = User.objects.create_user(
            username="demo-admin", email="old@example.com", password="oldpass"
        )
        user.is_staff = False
        user.is_superuser = False
        user.save(update_fields=["is_staff", "is_superuser"])

    recorded_calls = []

    def fake_call_command(name, *args, **kwargs):
        recorded_calls.append((name, args, kwargs))
        return None

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.call_command",
        fake_call_command,
    )

    stdout = StringIO()
    call_command(
        "create_tenant_superuser",
        schema=tenant.schema_name,
        username="demo-admin",
        password="newpass",
        stdout=stdout,
    )

    assert recorded_calls == [("migrate_schemas", (), {"schema": tenant.schema_name})]
    assert "ensured" in stdout.getvalue()

    with schema_context(tenant.schema_name):
        user = User.objects.get(username="demo-admin")
        assert user.is_staff is True
        assert user.is_superuser is True
        assert user.check_password("newpass")


@pytest.mark.django_db
def test_create_tenant_superuser_requires_password(monkeypatch):
    tenant = TenantFactory(schema_name="requirespass")
    tenant.create_schema(check_if_exists=True)

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.call_command",
        lambda *args, **kwargs: None,
    )
    monkeypatch.delenv("DJANGO_SUPERUSER_PASSWORD", raising=False)

    with pytest.raises(CommandError) as excinfo:
        call_command(
            "create_tenant_superuser",
            schema=tenant.schema_name,
            username="demo-admin",
        )

    assert "Missing password" in str(excinfo.value)

    with schema_context(tenant.schema_name):
        assert not User.objects.filter(username="demo-admin").exists()


@pytest.mark.django_db
def test_create_tenant_superuser_requires_username(monkeypatch):
    tenant = TenantFactory(schema_name="requiresuser")
    tenant.create_schema(check_if_exists=True)

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.call_command",
        lambda *args, **kwargs: None,
    )
    monkeypatch.delenv("DJANGO_SUPERUSER_USERNAME", raising=False)

    with pytest.raises(CommandError) as excinfo:
        call_command(
            "create_tenant_superuser",
            schema=tenant.schema_name,
            password="secret",
        )

    assert "Missing username" in str(excinfo.value)


@pytest.mark.django_db
def test_create_tenant_superuser_missing_tenant():
    with pytest.raises(CommandError) as excinfo:
        call_command(
            "create_tenant_superuser",
            schema="does-not-exist",
            username="demo-admin",
            password="secret",
        )

    assert "does not exist" in str(excinfo.value)


@pytest.mark.django_db
def test_add_domain_creates_and_sets_primary():
    tenant = TenantFactory(schema_name="tenant-add-domain")

    stdout = StringIO()
    call_command(
        "add_domain",
        "--schema",
        tenant.schema_name,
        "--domain",
        "HTTPS://Example.Com/demo",
        "--primary",
        stdout=stdout,
    )

    domain = Domain.objects.get(domain="example.com")
    assert domain.tenant_id == tenant.id
    assert domain.is_primary is True
    assert "set as primary" in stdout.getvalue()


@pytest.mark.django_db
def test_add_domain_requires_force_for_existing_assignment():
    original = TenantFactory(schema_name="original")
    DomainFactory(tenant=original, domain="shared.example.com")
    target = TenantFactory(schema_name="target")

    with pytest.raises(CommandError) as excinfo:
        call_command(
            "add_domain",
            "--schema",
            target.schema_name,
            "--domain",
            "shared.example.com",
        )

    assert "already assigned" in str(excinfo.value)


@pytest.mark.django_db
def test_add_domain_force_reassigns_and_sets_primary():
    original = TenantFactory(schema_name="orig-force")
    domain = DomainFactory(tenant=original, domain="force.example.com", is_primary=True)
    target = TenantFactory(schema_name="target-force")

    call_command(
        "add_domain",
        "--schema",
        target.schema_name,
        "--domain",
        "force.example.com",
        "--force-reassign",
        "--primary",
    )

    domain.refresh_from_db()
    assert domain.tenant_id == target.id
    assert domain.is_primary is True


@pytest.mark.django_db
def test_add_domain_switches_primary_within_tenant():
    tenant = TenantFactory(schema_name="swap-primary")
    old = DomainFactory(tenant=tenant, domain="old.example.com", is_primary=True)
    new = DomainFactory(tenant=tenant, domain="new.example.com", is_primary=False)

    call_command(
        "add_domain",
        "--schema",
        tenant.schema_name,
        "--domain",
        "new.example.com",
        "--primary",
    )

    old.refresh_from_db()
    new.refresh_from_db()
    assert old.is_primary is False
    assert new.is_primary is True


@pytest.mark.django_db
def test_add_domain_ensures_existing_domain_and_normalizes_input():
    tenant = TenantFactory(schema_name="tenant-ensure")
    domain = DomainFactory(tenant=tenant, domain="normalize.example.com")
    with schema_context(get_public_schema_name()):
        Domain.objects.filter(pk=domain.pk).update(is_primary=False)
    domain.refresh_from_db()
    assert domain.is_primary is False

    stdout = StringIO()
    call_command(
        "add_domain",
        "--schema",
        tenant.schema_name,
        "--domain",
        "  http://Normalize.Example.com:8000/some/path  ",
        stdout=stdout,
    )

    domain.refresh_from_db()
    assert domain.tenant_id == tenant.id
    assert domain.is_primary is False
    assert Domain.objects.filter(domain="normalize.example.com").count() == 1
    assert "ensured" in stdout.getvalue()


@pytest.mark.django_db
def test_add_domain_normalises_credentials_and_trailing_dot():
    tenant = TenantFactory(schema_name="tenant-normalize-creds")

    call_command(
        "add_domain",
        "--schema",
        tenant.schema_name,
        "--domain",
        "https://user:pass@Example.Com.:8443/path",
    )

    domain = Domain.objects.get(domain="example.com")
    assert domain.tenant_id == tenant.id


@pytest.mark.django_db
def test_add_domain_missing_tenant():
    with pytest.raises(CommandError) as excinfo:
        call_command(
            "add_domain",
            "--schema",
            "missing",
            "--domain",
            "missing.example.com",
        )

    assert "does not exist" in str(excinfo.value)


@pytest.mark.django_db
def test_collectstatic_command_resolves():
    call_command("collectstatic", "--dry-run", "--noinput", verbosity=0)


@pytest.mark.django_db
def test_create_tenant_superuser_uses_environment_defaults(monkeypatch):
    tenant = TenantFactory(schema_name="env-superuser")
    tenant.create_schema(check_if_exists=True)

    recorded_calls = []

    def fake_call_command(name, *args, **kwargs):
        recorded_calls.append((name, args, kwargs))
        return None

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.call_command",
        fake_call_command,
    )

    created_params = {}
    filter_calls = []

    class DummyQuerySet:
        def first(self):
            return None

    class DummyManager:
        def filter(self, *args, **kwargs):
            filter_calls.append(kwargs)
            return DummyQuerySet()

        def create_superuser(self, username, email, password):
            created_params["call"] = (username, email, password)
            return DummyUser(username=username, email=email)

    class DummyUser:
        objects = DummyManager()

        def __init__(self, username, email):
            self.username = username
            self.email = email
            self.is_staff = True
            self.is_superuser = True

    monkeypatch.setenv("DJANGO_SUPERUSER_USERNAME", "env-admin")
    monkeypatch.setenv("DJANGO_SUPERUSER_PASSWORD", "env-pass")
    monkeypatch.setenv("DJANGO_SUPERUSER_EMAIL", "env@example.com")

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.get_user_model",
        lambda: DummyUser,
    )

    stdout = StringIO()
    call_command("create_tenant_superuser", schema=tenant.schema_name, stdout=stdout)

    assert created_params["call"] == ("env-admin", "env@example.com", "env-pass")
    assert filter_calls == [{"username": "env-admin"}]
    assert recorded_calls == [("migrate_schemas", (), {"schema": tenant.schema_name})]
    assert "created" in stdout.getvalue()


@pytest.mark.django_db
def test_create_tenant_superuser_warns_when_no_changes_needed(monkeypatch):
    tenant = TenantFactory(schema_name="noop-superuser")
    tenant.create_schema(check_if_exists=True)

    monkeypatch.setattr(
        "customers.management.commands.create_tenant_superuser.call_command",
        lambda *args, **kwargs: None,
    )

    with schema_context(tenant.schema_name):
        User.objects.create_superuser(
            username="demo-admin",
            email="existing@example.com",
            password="existing-pass",
        )

    stdout = StringIO()
    call_command(
        "create_tenant_superuser",
        schema=tenant.schema_name,
        username="demo-admin",
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert "already exists" in output
