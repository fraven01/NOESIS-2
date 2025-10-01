import os
from pathlib import Path

import pytest


pytest_plugins = [
    "tests.plugins.rag_db",
]


@pytest.fixture(autouse=True, scope="session")
def ensure_default_pii_secret():
    """Guarantee tests start with hardened PII defaults.

    Some CI environments define ``PII_HMAC_SECRET`` or tweak redaction toggles
    like ``PII_LOGGING_REDACTION`` to ease local debugging.  When these
    environment variables leak into the pytest process they flip the Django
    settings before the test suite has a chance to run, causing the "default"
    assertions in ``common/tests/test_pii_flags.py`` to fail.

    We therefore reset the critical defaults that industrial mode relies on:
    an empty HMAC secret and logging redaction enabled.
    """

    from pytest import MonkeyPatch
    from django.conf import settings as django_settings

    patcher = MonkeyPatch()
    patcher.delenv("PII_HMAC_SECRET", raising=False)
    patcher.delenv("PII_LOGGING_REDACTION", raising=False)
    django_settings.PII_HMAC_SECRET = ""
    django_settings.PII_LOGGING_REDACTION = True
    yield
    patcher.undo()
    django_settings.PII_HMAC_SECRET = ""
    django_settings.PII_LOGGING_REDACTION = True


@pytest.fixture(autouse=True, scope="session")
def ensure_tenant_engine():
    """Skip tests if the PostgreSQL tenant backend isn't configured."""
    from django.conf import settings

    if settings.DATABASES["default"]["ENGINE"] != "django_tenants.postgresql_backend":
        pytest.skip(
            "Tests require the django-tenants PostgreSQL backend",
            allow_module_level=True,
        )


@pytest.fixture(autouse=True)
def tmp_media_root(tmp_path, settings):
    """Store uploaded files under a per-test temporary MEDIA_ROOT."""
    media = tmp_path / "media"
    media.mkdir(parents=True, exist_ok=True)
    settings.MEDIA_ROOT = str(media)
    yield


@pytest.fixture(autouse=True)
def disable_auto_create_schema(monkeypatch):
    monkeypatch.setattr("customers.models.Tenant.auto_create_schema", False)


@pytest.fixture
def pii_config_env(monkeypatch, settings):
    """Configure PII-related environment variables for a test."""

    env_keys = (
        "PII_MODE",
        "PII_POLICY",
        "PII_DETERMINISTIC",
        "PII_POST_RESPONSE",
        "PII_LOGGING_REDACTION",
        "PII_HMAC_SECRET",
        "PII_NAME_DETECTION",
    )
    boolean_keys = {
        "PII_DETERMINISTIC",
        "PII_POST_RESPONSE",
        "PII_LOGGING_REDACTION",
        "PII_NAME_DETECTION",
    }
    baseline = {key: getattr(settings, key) for key in env_keys}

    def _apply(**overrides):
        config = baseline | overrides
        for key in env_keys:
            value = config[key]
            if key in boolean_keys:
                if isinstance(value, str):
                    boolean_value = value.lower() == "true"
                else:
                    boolean_value = bool(value)
                monkeypatch.setenv(key, "true" if boolean_value else "false")
                setattr(settings, key, boolean_value)
                config[key] = boolean_value
            elif key == "PII_HMAC_SECRET":
                secret = "" if value is None else str(value)
                monkeypatch.setenv(key, secret)
                setattr(settings, key, secret)
                config[key] = secret
            else:
                text = str(value)
                monkeypatch.setenv(key, text)
                setattr(settings, key, text)
                config[key] = text
        return config

    return _apply


@pytest.fixture(autouse=True, scope="session")
def ensure_public_schema(django_db_setup, django_db_blocker):
    """Ensure the shared (public) schema migrations run before tenant setup."""
    from django.core.management import call_command

    with django_db_blocker.unblock():
        call_command("migrate_schemas", shared=True, interactive=False, verbosity=0)
        try:
            call_command("init_public", verbosity=0)
        except Exception:  # pragma: no cover - optional bootstrap
            pass


@pytest.fixture(scope="session")
def test_tenant_schema_name(django_db_setup, django_db_blocker):
    """Create a dedicated tenant schema for tests and return its name.

    Use a non-default name ('autotest') to avoid clashes with
    django_tenants' TenantTestCase which uses 'test' by default.
    """
    from customers.models import Tenant, Domain

    with django_db_blocker.unblock():
        tenant, _ = Tenant.objects.get_or_create(
            schema_name="autotest", defaults={"name": "Autotest Tenant"}
        )
        # Ensure schema exists even though auto_create_schema is disabled in tests
        tenant.create_schema(check_if_exists=True)
        # Map default Django test host to this tenant to avoid 404 from tenant middleware
        Domain.objects.get_or_create(
            domain="testserver", tenant=tenant, defaults={"is_primary": True}
        )
    return tenant.schema_name


@pytest.fixture(autouse=True)
def use_test_tenant(request, test_tenant_schema_name):
    """Run each test inside the test tenant schema by default."""
    from django_tenants.utils import schema_context

    try:
        from django_tenants.test.cases import TenantTestCase
    except Exception:
        TenantTestCase = None

    cls = getattr(request.node, "cls", None)
    if TenantTestCase and cls and issubclass(cls, TenantTestCase):
        # Let TenantTestCase manage schema lifecycle itself
        yield
    else:
        with schema_context(test_tenant_schema_name):
            yield


@pytest.fixture(autouse=True, scope="session")
def cleanup_documents_test_files_session():
    """Safety net: remove stray documents/test*.txt in repo root, pre/post session."""
    repo_docs = Path(__file__).resolve().parent / "documents"
    for phase in ("pre", "post"):
        for p in repo_docs.glob("test*.txt"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        if phase == "pre":
            yield


@pytest.fixture(autouse=True)
def tolerate_existing_migration_table(monkeypatch):
    """Work around DuplicateTable race when MigrationRecorder.ensure_schema runs twice.

    In some tenant-test flows, the migration table may already exist even if
    introspection temporarily misses it. Treat the specific 'already exists'
    error as benign to keep migration tests stable.
    """
    from django.db.migrations.recorder import MigrationRecorder
    from django.db.migrations.exceptions import MigrationSchemaMissing

    original = MigrationRecorder.ensure_schema

    def _safe_ensure(self):
        try:
            return original(self)
        except MigrationSchemaMissing as exc:  # pragma: no cover - defensive
            msg = str(exc).lower()
            if "existiert bereits" in msg or "already exists" in msg:
                return None
            raise

    monkeypatch.setattr(MigrationRecorder, "ensure_schema", _safe_ensure)


def pytest_collection_modifyitems(config, items):
    """Skip gold-marked tests unless the gold feature mode is active."""

    if os.getenv("PII_MODE", "industrial") == "gold":
        return

    skip_gold = pytest.mark.skip(reason="requires PII_MODE=gold")
    for item in items:
        if "gold" in item.keywords:
            item.add_marker(skip_gold)


@pytest.fixture
def mocker(monkeypatch):
    """Lightweight replacement for pytest-mock's `mocker` fixture.

    Provides a `patch` method returning a MagicMock, compatible with
    `assert_called_with` / `assert_has_calls` usage in tests.
    """
    from unittest.mock import MagicMock, call

    class _Mocker:
        def patch(self, target, new=None, **kwargs):
            if new is None:
                new = MagicMock(**kwargs)
            module_path, attr = target.rsplit(".", 1)
            module = __import__(module_path, fromlist=[attr])
            monkeypatch.setattr(module, attr, new)
            return new

        def call(self, *args, **kwargs):  # helper passthrough
            return call(*args, **kwargs)

    return _Mocker()


@pytest.fixture(autouse=True)
def ensure_profiles_userprofile_isolation_testdata(request, django_db_blocker):
    """Ensure TestUserProfileIsolation.setUpTestData runs exactly once.

    This unittest-style class relies on class-level setup to define tenants
    (tenant1/tenant2). Pytest may not invoke setUpTestData for this base, so we
    call it explicitly in a narrowly-scoped way.
    """
    cls = getattr(request.node, "cls", None)
    if (
        cls
        and cls.__name__ == "TestUserProfileIsolation"
        and hasattr(cls, "setUpTestData")
    ):
        if not getattr(cls, "__setUpTestData_done__", False):
            with django_db_blocker.unblock():
                cls.setUpTestData()
            setattr(cls, "__setUpTestData_done__", True)
