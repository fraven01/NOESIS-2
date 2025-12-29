import os
from pathlib import Path

import pytest

from ai_core.rag.embeddings import EmbeddingBatchResult
import documents.repository as doc_repo
from testsupport.tenant_fixtures import (
    DEFAULT_TEST_DOMAIN,
    bootstrap_tenant_schema,
    cleanup_test_tenants,
    ensure_tenant_domain,
)
from testsupport.tenants import TenantFactoryHelper


@pytest.fixture(autouse=True)
def stub_embedding_client(monkeypatch):
    from ai_core.rag import embeddings as embeddings_module

    class _DummyEmbeddingClient:
        def __init__(self) -> None:
            try:
                self._dim = int(os.getenv("TEST_EMBEDDING_DIM", "1536"))
            except (TypeError, ValueError):
                self._dim = 1536
            self.batch_size = 64

        def embed(self, texts):
            vectors = []
            for text in texts:
                normalized = (text or "").strip()
                magnitude = float(len(normalized.split()) or 1)
                tail = max(0, self._dim - 1)
                vectors.append([magnitude] + [0.0] * tail)
            return EmbeddingBatchResult(
                vectors=vectors,
                model="dummy-embed",
                model_used="primary",
                attempts=1,
                timeout_s=None,
            )

        def dim(self) -> int:
            return self._dim

    dummy = _DummyEmbeddingClient()
    monkeypatch.setattr(embeddings_module, "get_embedding_client", lambda: dummy)
    monkeypatch.setattr(embeddings_module, "_default_client", dummy, raising=False)
    yield


pytest_plugins = [
    "tests.plugins.rag_db",
]


@pytest.fixture(scope="session")
def django_db_modify_db_settings():
    """Ensure the test connection defaults to the public schema."""
    from django.conf import settings

    if settings.DATABASES["default"]["ENGINE"] != "django_tenants.postgresql_backend":
        return

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    options = settings.DATABASES["default"].setdefault("OPTIONS", {})
    existing = str(options.get("options", "")).strip()
    search_path_opt = f"-c search_path={public_schema}"
    if search_path_opt not in existing:
        options["options"] = f"{existing} {search_path_opt}".strip()


@pytest.fixture(autouse=True)
def default_lifecycle_store(monkeypatch):
    """Use in-memory lifecycle store in tests to avoid tenant FK churn."""

    store = doc_repo.DocumentLifecycleStore()
    monkeypatch.setattr(
        "documents.repository.DEFAULT_LIFECYCLE_STORE", store, raising=False
    )
    yield


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
    """Gate tenant-backed RAG fixtures on the expected Django engine."""
    from django.conf import settings

    if settings.DATABASES["default"]["ENGINE"] != "django_tenants.postgresql_backend":
        # Not all test environments provision the tenant backend (for example CI
        # jobs that only exercise non-RAG components). Returning early keeps the
        # RAG-specific fixtures idle while still allowing unrelated test modules
        # to execute instead of marking the entire session as skipped.
        return


@pytest.fixture(autouse=True)
def tmp_media_root(tmp_path, settings):
    """Store uploaded files under a per-test temporary MEDIA_ROOT."""
    media = tmp_path / "media"
    media.mkdir(parents=True, exist_ok=True)
    settings.MEDIA_ROOT = str(media)
    yield


@pytest.fixture(autouse=True)
def force_inmemory_repository(monkeypatch):
    """Use the in-memory documents repository in tests to avoid tenant FK churn."""
    from documents.repository import InMemoryDocumentsRepository
    import ai_core.services as services

    repo = InMemoryDocumentsRepository()
    monkeypatch.setattr(services, "_DOCUMENTS_REPOSITORY", repo, raising=False)
    yield


@pytest.fixture(autouse=True)
def disable_auto_create_schema(monkeypatch):
    monkeypatch.setattr("customers.models.Tenant.auto_create_schema", False)


@pytest.fixture
def tenant_factory():
    helper = TenantFactoryHelper()
    try:
        yield helper.create
    finally:
        helper.cleanup()


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
    from django.conf import settings
    from django.db import connection
    from django_tenants.utils import get_public_schema_name

    if "django_tenants" not in settings.INSTALLED_APPS:
        return

    with django_db_blocker.unblock():
        if hasattr(connection, "set_schema_to_public"):
            connection.set_schema_to_public()
        elif hasattr(connection, "set_schema"):
            connection.set_schema(get_public_schema_name())
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
    from django.conf import settings

    if "django_tenants" not in settings.INSTALLED_APPS:
        return "public"

    from customers.models import Tenant

    with django_db_blocker.unblock():
        tenant, _ = Tenant.objects.get_or_create(
            schema_name="autotest", defaults={"name": "Autotest Tenant"}
        )
        bootstrap_tenant_schema(tenant)
        ensure_tenant_domain(tenant, domain=DEFAULT_TEST_DOMAIN)
    return tenant.schema_name


@pytest.fixture(autouse=True)
def use_test_tenant(request, test_tenant_schema_name):
    """Run each test inside the test tenant schema by default."""
    from django.conf import settings

    if "django_tenants" not in settings.INSTALLED_APPS:
        yield
        return

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
def cleanup_test_tenants_fixture(django_db_blocker):
    """Drop schemas for tenants created during an individual test."""

    yield
    with django_db_blocker.unblock():
        from django.db import connection
        import logging

        # Skip cleanup inside atomic blocks to avoid opening a new connection there.
        if getattr(connection, "in_atomic_block", False):
            return

        # Reset transaction state without leaving a closed connection handle behind.
        try:
            try:
                connection.rollback()
            except Exception:
                pass
        except Exception:
            pass

        try:
            cleanup_test_tenants()
        except Exception:
            # If cleanup fails (e.g. DB unavailable), just log it
            logging.getLogger("testsupport").warning(
                "Final tenant cleanup failed", exc_info=True
            )
        finally:
            try:
                connection.ensure_connection()
            except Exception:
                pass


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_tenants_session(django_db_blocker):
    """Final cleanup pass for schemas created during the test session."""

    yield
    with django_db_blocker.unblock():
        from django.db import connection
        import logging

        try:
            try:
                connection.close()
            except Exception:
                pass
            try:
                connection.connection = None
            except Exception:
                pass
        except Exception:
            pass

        try:
            cleanup_test_tenants()
        except Exception:
            logging.getLogger("testsupport").warning(
                "Session tenant cleanup failed", exc_info=True
            )
        finally:
            try:
                connection.ensure_connection()
            except Exception:
                pass
