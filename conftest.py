import os
import re
from pathlib import Path

import pytest

# Disable OTEL exporters in tests to avoid noisy connection errors.
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["LOGGING_OTEL_INSTRUMENT"] = "false"
os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")
pgoptions = os.environ.get("PGOPTIONS", "")
if "search_path" in pgoptions:
    pgoptions = re.sub(r"-c\s*search_path=\S+", "-c search_path=public", pgoptions)
    pgoptions = re.sub(r"-csearch_path=\S+", "-c search_path=public", pgoptions)
else:
    pgoptions = f"{pgoptions} -c search_path=public"
os.environ["PGOPTIONS"] = " ".join(pgoptions.split()).strip()


def _patch_migration_recorder() -> None:
    """Ensure MigrationRecorder always has a valid schema selected."""
    try:
        from django.db.migrations.recorder import MigrationRecorder
        from django.db.migrations.exceptions import MigrationSchemaMissing

        try:
            from django_tenants.utils import get_public_schema_name
        except Exception:
            get_public_schema_name = None
    except Exception:
        return

    if getattr(MigrationRecorder, "_noesis_safe_patch", False):
        return

    original = MigrationRecorder.ensure_schema

    def _ensure_schema(self):
        try:
            from django.db import connection
        except Exception:
            return original(self)

        public_schema = "public"
        if get_public_schema_name:
            try:
                public_schema = get_public_schema_name()
            except Exception:
                pass
        target_schema = getattr(connection, "schema_name", None) or public_schema

        try:
            if (
                hasattr(connection, "set_schema_to_public")
                and target_schema == public_schema
            ):
                connection.set_schema_to_public()
            elif hasattr(connection, "set_schema"):
                connection.set_schema(target_schema)
            else:
                with connection.cursor() as cursor:
                    quoted = connection.ops.quote_name(target_schema)
                    cursor.execute(f"SET search_path TO {quoted}")
        except Exception:
            pass

        try:
            with connection.cursor() as cursor:
                quoted = connection.ops.quote_name(target_schema)
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")
        except Exception:
            pass

        try:
            return original(self)
        except MigrationSchemaMissing as exc:
            msg = str(exc).lower()
            if "existiert bereits" in msg or "already exists" in msg:
                return None
            raise

    MigrationRecorder.ensure_schema = _ensure_schema
    MigrationRecorder._noesis_safe_patch = True


def pytest_configure(config):
    """Ensure migrations always target a valid schema in tenant tests."""
    _patch_migration_recorder()


@pytest.fixture(autouse=True, scope="session")
def patch_migration_recorder_schema():
    _patch_migration_recorder()
    yield


@pytest.fixture(autouse=True)
def stub_embedding_client(monkeypatch):
    from ai_core.rag import embeddings as embeddings_module
    from ai_core.rag.embeddings import EmbeddingBatchResult

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
def django_db_modify_db_settings(request):
    """Ensure the test connection defaults to the public schema AND supports xdist isolation."""
    from django.conf import settings

    # 1. Handle xdist worker isolation (restore pytest-django behavior we overrode)
    # Check if we are running in an xdist worker
    worker_id = None
    if hasattr(request.config, "workerinput"):
        worker_id = request.config.workerinput.get("workerid")

    if worker_id:
        db_settings = settings.DATABASES["default"]
        original_name = db_settings.get("NAME")
        if original_name and worker_id not in original_name:
            # Standard pytest-django behavior: append _gwN
            db_settings["NAME"] = f"{original_name}_{worker_id}"
            # Check if TEST dict exists (Django 4+ sometimes separate)
            if "TEST" in db_settings:
                test_name = db_settings["TEST"].get("NAME")
                if test_name and worker_id not in test_name:
                    db_settings["TEST"]["NAME"] = f"{test_name}_{worker_id}"

    # 2. Handle django-tenants public schema enforcement
    if settings.DATABASES["default"]["ENGINE"] != "django_tenants.postgresql_backend":
        return

    public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
    options = settings.DATABASES["default"].setdefault("OPTIONS", {})
    existing = str(options.get("options", "")).strip()
    if existing:
        existing = re.sub(r"-c\s*search_path=\S+", "", existing)
        existing = re.sub(r"-csearch_path=\S+", "", existing)
        existing = " ".join(existing.split())
    search_path_opt = f"-c search_path={public_schema}"
    options["options"] = f"{existing} {search_path_opt}".strip()


@pytest.fixture(scope="session")
def django_db_setup(
    request,
    django_db_modify_db_settings,
    django_db_blocker,
    patch_migration_recorder_schema,
):
    """Custom django_db_setup that handles xdist database creation and 'already exists' errors.

    For parallel tests (pytest-xdist), each worker needs its own database.
    This fixture creates the worker database if it doesn't exist, then runs migrations.
    It also handles 'already exists' errors gracefully for --reuse-db scenarios.
    """
    from django.conf import settings
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    db_settings = settings.DATABASES["default"]
    db_name = db_settings.get("NAME")

    # Check if we need to create the database (for xdist workers)
    if db_name:
        # Connect to the default 'postgres' database to create our target database
        try:
            conn = psycopg2.connect(
                dbname="postgres",
                user=db_settings.get("USER"),
                password=db_settings.get("PASSWORD"),
                host=db_settings.get("HOST"),
                port=db_settings.get("PORT", 5432),
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", [db_name])
            exists = cursor.fetchone()

            if not exists:
                # Create the database
                cursor.execute(f'CREATE DATABASE "{db_name}"')

            cursor.close()
            conn.close()
        except psycopg2.Error:
            # If we can't connect to postgres, assume the database exists or will be created
            pass

    with django_db_blocker.unblock():
        from django.core.management import call_command
        from django.conf import settings

        # For django-tenants we run shared migrations in ensure_public_schema.
        if (
            settings.DATABASES["default"]["ENGINE"]
            == "django_tenants.postgresql_backend"
        ):
            return

        # Try to run migrations, catching "already exists" errors
        try:
            call_command(
                "migrate",
                verbosity=0,
                interactive=False,
                run_syncdb=True,
            )
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" not in error_str:
                raise e
            # Tables exist, migrations effectively complete


@pytest.fixture(autouse=True)
def default_lifecycle_store(monkeypatch):
    """Use in-memory lifecycle store in tests to avoid tenant FK churn."""
    import documents.repository as doc_repo

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
    from testsupport.tenants import TenantFactoryHelper

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

    # In xdist, if workers share the DB (or race on shared schemas), we MUST synchronize
    # the public schema migration to avoid DuplicateTable errors.
    from filelock import FileLock

    # Use a lock file scoped to the active test database (xdist worker isolation).
    db_name = connection.settings_dict.get("NAME", "default")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(db_name))
    lock_file = f"/tmp/public_schema_migration_{safe_name}.lock"

    with FileLock(lock_file):
        with django_db_blocker.unblock():

            # Ensure we are targeting the public schema
            if hasattr(connection, "set_schema_to_public"):
                connection.set_schema_to_public()
            elif hasattr(connection, "set_schema"):
                connection.set_schema(get_public_schema_name())

            # Run shared migrations inside the lock.
            # We catch specific errors that indicate race conditions or partial states.
            import logging

            logger = logging.getLogger(__name__)

            try:
                logger.info("Running public schema migrations...")
                call_command(
                    "migrate_schemas", shared=True, interactive=False, verbosity=1
                )
                logger.info("Public schema migrations completed successfully")
            except Exception as e:
                msg = str(e).lower()
                if "already exists" in msg or "duplicate" in msg:
                    logger.debug(
                        f"Public schema migration: tables already exist (expected): {msg}"
                    )
                else:
                    logger.error(f"Public schema migration failed: {e}", exc_info=True)
                    # Fallback: try standard migrate if the schema command failed oddly
                    try:
                        logger.info(
                            "Attempting fallback migration for customers app..."
                        )
                        call_command(
                            "migrate", "customers", interactive=False, verbosity=1
                        )
                        logger.info("Fallback migration completed")
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback migration also failed: {fallback_error}",
                            exc_info=True,
                        )
                        raise  # Re-raise to make the test setup failure visible

            try:
                call_command("init_public", verbosity=0)
            except Exception as init_error:
                logger.warning(
                    f"init_public command failed (may be expected): {init_error}"
                )


@pytest.fixture(scope="session")
def tenant_pool_schemas():
    """Provide stable, worker-scoped tenant schema names for pooled tests."""
    worker = os.getenv("PYTEST_XDIST_WORKER")
    prefix = f"pool_{worker}" if worker else "pool"
    return {
        "alpha": f"{prefix}_alpha",
        "beta": f"{prefix}_beta",
        "gamma": f"{prefix}_gamma",
        "delta": f"{prefix}_delta",
    }


@pytest.fixture(scope="session")
def tenant_pool(django_db_setup, django_db_blocker, tenant_pool_schemas):
    """Create a small pool of pre-migrated tenants reused across tests."""
    from django.conf import settings

    if settings.DATABASES["default"]["ENGINE"] != "django_tenants.postgresql_backend":
        return {}

    from django_tenants.utils import get_public_schema_name, schema_context
    from customers.models import Tenant
    from testsupport.tenant_fixtures import bootstrap_tenant_schema

    tenants: dict[str, Tenant] = {}
    original_auto_create = getattr(Tenant, "auto_create_schema", None)
    with django_db_blocker.unblock():
        try:
            Tenant.auto_create_schema = False
            with schema_context(get_public_schema_name()):
                for label, schema_name in tenant_pool_schemas.items():
                    tenant, created = Tenant.objects.get_or_create(
                        schema_name=schema_name,
                        defaults={"name": f"Pool {label.title()}"},
                    )
                    if not created:
                        desired_name = f"Pool {label.title()}"
                        if tenant.name != desired_name:
                            tenant.name = desired_name
                            tenant.save(update_fields=["name"])
                    tenants[label] = tenant

            for tenant in tenants.values():
                bootstrap_tenant_schema(tenant, migrate=True)
        finally:
            if original_auto_create is not None:
                Tenant.auto_create_schema = original_auto_create

    return tenants


@pytest.fixture(scope="session")
def test_tenant_schema_name(django_db_setup, django_db_blocker, ensure_public_schema):
    """Create a dedicated tenant schema for tests and return its name.

    Use a non-default name (default 'autotest', configurable via settings)
    to avoid clashes with
    django_tenants' TenantTestCase which uses 'test' by default.
    """
    from django.conf import settings
    from testsupport.tenant_fixtures import (
        DEFAULT_TEST_DOMAIN,
        bootstrap_tenant_schema,
        ensure_tenant_domain,
    )

    if "django_tenants" not in settings.INSTALLED_APPS:
        return "public"

    from customers.models import Tenant

    # CRITICAL: Disable auto_create_schema BEFORE any Tenant operations
    # This prevents django-tenants from triggering implicit migrations during save()
    # which can fail if the schema exists but tables don't, or cause cascade queries
    # to non-existent tables.
    Tenant.auto_create_schema = False

    from testsupport.tenant_fixtures import _advisory_lock

    test_schema = getattr(settings, "TEST_TENANT_SCHEMA", "autotest")

    with django_db_blocker.unblock():
        from django_tenants.utils import schema_context, get_public_schema_name
        from django.db import connection

        with _advisory_lock(f"tenant:{test_schema}"):
            with schema_context(get_public_schema_name()):
                tenant, _ = Tenant.objects.get_or_create(
                    schema_name=test_schema, defaults={"name": "Autotest Tenant"}
                )

            bootstrap_tenant_schema(tenant, migrate=True)

            # Verification: Check if critical tables exist.
            # If not, it means migration failed or state is inconsistent.
            required_tables = (
                "users_user",
                "cases_case",
                "documents_savedsearch",
                "documents_notificationdelivery",
            )
            missing_tables: list[str] = []
            with schema_context(tenant.schema_name):
                with connection.cursor() as cursor:
                    for table in required_tables:
                        cursor.execute("SELECT to_regclass(%s)", [table])
                        if cursor.fetchone()[0] is None:
                            missing_tables.append(table)

            if missing_tables:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Missing tables in {test_schema} schema: {missing_tables}. "
                    "Rebuilding schema..."
                )
                try:
                    from django.core.management import call_command

                    logger.info(
                        f"Attempting targeted documents migration for {test_schema}."
                    )
                    with schema_context(tenant.schema_name):
                        call_command(
                            "migrate",
                            "documents",
                            interactive=False,
                            verbosity=1,
                        )
                    missing_tables = []
                    with schema_context(tenant.schema_name):
                        with connection.cursor() as cursor:
                            for table in required_tables:
                                cursor.execute("SELECT to_regclass(%s)", [table])
                                if cursor.fetchone()[0] is None:
                                    missing_tables.append(table)
                    if not missing_tables:
                        logger.info(f"Documents migrations applied for {test_schema}.")
                except Exception:
                    logger.warning(
                        f"Targeted documents migration failed for {test_schema}.",
                        exc_info=True,
                    )
                if missing_tables:
                    try:
                        from testsupport import (
                            tenant_fixtures as tenant_fixtures_module,
                        )

                        try:
                            tenant_fixtures_module._MIGRATED_SCHEMAS.discard(
                                test_schema
                            )
                        except Exception:
                            pass
                    except Exception:
                        pass

                    try:
                        from django.db import connection

                        try:
                            connection.rollback()
                        except Exception:
                            pass
                        if hasattr(connection, "set_schema_to_public"):
                            connection.set_schema_to_public()
                        with connection.cursor() as cursor:
                            quoted = connection.ops.quote_name(test_schema)
                            cursor.execute(f"DROP SCHEMA IF EXISTS {quoted} CASCADE")
                    except Exception:
                        logger.error(
                            f"Failed to drop schema {test_schema} before rebuild.",
                            exc_info=True,
                        )
                        raise

                    try:
                        bootstrap_tenant_schema(tenant, migrate=True)
                    except Exception as e:
                        logger.error(f"Migration failed: {e}", exc_info=True)
                        raise

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
        try:
            context_manager = schema_context(
                test_tenant_schema_name, include_public=True
            )
        except TypeError:
            context_manager = schema_context(test_tenant_schema_name)
        with context_manager:
            try:
                from django.conf import settings
                from django.db import connection

                public_schema = getattr(settings, "PUBLIC_SCHEMA_NAME", "public")
                tenant_schema = test_tenant_schema_name
                quoted_tenant = connection.ops.quote_name(tenant_schema)
                quoted_public = connection.ops.quote_name(public_schema)
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"SET search_path TO {quoted_tenant}, {quoted_public}"
                    )
            except Exception:
                pass
            # Verification logic moved to test_tenant_schema_name, here we just yield
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
def cleanup_test_tenants_fixture(request, django_db_blocker, tenant_pool_schemas):
    """Drop schemas for tenants created during an individual test."""

    yield
    if request.node.get_closest_marker("slow") or request.node.get_closest_marker(
        "tenant_ops"
    ):
        return
    with django_db_blocker.unblock():
        from django.db import connection
        import logging
        from testsupport.tenant_fixtures import cleanup_test_tenants

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
            cleanup_test_tenants(preserve=list(tenant_pool_schemas.values()))
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
def cleanup_test_tenants_session(django_db_blocker, tenant_pool_schemas):
    """Final cleanup pass for schemas created during the test session."""

    yield
    with django_db_blocker.unblock():
        from django.db import connection
        import logging
        from testsupport.tenant_fixtures import cleanup_test_tenants

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
            cleanup_test_tenants(preserve=list(tenant_pool_schemas.values()))
        except Exception:
            logging.getLogger("testsupport").warning(
                "Session tenant cleanup failed", exc_info=True
            )
        finally:
            try:
                connection.ensure_connection()
            except Exception:
                pass
