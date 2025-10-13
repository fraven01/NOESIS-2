import os
from typing import Iterator

import psycopg2
import pytest
from psycopg2 import OperationalError, errors, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, make_dsn, parse_dsn

from ai_core.rag import vector_client as rag_vector_client
from ai_core.rag.vector_schema import render_schema_sql
from ai_core.rag.vector_store import reset_default_router
from common import logging as common_logging

pytest.importorskip(
    "pytest_django",
    reason="pytest-django is required for ai_core tests",
)


@pytest.fixture(autouse=True)
def clear_structlog_context():
    common_logging.clear_log_context()
    try:
        yield
    finally:
        common_logging.clear_log_context()


DEFAULT_SCHEMA_NAME = "rag"


def render_vector_schema(
    schema_name: str = DEFAULT_SCHEMA_NAME, dimension: int = 1536
) -> str:
    if not schema_name:
        raise ValueError("schema_name must be provided")
    return render_schema_sql(schema_name, dimension)


def drop_schema(cur, schema_name: str = DEFAULT_SCHEMA_NAME) -> None:
    if not schema_name:
        raise ValueError("schema_name must be provided")
    cur.execute(
        sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name))
    )


def reset_vector_schema(
    cur, schema_name: str = DEFAULT_SCHEMA_NAME, *, dimension: int = 1536
) -> None:
    """Reset the RAG schema in a transaction-safe way.

    Executes DDL outside of the caller's transaction to avoid breaking
    pytest-django's transactional test wrappers. Prefer a dedicated
    autocommit connection; fall back to temporarily enabling autocommit
    on the caller's connection if we cannot reconstruct a DSN.
    """
    if not schema_name:
        raise ValueError("schema_name must be provided")

    # Prefer using an environment-provided DSN to create an isolated
    # autocommit connection for DDL; this avoids touching Django's
    # transactional test wrappers. Fall back to the caller's connection
    # with a temporary autocommit toggle if needed.
    dsn = os.environ.get("RAG_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if dsn:
        try:
            ddl_conn = psycopg2.connect(dsn)
            ddl_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            try:
                with ddl_conn.cursor() as ddl_cur:
                    drop_schema(ddl_cur, schema_name)
                    ddl_cur.execute(render_vector_schema(schema_name, dimension))
            finally:
                ddl_conn.close()
        except Exception:
            # Last-resort fallback to current connection autocommit path
            dsn = None

    if not dsn:
        conn = cur.connection  # type: ignore[attr-defined]
        try:
            prev_level = conn.isolation_level
        except Exception:
            prev_level = None
        try:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            drop_schema(cur, schema_name)
            cur.execute(render_vector_schema(schema_name, dimension))
        finally:
            if prev_level is not None:
                try:
                    conn.set_isolation_level(prev_level)
                except Exception:
                    pass

    # Apply search_path on the caller's session for subsequent statements
    cur.execute(
        sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema_name))
    )


SCHEMA_SQL = render_vector_schema()


def _extract_dbname(dsn: str) -> str | None:
    if not dsn:
        return None
    try:
        canonical = make_dsn(dsn=dsn)
    except Exception:
        canonical = dsn
    try:
        parsed = parse_dsn(canonical)
    except Exception:
        return None
    name = parsed.get("dbname") or parsed.get("database")
    return str(name) if name else None


@pytest.fixture(autouse=True, scope="session")
def ensure_vector_extensions_in_django_test_db(django_db_setup, django_db_blocker):
    """Ensure pgvector/pg_trgm are available in the Django test database.

    Some environments run pytest with a PostgreSQL cluster that does not have
    the pgvector extension pre-enabled in the template database. Django creates
    a fresh test database which therefore misses the extension, leading to
    operator-class errors when HNSW/IVFFLAT indexes are created during tests.

    We defensively enable the extensions in the active Django test database to
    keep tests self-contained and idempotent.
    """
    from django.db import connection

    with django_db_blocker.unblock():
        try:
            with connection.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    # Prefer the latest available version in the cluster
                    try:
                        cur.execute("ALTER EXTENSION vector UPDATE")
                    except Exception:
                        pass
                    # Normalise extension schema for predictable opclass lookup
                    try:
                        cur.execute("ALTER EXTENSION vector SET SCHEMA public")
                    except Exception:
                        pass
                except errors.UndefinedFile:
                    # Extension not installed in the cluster; leave to per-test skips
                    return
                # Trigram support for lexical side of hybrid search
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                    try:
                        cur.execute("ALTER EXTENSION pg_trgm SET SCHEMA public")
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            # Do not hard-fail the session on extension issues; individual tests
            # will raise/skip with clearer messages where relevant.
            return


@pytest.fixture(scope="session")
def rag_test_dsn() -> Iterator[str]:
    dsn = os.environ.get(
        "AI_CORE_TEST_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )
    try:
        conn = psycopg2.connect(dsn)
    except OperationalError:
        pytest.skip("PostgreSQL with pgvector extension is required for RAG tests")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    try:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Ensure latest available pgvector features (HNSW/IVFFLAT opclasses)
            try:
                cur.execute("ALTER EXTENSION vector UPDATE")
            except Exception:
                pass
            # Normalise extension schema for predictable opclass resolution
            try:
                cur.execute("ALTER EXTENSION vector SET SCHEMA public")
            except Exception:
                pass
        except errors.UndefinedFile as exc:
            pytest.skip(f"pgvector extension not available: {exc}")
        reset_vector_schema(cur, DEFAULT_SCHEMA_NAME)
    finally:
        cur.close()
        conn.close()
    yield dsn
    conn = psycopg2.connect(dsn)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    try:
        drop_schema(cur, DEFAULT_SCHEMA_NAME)
    finally:
        cur.close()
        conn.close()


@pytest.fixture
def rag_database(rag_test_dsn: str, monkeypatch, settings) -> Iterator[str]:
    from django.conf import settings as django_settings
    from django.db import connection

    original_default_config = django_settings.DATABASES["default"].copy()
    original_default_name = original_default_config.get("NAME")
    original_active_name = connection.settings_dict.get("NAME")
    env_dbname = _extract_dbname(rag_test_dsn)
    if env_dbname:
        new_config = dict(original_default_config, NAME=env_dbname)
        django_settings.DATABASES["default"] = new_config
        settings.DATABASES["default"] = dict(
            settings.DATABASES["default"], NAME=env_dbname
        )
        if original_active_name:
            connection.settings_dict["NAME"] = original_active_name

    conn = psycopg2.connect(rag_test_dsn)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(DEFAULT_SCHEMA_NAME)
            )
        )
        try:
            cur.execute("TRUNCATE TABLE embeddings, chunks, documents CASCADE")
        except (
            errors.UndefinedTable,
            errors.InvalidSchemaName,
        ):
            reset_vector_schema(cur, DEFAULT_SCHEMA_NAME)
            cur.execute("TRUNCATE TABLE embeddings, chunks, documents CASCADE")
    conn.close()
    monkeypatch.setenv("DATABASE_URL", rag_test_dsn)
    monkeypatch.setenv("RAG_DATABASE_URL", rag_test_dsn)
    rag_vector_client.reset_default_client()
    reset_default_router()
    try:
        yield rag_test_dsn
    finally:
        rag_vector_client.reset_default_client()
        reset_default_router()
        if env_dbname is not None:
            django_settings.DATABASES["default"] = original_default_config
            settings.DATABASES["default"] = dict(
                settings.DATABASES["default"], NAME=original_default_name
            )
            if original_active_name:
                connection.settings_dict["NAME"] = original_active_name
