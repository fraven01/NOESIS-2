import os
import re
from pathlib import Path
from typing import Iterator

import psycopg2
import pytest
from psycopg2 import OperationalError, errors, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, make_dsn, parse_dsn

from ai_core.rag import vector_client as rag_vector_client
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

SCHEMA_SQL_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "docs" / "rag" / "schema.sql"
).read_text()

_SCHEMA_TOKEN_PATTERN = re.compile(rf"\b{re.escape(DEFAULT_SCHEMA_NAME)}\b")


def render_schema_sql(schema_name: str = DEFAULT_SCHEMA_NAME) -> str:
    if not schema_name:
        raise ValueError("schema_name must be provided")
    return _SCHEMA_TOKEN_PATTERN.sub(schema_name, SCHEMA_SQL_TEMPLATE)


def drop_schema(cur, schema_name: str = DEFAULT_SCHEMA_NAME) -> None:
    if not schema_name:
        raise ValueError("schema_name must be provided")
    cur.execute(
        sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name))
    )


def reset_vector_schema(cur, schema_name: str = DEFAULT_SCHEMA_NAME) -> None:
    drop_schema(cur, schema_name)
    cur.execute(render_schema_sql(schema_name))
    cur.execute(
        sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema_name))
    )


SCHEMA_SQL = render_schema_sql()


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
