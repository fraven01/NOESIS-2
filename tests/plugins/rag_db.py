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
    dsn = (
        os.environ.get("AI_CORE_TEST_DATABASE_URL")
        or os.environ.get("RAG_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql://postgres:postgres@localhost:5432/postgres"
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
    default_db_name = str(settings.DATABASES["default"].get("NAME") or "")

    effective_dsn = rag_test_dsn
    try:
        canonical_dsn = make_dsn(dsn=rag_test_dsn)
        parsed = parse_dsn(canonical_dsn)
    except Exception:
        parsed = {}
    else:
        current_name = parsed.get("dbname") or parsed.get("database")
        if default_db_name and current_name and current_name != default_db_name:
            parsed = {key: value for key, value in parsed.items() if value is not None}
            parsed["dbname"] = default_db_name
            effective_dsn = make_dsn(**parsed)

    conn = psycopg2.connect(effective_dsn)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    schema_initialised = False
    skip_reason: str | None = None
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except errors.InsufficientPrivilege:
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )
                result = cur.fetchone()
                if result is None:
                    skip_reason = "pgvector extension is required for RAG tests"
            if skip_reason is None:
                reset_vector_schema(cur, DEFAULT_SCHEMA_NAME)
                schema_initialised = True
    finally:
        conn.close()

    if skip_reason is not None:
        pytest.skip(skip_reason)

    monkeypatch.setenv("DATABASE_URL", effective_dsn)
    monkeypatch.setenv("RAG_DATABASE_URL", effective_dsn)
    rag_vector_client.reset_default_client()
    reset_default_router()
    try:
        yield effective_dsn
    finally:
        rag_vector_client.reset_default_client()
        reset_default_router()
        if schema_initialised:
            conn = psycopg2.connect(effective_dsn)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                reset_vector_schema(cur, DEFAULT_SCHEMA_NAME)
            conn.close()
