import os
from pathlib import Path
from typing import Iterator

import psycopg2
import pytest
from psycopg2 import OperationalError, errors
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from ai_core.rag import vector_client as rag_vector_client
from ai_core.rag.vector_store import reset_default_router
from common import logging as common_logging

os.environ.setdefault("RAG_EMBEDDING_DIM", "1536")

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


SCHEMA_SQL = (
    Path(__file__).resolve().parents[2] / "docs" / "rag" / "schema.sql"
).read_text()


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
        cur.execute("DROP SCHEMA IF EXISTS rag CASCADE")
        cur.execute("CREATE SCHEMA rag")
        cur.execute("SET search_path TO rag, public")
        cur.execute(SCHEMA_SQL)
    finally:
        cur.close()
        conn.close()
    yield dsn
    conn = psycopg2.connect(dsn)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    try:
        cur.execute("DROP SCHEMA IF EXISTS rag CASCADE")
    finally:
        cur.close()
        conn.close()


@pytest.fixture
def rag_database(rag_test_dsn: str, monkeypatch) -> Iterator[str]:
    conn = psycopg2.connect(rag_test_dsn)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("SET search_path TO rag, public")
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
