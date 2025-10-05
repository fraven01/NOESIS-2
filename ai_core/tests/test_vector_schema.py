from __future__ import annotations

import uuid

import psycopg2
import pytest
from django.db import connection

from ai_core.rag.embedding_config import reset_embedding_configuration_cache
from ai_core.rag.vector_schema import (
    VectorSchemaError,
    VectorSchemaErrorCode,
    build_vector_schema_plan,
    render_schema_sql,
    validate_vector_schemas,
)


def _configure_vector_spaces(settings, spaces):
    settings.RAG_VECTOR_STORES = spaces
    settings.RAG_EMBEDDING_PROFILES = {
        "default": {
            "model": "test-model",
            "dimension": next(iter(spaces.values()))["dimension"],
            "vector_space": next(iter(spaces.keys())),
        }
    }
    reset_embedding_configuration_cache()


def test_build_vector_schema_plan_renders_per_space(settings) -> None:
    spaces = {
        "global": {"backend": "pgvector", "schema": "rag_alpha", "dimension": 6},
        "archive": {"backend": "pgvector", "schema": "rag_archive", "dimension": 8},
    }
    _configure_vector_spaces(settings, spaces)

    plan = build_vector_schema_plan()

    assert {ddl.space_id for ddl in plan} == {"global", "archive"}
    assert all(f"SET search_path TO {spaces[ddl.space_id]['schema']}" in ddl.sql for ddl in plan)
    assert any("vector(6)" in ddl.sql for ddl in plan)
    assert any("vector(8)" in ddl.sql for ddl in plan)

    reset_embedding_configuration_cache()


def test_validate_vector_schemas_detects_dimension_conflict(settings) -> None:
    spaces = {
        "primary": {"backend": "pgvector", "schema": "rag_conflict", "dimension": 3},
        "secondary": {"backend": "pgvector", "schema": "rag_conflict", "dimension": 7},
    }
    settings.RAG_VECTOR_STORES = spaces
    settings.RAG_EMBEDDING_PROFILES = {
        "p": {"model": "m", "dimension": 3, "vector_space": "primary"}
    }
    reset_embedding_configuration_cache()

    with pytest.raises(VectorSchemaError) as excinfo:
        validate_vector_schemas()
    assert VectorSchemaErrorCode.SCHEMA_DIMENSION_CONFLICT in str(excinfo.value)

    reset_embedding_configuration_cache()


@pytest.mark.django_db
def test_rendered_schema_sql_enforces_dimension_guard(settings) -> None:
    schema_name = "rag_smoke"
    dimension = 4
    sql = render_schema_sql(schema_name, dimension)

    params = connection.get_connection_params()
    ddl_conn = psycopg2.connect(**params)
    skip_exc: Exception | None = None
    try:
        ddl_conn.set_session(autocommit=True)
        with ddl_conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            try:
                cur.execute(sql)
            except psycopg2.errors.InsufficientPrivilege as exc:
                ddl_conn.rollback()
                skip_exc = exc

        if skip_exc is None:
            with ddl_conn.cursor() as cur:
                cur.execute(f"SET search_path TO {schema_name}, public")
                doc_id = uuid.uuid4()
                chunk_id = uuid.uuid4()
                tenant_id = uuid.uuid4()
                cur.execute(
                    "INSERT INTO documents (id, tenant_id, source, hash, metadata)"
                    " VALUES (%s, %s, %s, %s, '{}'::jsonb)",
                    (doc_id, str(tenant_id), "test", "hash"),
                )
                cur.execute(
                    "INSERT INTO chunks (id, document_id, ord, text, tokens, metadata)"
                    " VALUES (%s, %s, %s, %s, %s, '{}'::jsonb)",
                    (chunk_id, doc_id, 0, "chunk", 5),
                )
                vector_literal = "[" + ",".join("0" for _ in range(dimension)) + "]"
                cur.execute(
                    "INSERT INTO embeddings (id, chunk_id, embedding)"
                    " VALUES (%s, %s, %s::vector)",
                    (uuid.uuid4(), chunk_id, vector_literal),
                )

                wrong_literal = "[" + ",".join("0" for _ in range(dimension + 1)) + "]"
                with pytest.raises(psycopg2.errors.DataException):
                    cur.execute(
                        "INSERT INTO embeddings (id, chunk_id, embedding)"
                        " VALUES (%s, %s, %s::vector)",
                        (uuid.uuid4(), chunk_id, wrong_literal),
                    )
    finally:
        try:
            ddl_conn.rollback()
        except Exception:
            pass
        if skip_exc is None:
            with ddl_conn.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
        ddl_conn.close()

    if skip_exc is not None:
        pytest.skip(f"requires extension privileges: {skip_exc}")

    reset_embedding_configuration_cache()
