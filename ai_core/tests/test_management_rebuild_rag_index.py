from __future__ import annotations

import io
import json
import os
import re

import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn
from django.core.management import call_command
from django.db import connection
from django.core.files.uploadedfile import SimpleUploadedFile

from ai_core.ingestion import process_document
from ai_core.management.commands.rebuild_rag_index import Command
from ai_core.rag import vector_client
from ai_core.infra import object_store, rate_limit
from tests.plugins.rag_db import drop_schema, reset_vector_schema
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)


def _parse_dbname(dsn: str) -> str | None:
    try:
        parsed = parse_dsn(dsn)
    except Exception:
        try:
            canonical = make_dsn(dsn)
        except Exception:
            return None
        try:
            parsed = parse_dsn(canonical)
        except Exception:
            return None
    name = parsed.get("dbname") or parsed.get("database")
    return str(name) if name else None


def _assert_env_matches_default_database(settings) -> tuple[str, str]:
    env_dsn = os.environ.get("RAG_DATABASE_URL") or os.environ.get("DATABASE_URL")
    assert env_dsn, "Expected RAG_DATABASE_URL or DATABASE_URL to be set"

    env_dbname = _parse_dbname(env_dsn)
    assert env_dbname, "Unable to parse database name from environment DSN"

    default_config = settings.DATABASES["default"]
    default_name = str(default_config.get("NAME"))
    assert (
        env_dbname == default_name
    ), "Environment DSN should target Django's default database configuration"

    active_name = str(connection.settings_dict.get("NAME"))
    assert (
        active_name != env_dbname
    ), "Django connection should point to the test database clone during pytest"
    return env_dbname, active_name


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
@pytest.mark.parametrize(
    "index_kind,settings_overrides,expected_index,expected_params",
    [
        (
            "HNSW",
            {"RAG_HNSW_M": 18, "RAG_HNSW_EF_CONSTRUCTION": 88},
            "embeddings_embedding_hnsw",
            {"m": 18, "ef_construction": 88},
        ),
        (
            "IVFFLAT",
            {"RAG_IVF_LISTS": 512},
            "embeddings_embedding_ivfflat",
            {"lists": 512},
        ),
    ],
)
def test_rebuild_rag_index_creates_expected_index(
    settings,
    index_kind: str,
    settings_overrides: dict[str, int],
    expected_index: str,
    expected_params: dict[str, int],
) -> None:
    _assert_env_matches_default_database(settings)
    vector_client.reset_default_client()

    settings.RAG_INDEX_KIND = index_kind
    for key, value in settings_overrides.items():
        setattr(settings, key, value)

    stdout = io.StringIO()
    call_command("rebuild_rag_index", stdout=stdout)
    message = stdout.getvalue()
    assert f"{index_kind}" in message

    with connection.cursor() as cur:
        cur.execute("SET search_path TO rag, public")
        cur.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'embeddings'
              AND indexname = %s
            """,
            (expected_index,),
        )
        row = cur.fetchone()

    assert row is not None, f"expected index {expected_index} to be present"
    indexdef = row[0]

    for param, value in expected_params.items():
        pattern = rf"{param}\s*=\s*'?{value}'?(?:::\w+)?"
        assert re.search(pattern, indexdef), f"missing {param}={value} in {indexdef!r}"


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_rebuild_rag_index_uses_scope_with_default_flag(settings) -> None:
    settings.RAG_VECTOR_STORES = {
        "global": {"backend": "pgvector", "schema": "rag"},
        "enterprise": {
            "backend": "pgvector",
            "schema": "rag_enterprise",
            "default": True,
        },
    }
    if hasattr(settings, "RAG_VECTOR_DEFAULT_SCOPE"):
        delattr(settings, "RAG_VECTOR_DEFAULT_SCOPE")

    vector_client.reset_default_client()

    with connection.cursor() as cur:
        reset_vector_schema(cur, "rag_enterprise")

    try:
        stdout = io.StringIO()
        call_command("rebuild_rag_index", stdout=stdout)

        with connection.cursor() as cur:
            cur.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier("rag_enterprise")
                )
            )
            cur.execute(
                """
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = current_schema()
                  AND tablename = 'embeddings'
                  AND indexname = 'embeddings_embedding_hnsw'
                """
            )
            row = cur.fetchone()
        assert row is not None, "expected HNSW index in rag_enterprise schema"
    finally:
        with connection.cursor() as cur:
            drop_schema(cur, "rag_enterprise")


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_rebuild_rag_index_falls_back_when_cosine_ops_missing(
    monkeypatch, settings
) -> None:
    _assert_env_matches_default_database(settings)
    vector_client.reset_default_client()

    settings.RAG_INDEX_KIND = "HNSW"


    def fake_operator_class_exists(cur, operator_class: str, access_method: str) -> bool:

        if operator_class == "vector_cosine_ops":
            return False
        return operator_class == "vector_l2_ops"

    monkeypatch.setattr(
        vector_client,
        "operator_class_exists",
        fake_operator_class_exists,
        raising=True,
    )

    stdout = io.StringIO()
    call_command("rebuild_rag_index", stdout=stdout)

    with connection.cursor() as cur:
        cur.execute("SET search_path TO rag, public")
        cur.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'embeddings'
              AND indexname = 'embeddings_embedding_hnsw'
            """
        )
        row = cur.fetchone()

    assert row is not None
    assert "vector_l2_ops" in row[0]


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_rebuild_rag_index_missing_embeddings_table(settings) -> None:
    _assert_env_matches_default_database(settings)
    vector_client.reset_default_client()

    with connection.cursor() as cur:
        cur.execute("SET search_path TO rag, public")
        cur.execute("DROP TABLE IF EXISTS embeddings CASCADE")

    call_command("rebuild_rag_index")

    with connection.cursor() as cur:
        cur.execute("SET search_path TO rag, public")
        cur.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'embeddings'
              AND indexname IN ('embeddings_embedding_hnsw', 'embeddings_embedding_ivfflat')
            """
        )
        assert cur.fetchone() is not None, "expected embeddings indexes to be recreated"


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_rebuild_rag_index_health_check(
    client,
    monkeypatch,
    tmp_path,
    settings,
    test_tenant_schema_name,
) -> None:
    tenant = test_tenant_schema_name
    case = "case-index-health"
    store_path = tmp_path / "object-store"
    monkeypatch.setattr(object_store, "BASE_PATH", store_path)
    monkeypatch.setattr(rate_limit, "check", lambda *_args, **_kwargs: True)

    def _upload(content: str, external_id: str) -> str:
        upload = SimpleUploadedFile(
            f"{external_id}.txt", content.encode("utf-8"), content_type="text/plain"
        )
        payload = {
            "file": upload,
            "metadata": json.dumps({"external_id": external_id}),
        }
        response = client.post(
            "/ai/rag/documents/upload/",
            data=payload,
            **{
                META_TENANT_SCHEMA_KEY: tenant,
                META_TENANT_ID_KEY: tenant,
                META_CASE_ID_KEY: case,
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["external_id"] == external_id
        return body["document_id"]

    doc_one = _upload("Vector index smoke check one", "index-health-one")
    doc_two = _upload("Vector index smoke check two", "index-health-two")

    first_result = process_document(tenant, case, doc_one, tenant_schema=tenant)
    second_result = process_document(tenant, case, doc_two, tenant_schema=tenant)

    assert first_result["inserted"] == 1
    assert second_result["inserted"] == 1

    call_command("rebuild_rag_index")

    index_kind = str(getattr(settings, "RAG_INDEX_KIND", "HNSW")).upper()
    expected_index = (
        "embeddings_embedding_hnsw"
        if index_kind == "HNSW"
        else "embeddings_embedding_ivfflat"
    )

    client_handle = vector_client.get_default_client()
    schema_name = getattr(client_handle, "_schema", "rag")

    with connection.cursor() as cur:
        cur.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema_name))
        )
        cur.execute(
            """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND tablename = 'embeddings'
              AND indexname = %s
            """,
            (expected_index,),
        )
        row = cur.fetchone()
    assert row is not None, f"expected index {expected_index} to be present"

    results = client_handle.search(
        "Vector index smoke check",
        tenant_id=tenant,
        top_k=2,
        filters={"case": case},
    )

    assert len(results) == 2
    ids = {chunk.meta.get("external_id") for chunk in results}
    assert ids == {"index-health-one", "index-health-two"}
