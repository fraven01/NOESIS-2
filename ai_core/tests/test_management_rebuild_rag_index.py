from __future__ import annotations

import io
import json
import re

import pytest
from psycopg2 import sql
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.core.files.uploadedfile import SimpleUploadedFile

from ai_core.ingestion import process_document
from ai_core.rag import vector_client
from ai_core.infra import object_store, rate_limit
from tests.plugins.rag_db import SCHEMA_SQL
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)


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
        cur.execute("DROP SCHEMA IF EXISTS rag_enterprise CASCADE")
        cur.execute("CREATE SCHEMA rag_enterprise")
        cur.execute("SET search_path TO rag_enterprise, public")
        cur.execute(SCHEMA_SQL)

    try:
        stdout = io.StringIO()
        call_command("rebuild_rag_index", stdout=stdout)

        with connection.cursor() as cur:
            cur.execute("SET search_path TO rag_enterprise, public")
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
            cur.execute("DROP SCHEMA IF EXISTS rag_enterprise CASCADE")


@pytest.mark.django_db
@pytest.mark.usefixtures("rag_database")
def test_rebuild_rag_index_missing_embeddings_table() -> None:
    with connection.cursor() as cur:
        cur.execute("SET search_path TO rag, public")
        cur.execute("DROP TABLE IF EXISTS embeddings CASCADE")

    with pytest.raises(CommandError) as excinfo:
        call_command("rebuild_rag_index")

    assert (
        "Vector table 'embeddings' not found; ensure the RAG schema is initialised"
        == str(excinfo.value)
    )


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
