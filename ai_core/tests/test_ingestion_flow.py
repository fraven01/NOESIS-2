import json
import uuid
from pathlib import Path
from uuid import UUID

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from psycopg2 import sql
from types import SimpleNamespace

from ai_core import ingestion
from ai_core.ingestion import process_document
from ai_core.infra import object_store, rate_limit
from ai_core.rag import vector_client as rag_vector_client
from ai_core.views import make_fallback_external_id
from common.constants import (
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
    META_CASE_ID_KEY,
)


@pytest.mark.django_db
def test_upload_ingest_query_end2end(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
    rag_database,
):
    tenant = test_tenant_schema_name
    http_client = client

    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)
    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=lambda *a, **k: None)
    )
    monkeypatch.setattr(ingestion, "_cleanup_artifacts", lambda paths: [])

    # Upload
    upload = SimpleUploadedFile(
        "note.txt",
        b"# Heading\n\nhello ZEBRAGURKE world",
        content_type="text/markdown",
    )
    payload = {
        "file": upload,
        "metadata": json.dumps({"label": "e2e", "external_id": "doc-e2e"}),
    }

    resp = http_client.post(
        "/ai/rag/documents/upload/",
        data=payload,
        **{
            META_TENANT_SCHEMA_KEY: tenant,
            META_TENANT_ID_KEY: tenant,
            # META_CASE_ID_KEY: "",  # Caseless mode
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["workflow_id"]  # Generated UUID
    doc_id = body["document_id"]
    trace_id = body["trace_id"]

    tenant_segment = object_store.sanitize_identifier(tenant)
    case_segment = object_store.sanitize_identifier("upload")
    metadata_path = Path(
        tmp_path, tenant_segment, case_segment, "uploads", f"{doc_id}.meta.json"
    )
    stored_metadata = json.loads(metadata_path.read_text())
    assert stored_metadata["external_id"] == "doc-e2e"

    # Ingestion (direkt Task ausführen; alternativ run_ingestion.delay(...) und warten)
    result = process_document(
        tenant,
        "upload",
        doc_id,
        "standard",
        tenant_schema=tenant,
        trace_id=trace_id,
    )
    assert result["written"] >= 1
    assert result["embedding_profile"] == "standard"
    assert result["vector_space_id"] == "global"

    parsed_path = ingestion._parsed_blocks_path(tenant, "upload", UUID(doc_id))
    parsed_payload = object_store.read_json(parsed_path)
    blocks = parsed_payload.get("blocks", [])
    assert blocks, "expected parsed blocks to be persisted"
    first_block = blocks[0]
    assert first_block.get("kind") == "heading"
    assert first_block.get("section_path")
    assert first_block.get("page_index") is None

    status_state = object_store.read_json(
        ingestion._status_store_path(tenant, "upload", doc_id)
    )
    chunk_step = status_state["steps"]["chunk"]
    chunk_path = chunk_step["path"]
    chunk_payload = object_store.read_json(chunk_path)
    zebra_chunk = next(
        (
            entry
            for entry in chunk_payload.get("chunks", [])
            if "ZEBRAGURKE" in entry["content"]
        ),
        None,
    )
    assert zebra_chunk is not None
    parent_ids = zebra_chunk["meta"].get("parent_ids", [])
    assert f"{doc_id}#doc" in parent_ids
    assert any(parent_id.endswith("#sec-1") for parent_id in parent_ids)
    assert zebra_chunk["meta"]["document_id"] == doc_id

    vector_client = rag_vector_client.get_default_client()
    documents_table = vector_client._table("documents")
    chunks_table = vector_client._table("chunks")
    with vector_client._connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id::text, metadata FROM {} WHERE id = %s").format(
                    documents_table
                ),
                (doc_id,),
            )
            stored = cur.fetchone()
            assert stored is not None
            stored_id, metadata = stored
            assert str(stored_id) == doc_id
            parent_nodes = dict((metadata or {}).get("parent_nodes") or {})
            assert f"{doc_id}#doc" in parent_nodes
            root_parent = parent_nodes[f"{doc_id}#doc"]
            assert isinstance(root_parent, dict)
            assert root_parent.get("document_id") == doc_id

            cur.execute(
                sql.SQL("SELECT metadata FROM {} WHERE document_id = %s").format(
                    chunks_table
                ),
                (doc_id,),
            )
            chunk_rows = cur.fetchall()
    assert chunk_rows
    for (chunk_meta,) in chunk_rows:
        assert isinstance(chunk_meta, dict)
        assert chunk_meta.get("document_id") == doc_id

    # Query (Demo-Endpunkt wurde entfernt und meldet HTTP 410)
    resp = http_client.post(
        "/ai/v1/rag-demo/",
        data=json.dumps({"query": "zebragurke", "top_k": 3}),
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: tenant,
            META_TENANT_ID_KEY: tenant,
            # META_CASE_ID_KEY: case,
        },
    )
    assert resp.status_code == 410
    data = resp.json()
    assert data["code"] == "rag_demo_removed"
    assert (
        data["detail"]
        == "The RAG demo endpoint is deprecated and no longer available in the MVP build."
    )


@pytest.mark.django_db
def test_ingestion_run_reports_missing_documents(
    client,
    monkeypatch,
    tmp_path,
    test_tenant_schema_name,
    rag_database,
):
    tenant = test_tenant_schema_name

    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    monkeypatch.setattr(
        "ai_core.views.run_ingestion", SimpleNamespace(delay=lambda *a, **k: None)
    )

    upload = SimpleUploadedFile(
        "note.txt", b"hello ZEBRAGURKE world", content_type="text/plain"
    )
    payload = {
        "file": upload,
        "metadata": json.dumps({"label": "e2e", "external_id": "doc-e2e"}),
    }

    resp = client.post(
        "/ai/rag/documents/upload/",
        data=payload,
        **{
            META_TENANT_SCHEMA_KEY: tenant,
            META_TENANT_ID_KEY: tenant,
            META_CASE_ID_KEY: "upload",
        },
    )
    assert resp.status_code == 202
    doc_id = resp.json()["document_id"]

    calls: list[tuple[tuple, dict]] = []

    class DummyTask:
        def delay(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr("ai_core.views.run_ingestion", DummyTask())

    missing_id = str(uuid.uuid4())
    # Simuliere Nutzereingaben mit führenden/nachgestellten Leerzeichen –
    # der Service trimmt diese vor dem Task-Dispatch.
    run_payload = {
        "document_ids": [f"  {doc_id}  ", f"  {missing_id}  "],
        "priority": "normal",
        "embedding_profile": "standard",
    }

    resp = client.post(
        "/ai/rag/ingestion/run/",
        data=json.dumps(run_payload),
        content_type="application/json",
        **{
            META_TENANT_SCHEMA_KEY: tenant,
            META_TENANT_ID_KEY: tenant,
            META_CASE_ID_KEY: "upload",
        },
    )

    assert resp.status_code == 202
    body = resp.json()
    # Erwartet: normalisierte (getrimmte) UUID in der Invalid-Liste
    assert body["invalid_ids"] == [missing_id]

    assert len(calls) == 1
    args, kwargs = calls[0]
    # Erwartet: Task erhält getrimmte document_ids
    assert list(args[2]) == [doc_id]
    assert args[3] == "standard"
    assert kwargs["tenant_schema"] == tenant


def test_fallback_external_id_consistency():
    result = make_fallback_external_id("note.txt", 11, b"hello world")
    assert result == "730b11756bd5a6af33f1ee8c07433a1042d6626af49ba4296d1170f0fdd71eff"
