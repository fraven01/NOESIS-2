"""Tests for the reembed_documents management command."""

from __future__ import annotations

from typing import Any

import pytest
from django.core.management import call_command
from django_tenants.utils import schema_context

from ai_core.rag.embedding_config import reset_embedding_configuration_cache
from customers.models import Tenant
from documents.models import Document


@pytest.fixture(autouse=True)
def _reset_embedding_cache():
    reset_embedding_configuration_cache()
    yield
    reset_embedding_configuration_cache()


@pytest.mark.django_db
def test_reembed_documents_queues_tasks(settings, test_tenant_schema_name, monkeypatch):
    tenant = Tenant.objects.get(schema_name=test_tenant_schema_name)

    settings.RAG_VECTOR_STORES = {
        "global": {"backend": "pgvector", "schema": "rag", "dimension": 16}
    }
    settings.RAG_EMBEDDING_PROFILES = {
        "standard": {
            "model": "oai-embed",
            "dimension": 16,
            "vector_space": "global",
            "chunk_hard_limit": 512,
        }
    }
    settings.RAG_DEFAULT_EMBEDDING_PROFILE = "standard"

    with schema_context(tenant.schema_name):
        doc_one = Document.objects.create(
            tenant=tenant,
            hash="hash-one",
            source="upload",
            case_id="case-1",
            trace_id="trace-1",
        )
        doc_two = Document.objects.create(
            tenant=tenant,
            hash="hash-two",
            source="upload",
            case_id="",
            trace_id="",
        )

    import ai_core.management.commands.reembed_documents as reembed_documents

    calls: list[dict[str, Any]] = []

    def fake_apply_async(*, kwargs, queue, countdown):
        calls.append({"kwargs": kwargs, "queue": queue, "countdown": countdown})

    monkeypatch.setattr(
        reembed_documents.process_document, "apply_async", fake_apply_async
    )
    monkeypatch.setattr(
        reembed_documents,
        "_fetch_chunk_counts",
        lambda *_args, **_kwargs: {doc_one.id: 3, doc_two.id: 0},
    )
    monkeypatch.setattr(
        reembed_documents, "reserve_reembed_delay", lambda *_args, **_kwargs: 0.0
    )
    monkeypatch.setattr(
        reembed_documents, "ensure_vector_space_schema", lambda *_args, **_kwargs: None
    )

    progress_payloads: list[dict[str, Any]] = []

    def fake_init_progress(*_args, **kwargs):
        progress_payloads.append(kwargs)

    monkeypatch.setattr(reembed_documents, "init_reembed_progress", fake_init_progress)
    monkeypatch.setattr(
        reembed_documents, "increment_reembed_progress", lambda *_args, **_kwargs: None
    )

    call_command(
        "reembed_documents",
        tenant=test_tenant_schema_name,
        embedding_profile="standard",
    )

    assert len(calls) == 2
    assert all(call["queue"] == "ingestion-bulk" for call in calls)

    call_by_document = {
        call["kwargs"]["state"]["document_id"]: call["kwargs"] for call in calls
    }
    kwargs_one = call_by_document[str(doc_one.id)]
    kwargs_two = call_by_document[str(doc_two.id)]

    state_one = kwargs_one["state"]
    state_two = kwargs_two["state"]

    assert state_one["tenant_id"] == tenant.schema_name
    assert state_one["tenant_schema"] == tenant.schema_name
    assert state_one["embedding_profile"] == "standard"

    assert state_two["tenant_id"] == tenant.schema_name
    assert state_two["tenant_schema"] == tenant.schema_name

    progress_key = kwargs_one["reembed_progress_key"]
    assert progress_key == kwargs_two["reembed_progress_key"]

    assert state_one["document_id"] == str(doc_one.id)
    assert state_one["case_id"] == "case-1"
    assert state_one["trace_id"] == "trace-1"

    assert state_two["document_id"] == str(doc_two.id)
    assert state_two["case_id"] is None
    assert state_two["trace_id"] is None

    meta_one = kwargs_one["meta"]
    assert meta_one["scope_context"]["tenant_id"] == tenant.schema_name

    assert progress_payloads
    init_payload = progress_payloads[0]
    assert init_payload["total_documents"] == 2
    assert init_payload["total_chunks"] == 4
