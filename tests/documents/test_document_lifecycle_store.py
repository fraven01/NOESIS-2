import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from django.db import connection
from django_tenants.utils import schema_context

from documents.repository import ACTIVE_STATE, RETIRED_STATE, PersistentDocumentLifecycleStore


@pytest.mark.django_db
def test_persistent_store_persists_across_instances():
    store_a = PersistentDocumentLifecycleStore()
    store_b = PersistentDocumentLifecycleStore()
    document_id = uuid4()

    timestamp = datetime.now(timezone.utc)
    created = store_a.record_document_state(
        tenant_id="tenant-a",
        document_id=document_id,
        workflow_id=None,
        state=ACTIVE_STATE,
        reason="ingested",
        changed_at=timestamp,
    )

    fetched = store_b.get_document_state(
        tenant_id="tenant-a", document_id=document_id, workflow_id=None
    )

    assert fetched is not None
    assert fetched.state == ACTIVE_STATE
    assert fetched.reason == "ingested"
    assert fetched.changed_at == created.changed_at


@pytest.mark.django_db(transaction=True)
def test_persistent_store_allows_concurrent_mutations():
    store = PersistentDocumentLifecycleStore()
    document_id = uuid4()
    workflow_id = "wf-1"

    store.record_document_state(
        tenant_id="tenant-concurrent",
        document_id=document_id,
        workflow_id=workflow_id,
        state=ACTIVE_STATE,
        reason="initial",
    )

    barrier = threading.Barrier(2)
    results: list[str] = []
    active_schema = getattr(connection, "schema_name", "public")

    def _transition(target_state: str, reason: str) -> None:
        barrier.wait()
        conn = connection
        conn.ensure_connection()
        try:
            with schema_context(active_schema):
                record = _persist_transition(target_state, reason)
            results.append(record.state)
        finally:
            conn.close()

    def _persist_transition(target_state: str, reason: str):
        local_store = PersistentDocumentLifecycleStore()
        return local_store.record_document_state(
            tenant_id="tenant-concurrent",
            document_id=document_id,
            workflow_id=workflow_id,
            state=target_state,
            reason=reason,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_transition, RETIRED_STATE, "retired"),
            executor.submit(_transition, ACTIVE_STATE, "reactivated"),
        ]
        for future in futures:
            future.result()

    final = PersistentDocumentLifecycleStore().get_document_state(
        tenant_id="tenant-concurrent",
        document_id=document_id,
        workflow_id=workflow_id,
    )

    assert len(results) == 2
    assert final is not None
    assert final.state in {ACTIVE_STATE, RETIRED_STATE}
    assert final.reason in {"retired", "reactivated"}


@pytest.mark.django_db
def test_persistent_store_persists_ingestion_runs():
    store = PersistentDocumentLifecycleStore()
    tenant = "tenant-ingestion"
    case = "case-1"
    run_id = "run-123"

    queued = store.record_ingestion_run_queued(
        tenant=tenant,
        case=case,
        run_id=run_id,
        document_ids=["doc-1", "doc-2"],
        invalid_document_ids=["bad-1"],
        queued_at="2024-01-01T00:00:00Z",
        trace_id="trace-1",
        embedding_profile="profile-a",
        source="crawler",
    )

    assert queued["status"] == "queued"
    assert queued["document_ids"] == ["doc-1", "doc-2"]

    running = PersistentDocumentLifecycleStore().mark_ingestion_run_running(
        tenant=tenant,
        case=case,
        run_id=run_id,
        started_at="2024-01-01T00:01:00Z",
        document_ids=["doc-1", "doc-2", "doc-3"],
    )

    assert running is not None
    assert running["status"] == "running"
    assert running["document_ids"] == ["doc-1", "doc-2", "doc-3"]

    completed = PersistentDocumentLifecycleStore().mark_ingestion_run_completed(
        tenant=tenant,
        case=case,
        run_id=run_id,
        finished_at="2024-01-01T00:05:00Z",
        duration_ms=120000.0,
        inserted_documents=2,
        replaced_documents=1,
        skipped_documents=0,
        inserted_chunks=5,
        invalid_document_ids=["bad-1"],
        document_ids=["doc-1", "doc-2", "doc-3"],
        error=None,
    )

    assert completed is not None
    assert completed["status"] == "succeeded"
    assert completed["invalid_document_ids"] == ["bad-1"]
    assert completed["document_ids"] == ["doc-1", "doc-2", "doc-3"]
