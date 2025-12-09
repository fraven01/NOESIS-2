import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from documents.models import DocumentIngestionRun
from documents.repository import (
    ACTIVE_STATE,
    RETIRED_STATE,
    DocumentLifecycleStore,
    PersistentDocumentLifecycleStore,
)


def test_memory_store_persists_across_instances():
    store_a = DocumentLifecycleStore()
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

    fetched = store_a.get_document_state(
        tenant_id="tenant-a", document_id=document_id, workflow_id=None
    )

    assert fetched is not None
    assert fetched.state == ACTIVE_STATE
    assert fetched.reason == "ingested"
    assert fetched.changed_at == created.changed_at


def test_memory_store_allows_concurrent_mutations():
    store = DocumentLifecycleStore()
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

    def _persist_transition(target_state: str, reason: str):
        return store.record_document_state(
            tenant_id="tenant-concurrent",
            document_id=document_id,
            workflow_id=workflow_id,
            state=target_state,
            reason=reason,
        )

    def _transition(target_state: str, reason: str) -> None:
        barrier.wait()
        record = _persist_transition(target_state, reason)
        results.append(record.state)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_transition, RETIRED_STATE, "retired"),
            executor.submit(_transition, ACTIVE_STATE, "reactivated"),
        ]
        for future in futures:
            future.result()

    final = store.get_document_state(
        tenant_id="tenant-concurrent",
        document_id=document_id,
        workflow_id=workflow_id,
    )

    assert len(results) == 2
    assert final is not None
    assert final.state in {ACTIVE_STATE, RETIRED_STATE}
    assert final.reason in {"retired", "reactivated"}


def test_memory_store_persists_ingestion_runs():
    store = DocumentLifecycleStore()
    tenant_id = "tenant-ingestion"
    case = "case-1"
    run_id = "run-123"
    document_id = "test_document_id"
    queued = store.record_ingestion_run_queued(
        tenant_id=tenant_id,
        case=case,
        run_id=run_id,
        document_ids=[document_id],
        invalid_document_ids=[],
        queued_at="2024-01-01T00:00:00Z",
    )
    assert queued["document_ids"] == [document_id]
    assert queued["tenant_id"] == tenant_id
    assert queued["status"] == "queued"
    running = store.mark_ingestion_run_running(
        tenant_id=tenant_id,
        case=case,
        run_id=run_id,
        started_at="2024-01-01T00:01:00Z",
        document_ids=["doc-1", "doc-2", "doc-3"],
    )

    assert running is not None
    assert running["status"] == "running"
    assert running["document_ids"] == ["doc-1", "doc-2", "doc-3"]

    completed = store.mark_ingestion_run_completed(
        tenant_id=tenant_id,
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


@pytest.mark.django_db
def test_persistent_store_persists_ingestion_runs():
    store = PersistentDocumentLifecycleStore()
    tenant_id = "tenant-persistent"
    case = "case-42"
    run_id = "run-persist"

    queued = store.record_ingestion_run_queued(
        tenant_id=tenant_id,
        case=case,
        run_id=run_id,
        document_ids=["doc-a"],
        invalid_document_ids=["bad-a"],
        queued_at="2024-01-01T00:00:00Z",
        trace_id="trace-1",
        collection_id="collection-1",
    )

    db_run = DocumentIngestionRun.objects.get(tenant_id=tenant_id, case=case)
    assert db_run.status == "queued"
    assert queued["status"] == "queued"
    assert db_run.document_ids == ["doc-a"]
    assert db_run.collection_id == "collection-1"

    running = store.mark_ingestion_run_running(
        tenant_id=tenant_id,
        case=case,
        run_id=run_id,
        started_at="2024-01-01T00:01:00Z",
        document_ids=["doc-a", "doc-b"],
    )

    db_run.refresh_from_db()
    assert db_run.status == "running"
    assert running is not None
    assert running["status"] == "running"
    assert running["document_ids"] == ["doc-a", "doc-b"]

    completed = store.mark_ingestion_run_completed(
        tenant_id=tenant_id,
        case=case,
        run_id=run_id,
        finished_at="2024-01-01T00:05:00Z",
        duration_ms=120000.0,
        inserted_documents=2,
        replaced_documents=1,
        skipped_documents=0,
        inserted_chunks=5,
        invalid_document_ids=["bad-a"],
        document_ids=["doc-a", "doc-b"],
        error=None,
    )

    db_run.refresh_from_db()
    assert db_run.status == "succeeded"
    assert db_run.invalid_document_ids == ["bad-a"]
    assert completed is not None
    assert completed["status"] == "succeeded"
    assert completed["invalid_document_ids"] == ["bad-a"]

    fetched = store.get_ingestion_run(tenant_id=tenant_id, case=case)
    assert fetched is not None
    assert fetched["status"] == "succeeded"
