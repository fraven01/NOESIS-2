import pytest
from uuid import uuid4
from django.apps import apps
from django_tenants.utils import schema_context
from documents.repository import PersistentDocumentLifecycleStore


@pytest.mark.slow
@pytest.mark.django_db
@pytest.mark.xdist_group("tenant_ops")
def test_lifecycle_persistence_end_to_end(tenant, transactional_db):
    """Verify that lifecycle changes are persisted to the Document model."""

    Document = apps.get_model("documents", "Document")
    store = PersistentDocumentLifecycleStore()

    with schema_context(tenant.schema_name):
        # 1. Setup Document
        doc_id = uuid4()
        doc = Document.objects.create(
            id=doc_id,
            tenant=tenant,
            hash="test-hash",
            source="test-source",
            lifecycle_state="pending",
        )

        # 2. Record State (Active)
        record = store.record_document_state(
            tenant_id=tenant.schema_name,
            document_id=doc_id,
            workflow_id=None,
            state="active",
            reason="ingestion_complete",
            policy_events=["policy_a"],
        )

        assert record.state == "active"

        # 3. Verify Persistence (Refresh from DB)
        doc.refresh_from_db()
        assert doc.lifecycle_state == "active"
        assert doc.lifecycle_updated_at is not None

        meta = doc.metadata.get("lifecycle")
        assert meta is not None
        assert meta["state"] == "active"
        assert meta["reason"] == "ingestion_complete"
        assert meta["policy_events"] == ["policy_a"]

        # 4. Verify Get State (Cache Hit)
        fetched = store.get_document_state(
            tenant_id=tenant.schema_name, document_id=doc_id, workflow_id=None
        )
        assert fetched.state == "active"
        assert fetched.reason == "ingestion_complete"

        # 5. Verify Get State (Cache Miss / Cold Start)
        # Clear cache manually
        store._documents.clear()

        fetched_cold = store.get_document_state(
            tenant_id=tenant.schema_name, document_id=doc_id, workflow_id=None
        )
        assert fetched_cold is not None
        assert fetched_cold.state == "active"
        assert fetched_cold.reason == "ingestion_complete"
        assert fetched_cold.policy_events == ("policy_a",)

        # 6. Test Transition to Retired
        store.record_document_state(
            tenant_id=tenant.schema_name,
            document_id=doc_id,
            workflow_id=None,
            state="retired",
            reason="ttl_expired",
        )

        doc.refresh_from_db()
        assert doc.lifecycle_state == "retired"
        assert doc.metadata["lifecycle"]["reason"] == "ttl_expired"
